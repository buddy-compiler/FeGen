from FeGen import *
from FeGen.AttributeGrammar.Rule import Rule, ParserRule, TerminalRule, Production, OneOrMore, ZeroOrMore, ZeroOrOne, Alternate, Concat, AttrError
from typing import Dict, List, Set

from xdsl.dialects import builtin, func, arith, tosa
from xdsl.printer import Printer

from dialects import toy

from xdsl.builder import Builder, InsertPoint


class ToyError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class ToyVar:
    def __init__(self, name, ty, mlirvalue):
        self.name = name
        self.ty = ty
        self.mlirvalue = mlirvalue
        
class ToyType:
    def __init__(self, name, mlirty):
        self.name = name
        self.mlirty = mlirty

class ToyUnrankedTensorType(ToyType):
    def __init__(self):
        super().__init__("UnrankedTensorType", toy.UnrankedTensorTypeF64(builtin.f64))

    def __eq__(self, value):
        if isinstance(value, ToyUnrankedTensorType):
            return True
        return False

class ToyRankedTensorType(ToyType):
    def __init__(self, shape):
        super().__init__("RankedTensorType", toy.TensorType(builtin.f64, shape))
        self.shape: List[int] = shape

    def __eq__(self, value):
        if not isinstance(value, ToyRankedTensorType):
            return False
        return self.shape == value.shape
        
class ToyFunction:
    def __init__(self, name, params, mlirvalue):
        self.name = name
        self.params : Dict[str, ToyType] = params
        self.mlirvalue: toy.FuncOp = mlirvalue
        
        
class Scope:
    def __init__(self):
        self.var_table : Dict[str, ToyVar] = {}
        
    def loopup(self, name):
        return self.var_table.get(name)
    
    def insert(self, var: ToyVar):
        self.var_table.update({var.name: var})

class Visitor:
    def __init__(self):
        self.globalscope = Scope()
        self.scopestack = [self.globalscope]
        self.funcmap : Dict[str, ToyFunction] = {}
        self.typemap : Dict[str, ToyType] = {}
        
        self.module = builtin.ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))
        
    @property
    def current_scope(self):
        return self.scopestack[-1]
        
    def push_scope(self):
        self.scopestack.append(Scope())
    
    def pop_scope(self):
        self.scopestack.pop(-1)
    
    def getUnrankedTensorType(self):
        return ToyUnrankedTensorType()
    
    def getRankedTensorType(self, shape):
        return ToyRankedTensorType(shape)
    
    @sema
    def visit(self, g: Rule, *args, **kwds):
        rulename = g.name
        visitor_name = f"visit_{rulename}"
        visitor_func = getattr(self, visitor_name, None)
        if visitor_func is None:
            raise NotImplementedError(f"Can not find visit function named {visitor_name}")
        else:
            visitor_func(g, *args, **kwds)
      
    def visit_module(self, g: ParserRule, g_func_defs: OneOrMore):
        for g_func in g_func_defs:
            # visit
            self.push_scope()
            # get attr func_inst
            func_obj: ToyFunction = g_func.get_attr("func_inst")
            func_name = func_obj.name
            self.pop_scope()
            self.funcmap.update({func_name: func_obj})
            # verify module
            try:
                self.module.verify()
            except Exception:
                print("module verification error")
                raise
    
    def visit_funcDefine(self, g: ParserRule, g_proto: ParserRule, g_block: ParserRule):
        # get attr func_inst
        func_obj: ToyFunction = g_proto.get_attr("func_inst")
        # update scope
        param_names = list(func_obj.params.keys())
        param_types = list(func_obj.params.values())
        param_args = func_obj.mlirvalue.body.block.args
        for p_name, p_type, p_mlirvalue in zip(param_names, param_types, param_args):
            self.current_scope.insert(ToyVar(p_name, p_type, p_mlirvalue))
        # ser insert point to function
        self.builder.insert(func_obj.mlirvalue)
        parent_builder = self.builder
        self.builder = Builder(InsertPoint.at_end(func_obj.mlirvalue.body.block))
        # get attr ops
        g_block.visit()
        # Implicitly return void if no return statement was emitted.
        return_op = None
        block = func_obj.mlirvalue.body.block
        if block.ops:
            last_op = block.last_op
            if isinstance(last_op, toy.ReturnOp):
                return_op = last_op
                if return_op.input is not None:
                    return_arg = return_op.input
                    return_types = [return_arg.type]
                    input_types = func_obj.mlirvalue.function_type.inputs
                    func_obj.mlirvalue.function_type = func.FunctionType.from_lists(input_types, return_types)
        if return_op is None:
            self.builder.insert(toy.ReturnOp())
        # restore builder
        self.builder = parent_builder
        # set attr func_inst
        g.set_attr("func_inst", func_obj)
    
    def visit_prototype(self, g: ParserRule, g_id: TerminalRule, g_decl_List: ZeroOrOne):
        name = g_id.getText()
        params = {}
        func_inputtys = []
        if g_decl_List.exist():
            # get attr params
            params: Dict[str, ToyType] = g_decl_List.get_attr("params")
            # get mlir function params
            func_inputtys = [paramty.mlirty for paramty in params.values()]
        func_mlirty = func.FunctionType.from_lists(func_inputtys, [])
        private = (not name == "main")
        func_mlirvalue = toy.FuncOp(name, func_mlirty, private=private)
        func_inst = ToyFunction(name, params, func_mlirvalue)
        # set attr func_inst
        g.set_attr("func_inst", func_inst)

    def visit_params(self, g: ParserRule, g_first: ParserRule, g_other: ZeroOrMore):
        g_params = [g_first]
        for comma_param in g_other:
            g_params.append(comma_param[1])
        params = {}
        for g_param in g_params:
            name = g_param.get_attr("name")
            ty = g_param.get_attr("type")
            params.update({name: ty})
        g.set_attr("params", params)
    
    
    def visit_param(self, g: ParserRule, g_alt: Alternate):
        actual_alt = g_alt.get_actual_alt()
        if g_alt.get_actual_alt_index() == 0:
            assert isinstance(actual_alt, TerminalRule)
            name = actual_alt.getText()
            ty = self.getUnrankedTensorType()
        else:
            assert isinstance(actual_alt, Concat)
            ty_name = actual_alt[0].getText()
            ty = self.typemap.get(ty_name)
            if ty is None:
                raise ToyError(f"undefined type: {ty_name}")
            name = actual_alt[1].getText()
        # set attr name and type
        g.set_attr("name", name)
        g.set_attr("type", ty)
    
    
    def visit_block(self, g: ParserRule, g_blockexprs: ZeroOrMore):
        for g_blockexpr_semi in g_blockexprs:
            assert isinstance(g_blockexpr_semi, Concat)
            g_blockexpr = g_blockexpr_semi[0]
            assert isinstance(g_blockexpr, ParserRule)
            g_blockexpr.visit()
    
    
    def visit_blockExpr(self, g: ParserRule, g_alt: Alternate):
        g_alt.get_actual_alt().visit()
    
    
    def visit_varDecl(self, g: ParserRule, g_alt: Alternate):
        g_alt.get_actual_alt().visit()
    
    def visit_tensorVarDecl(self, g: ParserRule, g_id: TerminalRule, g_type: ParserRule, g_expr: ParserRule):
        name = g_id.getText()
        value = g_expr.get_attr("value")
        ty = g_expr.get_attr("type")
        if g_type.exist():
            ty: ToyType = g_type.get_attr("type")
            assert isinstance(ty, ToyRankedTensorType)
            reshapeop = toy.ReshapeOp(value, ty.shape)
            self.builder.insert(reshapeop)
            value = reshapeop.res
        var = ToyVar(name, ty, value)
        self.current_scope.insert(var)
        
        
    def visit_type(self, g: ParserRule, g_first_num: TerminalRule, g_other_num: ZeroOrMore):
        nums = []
        nums.append(int(g_first_num.getText()))
        for comma_num in g_other_num:
            nums.append(int(comma_num[1].getText()))
        ty = self.getRankedTensorType(nums)
        g.set_attr("type", ty)
        
    
    def visit_structVarDecl(self, g):
        raise NotImplementedError()
    
    
    def visit_returnExpr(self, g: ParserRule, g_alt: Alternate):
        if g_alt.get_actual_alt_index() == 0:
            self.builder.insert(toy.ReturnOp())
        else:
            ret = g_alt.get_actual_alt()[1].get_attr("value")
            self.builder.insert(toy.ReturnOp(ret))
    
    def visit_expression(self, g: ParserRule, g_add: ParserRule):
        g.set_attr("value", g_add.get_attr("value"))
        g.set_attr("type", g_add.get_attr("type"))
        
    
    def visit_add_expr(self, g: ParserRule, g_alt: Alternate):
        if g_alt.get_actual_alt_index() == 0:
            g.set_attr("value", g_alt.get_attr("value"))
            g.set_attr("type", g_alt.get_attr("type"))
        else:
            actual_alt = g_alt.get_actual_alt()
            assert isinstance(actual_alt, Concat)
            g_lhs = actual_alt[0]
            g_rhs = actual_alt[2]
            lhs = g_lhs.get_attr("value")
            rhs = g_rhs.get_attr("value")
            addop = toy.AddOp(lhs, rhs)
            self.builder.insert(addop)
            g.set_attr("value", addop.res)
            g.set_attr("type", g_lhs.get_attr("type"))
    
    
    def visit_term_expr(self, g: ParserRule, g_alt: Alternate):
        if g_alt.get_actual_alt_index() == 0:
            g.set_attr("value", g_alt.get_attr("value"))
            g.set_attr("type", g_alt.get_attr("type"))
        else:
            actual_alt = g_alt.get_actual_alt()
            assert isinstance(actual_alt, Concat)
            g_lhs = actual_alt[0]
            g_rhs = actual_alt[2]
            lhs = g_lhs.get_attr("value")
            rhs = g_rhs.get_attr("value")
            mulop = toy.MulOp(lhs, rhs)
            self.builder.insert(mulop)
            g.set_attr("value", mulop.res)
            g.set_attr("type", g_lhs.get_attr("type"))
            
    
    def visit_prim_expr(self, g: ParserRule, g_alt: Alternate):
        idx = g_alt.get_actual_alt_index()
        if idx == 0:
            num = int(g_alt.getText())
            ty = self.getRankedTensorType([1])
            value = self.builder.insert(toy.ConstantOp.from_list([num], []))
            g.set_attr("value", value.res)
            g.set_attr("type", ty)
        elif idx == 1:
            data = g_alt.get_attr("value")
            ty: ToyRankedTensorType = g_alt.get_attr("type")
            value = self.builder.insert(toy.ConstantOp.from_list(data, ty.shape))
            g.set_attr("value", value.res)
            g.set_attr("type", ty)
        elif idx in [1, 2, 3, 4]:
            g.set_attr("value", g_alt.get_attr("value"))
            g.set_attr("type", g_alt.get_attr("type"))
        else:
            assert False
    
    def visit_tensorLiteral(self, g: ParserRule, g_alt: Alternate):
        if g_alt.get_actual_alt_index() == 0:
            actual_alt = g_alt.get_actual_alt()
            assert isinstance(actual_alt, Concat)
            g_content = actual_alt[1]
            assert isinstance(g_content, ZeroOrOne)
            if g_content.exist():
                g_content = g_content.prod
                data = g_content.get_attr("value", True)
                isnum = False
                try:
                    type_list : List[ToyRankedTensorType] = g_content.get_attr("type", True)
                except AttrError:
                    isnum = True
                    
                if isnum:
                    dataty = self.getRankedTensorType([len(data)])
                    g.set_attr("value", data)
                    g.set_attr("type", dataty)
                else:
                    type_list: List[ToyRankedTensorType] = g_content.get_attr("type", True)
                    first_ty = type_list[0]
                    for ty in type_list:
                        if not first_ty == ty:
                            msg = "expect elements of tensor literal have the same shape: {}".format(g.getText())
                            raise ToyError(msg)
                    dataty = self.getRankedTensorType([len(data), *first_ty.shape])
                    # flatten data
                    flatten_data = []
                    for e in data:
                        flatten_data += e
                    g.set_attr("value", flatten_data)
                    g.set_attr("type", dataty)
            else:
                g.set_attr("value", [])
                g.set_attr("type", self.getRankedTensorType([]))
        else:
            value = int(g_alt.getText())
            g.set_attr("value", value)
            
    
    def visit_variable_access(self, g: ParserRule, g_alt):
        g.set_attr("value", g_alt.get_attr("value"))
        g.set_attr("type", g_alt.get_attr("type"))
        
    def visit_name_access(self, g: ParserRule):
        name = g.getText()
        var = self.current_scope.loopup(name)
        if var is None:
            raise ToyError(f"Undefined reference to {name}")
        g.set_attr("value", var.mlirvalue)
        g.set_attr("type", var.ty)
        
    def visit_func_call(self, g: ParserRule, g_id: TerminalRule, g_param: ZeroOrOne):
        func_name = g_id.getText()
        if func_name == "print":
            values = g_param.get_attr("value", True)
            assert len(values) == 1, f"`print` does not accept multiple arguments: {g.getText()}"
            self.builder.insert(toy.PrintOp(values[0]))
        elif func_name == "transpose":
            values = g_param.get_attr("value", True)
            assert len(values) == 1, "`transpose` does not accept multiple arguments."
            ty = g_param.get_attr("type", True)[0]
            trans = self.builder.insert(toy.TransposeOp(values[0]))
            if isinstance(ty, ToyRankedTensorType):
                shape = ty.shape
                ty = self.getRankedTensorType(shape.reverse())
            g.set_attr("value", trans.res)
            g.set_attr("type", ty)
        else:
            func_inst = self.funcmap.get(func_name)
            if func_inst is None:
                raise ToyError(f"undefined reference to function: `{func_name}`")
            callee = func_inst.name
            operands = []
            if g_param.exist():
                operands = g_param.get_attr("value", True)
            
            funccall = self.builder.insert(toy.GenericCallOp(callee, operands, [toy.UnrankedTensorTypeF64(builtin.f64)]))
            g.set_attr("value", funccall.res[0])
            g.set_attr("type", self.getUnrankedTensorType())