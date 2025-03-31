from FeGen import *
import logging
logging.basicConfig(level=logging.DEBUG)

from xdsl.dialects import func, builtin, arith
from xdsl.ir import Operation
from xdsl.printer import Printer

from typing import List, Dict, Tuple

class Type:
    def __init__(self, name, mlirty):
        self.name = name
        self.mlirty = mlirty

class IntegerType(Type):
    def __init__(self):
        super().__init__("int", builtin.i32)

class FloatType(Type):
    def __init__(self):
        super().__init__("float", builtin.f32)
        
class VoidType(Type):
    def __init__(self):
        super().__init__("void", None)

class Variable:
    def __init__(self, name, ty, value):
        self.name = name
        self.ty: Type = ty
        self.value: Operation = value


class SymbolTable:
    def __init__(self):
        self.varTable = dict()
        self.funcTable = dict()

    def insert_func(self, func: func.FuncOp):
        self.funcTable.update({func.name: func})
        
    def insert_var(self, var: Variable):
        self.varTable.update({var.name: var})

    def lookup_var(self, name):
        return self.varTable.get(name)


class SimpleC(FeGenGrammar):
    """
    module: funcDefine+
    funcDefine: prototype block
    prototype: type Identifier ParentheseOpen params ParentheseClose
    type: INT | FLOAT | VOID
    params: param (Comma param)*
    param: type Identifier
    block: BracketOpen (blockExpr Semicolon)* BracketClose
    blockExpr: varDecl | returnExpr | func_call 
    varDecl: type Identifier (Equal expression)?
    returnExpr: RETURN expression?
    func_call: Identifier ParentheseOpen func_call_params?  ParentheseClose
    func_call_params: expression (Comma expression)*
    expression: add_expr
    add_expr: term_expr ADD add_expr | term_expr SUB add_expr
    term_expr: prim_expr MUL term_expr | prim_expr DIV term_expr
    prim_expr: NUM | ID | ParentheseOpen expression ParentheseClose
    ParentheseOpen: "("
    ParentheseClose: ")"
    INT: "int"
    FLOAT: "float"
    VOID: "void"
    Comma: ","
    ADD: "+"
    SUB: "-"
    MUL: "*"
    DIV: "/"
    Identifier: "[a-zA-Z_][a-zA-Z0-9_]*"
    """
    def __init__(self, output_dir_name=".fegen"):
        super().__init__(output_dir_name)
        self.global_scope = SymbolTable()
        self.scopestack : List[SymbolTable] = [self.global_scope]    
    
    
    @property
    def current_scope(self):
        return self.scopestack[-1]


    def push_scope(self):
        self.scopestack.append(SymbolTable())
    
    
    def pop_scope(self):
        self.scopestack.pop()


    @sema
    def build_module(self, ops):
        themodule = builtin.ModuleOp(ops)
        return themodule
    
    @parser
    def module(self):
        g = newParserRule()
        g_funcdefs = one_or_more(self.funcDefine())
        g.setProduction(g_funcdefs)
        ops = []
        for g_funcdef in g_funcdefs:
            self.push_scope()
            funcop = g_funcdef.get_attr("funcop")
            ops.append(funcop)
            self.pop_scope()
        themodule = self.build_module(ops)
        g.set_attr("module", themodule)
        return g
    
    @parser
    def funcDefine(self):
        g = newParserRule()
        g_prototype = self.prototype()
        g_block = self.block()
        g.setProduction(g_prototype, g_block)

        funcop: func.FuncOp = g_prototype.get_attr("funcop")
        stmts: List[Operation] = g_block.get_attr("stmts")
        funcop.body.block.add_ops(stmts)
        g.set_attr("funcop", funcop)
        return g
    
    @parser
    def prototype(self):
        g = newParserRule()
        g_retty = self.type()
        g_name = self.Identifier()
        g_params = self.params()
        g.setProduction(concat(g_retty, g_name, self.ParentheseOpen(), zero_or_one(g_params), self.ParentheseClose()))
        # get function name 
        func_name = g_name.getText()
        # get function return type 
        retty: Type = g_retty.get_attr("type")
        if isinstance(retty, VoidType):
            func_ret_types = []
        else:
            func_ret_types = [retty.mlirty]
        # get function params
        if g_params.exist():
            func_params: List[Variable] = g_params.get_attr("params")
            mlir_param_types = []
            for func_param in func_params:
                # insert param to scope
                self.current_scope.insert_var(func_param)
                mlir_param_types.append(func_param.ty.mlirty)
        else:
            mlir_param_types = []
        functy = func.FunctionType.from_lists(mlir_param_types, func_ret_types)
        funcop = func.FuncOp(func_name, function_type=functy, visibility="private")
        if g_params.exist():
            assert len(funcop.args) == len(func_params)
            for func_param, func_arg in zip(func_params, funcop.args):
                func_param.value = func_arg
        g.set_attr("funcop", funcop)
        return g
    
    @parser
    def type(self):
        g = newParserRule()
        def INT():
            g.set_attr("type", IntegerType())
            return self.INT()
        
        def FLOAT():
            g.set_attr("type", FloatType())
            return self.FLOAT()
        
        def VOID():
            g.set_attr("type", VoidType())
            return self.VOID()
        
        g_alt = alternate(INT, FLOAT, VOID)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def params(self):
        g = newParserRule()
        g_first = self.param()
        g_other = zero_or_more(self.Comma(), self.param())
        # collect param types
        param_vars = []
        first_param = g_first.get_attr("param_vars")
        param_vars.append(first_param)
        for g_ in g_other:
            g_param = g_[1]
            param_vars.append(g_param.get_attr("param_vars"))
        g.set_attr("params", param_vars)
        g.setProduction(g_first, g_other)
        return g
    
    @parser
    def param(self):
        g = newParserRule()
        g_ty = self.type()
        g_id = self.Identifier()
        g.setProduction(g_ty, g_id)
        # set type of param
        ty = g_ty.get_attr("type")
        name = g_id.getText()
        var = Variable(name, ty, None)
        g.set_attr("param_vars", var)
        return g
    
    @parser
    def block(self):
        g = newParserRule()
        g_blockexprs = zero_or_more(self.blockExpr(), self.Semicolon())
        g.setProduction(self.BracketOpen(), g_blockexprs, self.BracketClose())
        # collect stmts
        stmts = []
        for g_blockexpr_semi in g_blockexprs:
            g_block = g_blockexpr_semi[0]
            stmts.append(g_block.get_attr("stmt")) 
        g.set_attr("stmts", stmts)
        return g
    
    @parser
    def blockExpr(self):
        g = newParserRule()
        def vardecl():
            g_decl = self.varDecl()
            g.set_attr("stmt", g_decl.get_attr("stmt"))
            return g_decl
        def retDecl():
            g_ret = self.returnExpr()
            g.set_attr("stmt", g_ret.get_attr("stmt"))
            return g_ret
        def callDecl():
            g_call = self.func_call()
            g.set_attr("stmt", g_call.get_attr("stmt"))
            return g_call
        g_alt = alternate(vardecl, retDecl, callDecl)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def varDecl(self):
        g = newParserRule()
        g_ty = self.type()
        g_name = self.Identifier()
        g_expr = self.expression()
        g.setProduction(g_ty, g_name, zero_or_one(self.Equal(), g_expr))
        
        var_name = g_name.getText()
        ty = g_ty.get_attr("type")
        expr: Operation = g_expr.get_attr("value")
        var = Variable(var_name, ty, expr)
        self.current_scope.insert_var(var)
        g.set_attr("stmt", expr)
        return g
    
    @parser
    def returnExpr(self):
        g = newParserRule()
        g_expr = self.expression()
        g.setProduction(self.Return(), zero_or_one(g_expr))
        if not g_expr.exist():
            g.set_attr("stmt", func.ReturnOp())
        else:
            expr = g_expr.get_attr("value")
            g.set_attr("stmt", func.ReturnOp(expr))            
        return g
    
    @parser
    def func_call(self):
        g = newParserRule()
        g_name = self.Identifier()
        g_params = self.func_call_params()
        g.setProduction(g_name, self.ParentheseOpen(), zero_or_one(g_params), self.ParentheseClose())
        return g
    
    @parser
    def func_call_params(self):
        g = newParserRule()
        g_first = self.expression()
        g_other = zero_or_more(self.Comma(), self.expression())
        g.setProduction(g_first, g_other)
        return g
    
    @parser
    def expression(self):
        g = newParserRule()
        g_add_expr = self.add_expr()
        g.setProduction(g_add_expr)
        g.set_attr("value", g_add_expr.get_attr("value"))
        g.set_attr("type", g_add_expr.get_attr("type"))
        return g
    
    @parser
    def add_expr(self):
        g = newParserRule()
        g_lhs = self.term_expr()
        lhs =  g_lhs.get_attr("value")
        lhs_type = g_lhs.get_attr("type")
        def add():
            g_rhs = self.add_expr()
            # get right handside
            rhs = g_rhs.get_attr("value")
            rhs_type = g_rhs.get_attr("type")
            if isinstance(lhs_type, IntegerType) and isinstance(rhs_type, IntegerType):
                value = arith.AddiOp(lhs, rhs)
                retty = lhs_type
            elif isinstance(rhs_type, FloatType) and isinstance(rhs_type, FloatType):
                value = arith.AddfOp(lhs, rhs)
                retty = lhs_type
            else:
                assert False, "unsupport add: {}".format(g.getText())
            g.set_attr("value", value)
            g.set_attr("type", retty)
            return concat(g_lhs, self.Add(), g_rhs)
        def sub():
            g_rhs = self.add_expr()
            # get right handside
            rhs = g_rhs.get_attr("value")
            rhs_type = g_rhs.get_attr("type")
            if isinstance(lhs_type, IntegerType) and isinstance(rhs_type, IntegerType):
                value = arith.SubiOp(lhs, rhs)
                retty = lhs_type
            elif isinstance(rhs_type, FloatType) and isinstance(rhs_type, FloatType):
                value = arith.SubfOp(lhs, rhs)
                retty = lhs_type
            else:
                assert False, "unsupport sub: {}".format(g.getText())
            g.set_attr("value", value)
            g.set_attr("type", retty)
            return concat(g_lhs, self.Sub(), g_rhs)
        def term():
            g.set_attr("value", lhs)
            g.set_attr("type", lhs_type)
            return g_lhs
        g_alt = alternate(add, sub, term)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def term_expr(self):
        g = newParserRule()
        g_lhs = self.prim_expr()
        lhs =  g_lhs.get_attr("value")
        lhs_type = g_lhs.get_attr("type")
        def mul():
            g_rhs = self.term_expr()
            # get right handside
            rhs = g_rhs.get_attr("value")
            rhs_type = g_rhs.get_attr("type")
            if isinstance(lhs_type, IntegerType) and isinstance(rhs_type, IntegerType):
                value = arith.MuliOp(lhs, rhs)
                retty = lhs_type
            elif isinstance(rhs_type, FloatType) and isinstance(rhs_type, FloatType):
                value = arith.MulfOp(lhs, rhs)
                retty = lhs_type
            else:
                assert False, "unsupport mul: {}".format(g.getText())
            g.set_attr("value", value)
            g.set_attr("type", retty)
            return concat(g_lhs, self.Mul(), g_rhs)
        def div():
            g_rhs = self.term_expr()
            # get right handside
            rhs = g_rhs.get_attr("value")
            rhs_type = g_rhs.get_attr("type")
            if isinstance(lhs_type, IntegerType) and isinstance(rhs_type, IntegerType):
                value = arith.DivSIOp(lhs, rhs)
                retty = lhs_type
            elif isinstance(rhs_type, FloatType) and isinstance(rhs_type, FloatType):
                value = arith.DivfOp(lhs, rhs)
                retty = lhs_type
            else:
                assert False, "unsupport div: {}".format(g.getText())
            g.set_attr("value", value)
            g.set_attr("type", retty)
            return concat(g_lhs, self.Div(), g_rhs)
        def prim():
            g.set_attr("value", lhs)
            g.set_attr("type", lhs_type)
            return g_lhs
        g_alt = alternate(mul, div, prim)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    
    @parser
    def prim_expr(self):
        g = newParserRule()
        def num():
            g_num = self.Number()
            num = int(g_num.getText())
            g.set_attr("value", arith.ConstantOp(builtin.IntegerAttr(num, builtin.i32)))
            g.set_attr("type", IntegerType())
            return g_num
        def id():
            g_id = self.Identifier()
            var: Variable = self.current_scope.lookup_var(g_id.getText())
            if var is None:
                assert False, "undefined var: {}".format(g_id.getText())
            g.set_attr("value", var.value)
            g.set_attr("type", var.ty)
            return g_id
        def expr():
            g_expr = self.expression()
            g.set_attr("value", g_expr.get_attr("value"))
            g.set_attr("type", g_expr.get_attr("value"))
            return concat(self.ParentheseOpen(), g_expr, self.ParentheseClose())
        g_alt = alternate(num, id, expr)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    

    @lexer
    def ParentheseOpen(self):
        return newTerminalRule("(")
    
    @lexer
    def ParentheseClose(self):
        return newTerminalRule(")")
    
    @lexer
    def BracketOpen(self):
        return newTerminalRule("{")
    
    @lexer
    def BracketClose(self):
        return newTerminalRule("}")
    
    @lexer
    def SbracketOpen(self):
        return newTerminalRule("[")
    
    @lexer
    def SbracketClose(self):
        return newTerminalRule("]")
    
    @lexer
    def Return(self):
        return newTerminalRule("return")
    
    @lexer
    def Semicolon(self):
        return newTerminalRule(";")
    
    @lexer
    def Var(self):
        return newTerminalRule("var")

    @lexer
    def Def(self):
        return newTerminalRule("def")
    
    @lexer
    def Struct(self):
        return newTerminalRule("struct")
    
    @lexer
    def INT(self):
        return newTerminalRule("int")
    
    @lexer
    def FLOAT(self):
        return newTerminalRule("float")
    
    @lexer
    def VOID(self):
        return newTerminalRule("void")
    
    @lexer
    def Number(self):
        return newTerminalRule(regular_expr("[0-9]+"))
    
    @lexer
    def Equal(self):
        return newTerminalRule("=")
    
    @lexer
    def AngleBracketsOpen(self):
        return newTerminalRule("<")
    
    @lexer
    def AngleBracketsClose(self):
        return newTerminalRule(">")
    
    @lexer
    def Comma(self):
        return newTerminalRule(",")
    
    @lexer
    def Add(self):
        return newTerminalRule("+")
    
    @lexer
    def Sub(self):
        return newTerminalRule("-")
    
    @lexer
    def Mul(self):
        return newTerminalRule("*")
    
    @lexer
    def Div(self):
        return newTerminalRule("/")
    
    @lexer
    def Dot(self):
        return newTerminalRule(".")
    
    @lexer
    def Identifier(self):
        return newTerminalRule(regular_expr("[a-zA-Z][a-zA-Z0-9_]*"))
    
    
def main():
    code = """
int main(){
    int a = 10;
    int b = 20;
    int c = a + b;
    return c;
}

int add(int a, int b){
    int c = a + b;
    return c;
}
    """
    simplc_g = SimpleC()
    mylexer = simplc_g.lexer()
    tokens = mylexer.input(code)
    # print("\n".join([token.__str__() for token in tokens]))
    myparser = simplc_g.parser(mylexer, "module")
    root = myparser.parse(code)
    theModule = root.get_attr("module")
    printer = Printer()
    printer.print(theModule)
    
main()