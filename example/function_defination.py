from typing import List, Dict, Tuple
from xdsl.dialects.builtin import IntegerType, SSAValue, Region, Block, IntegerAttr
from xdsl.context import Context
from xdsl.printer import Printer
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.memref as memref
import xdsl.dialects.func as func
from FeGen import *

"""
function_defination
[syn func_decl: SSAValue]
    : proto_type func_body {
        funcname = $proto_type.funcname
        functy = $proto_type.functy
        myfunc = func.FuncOp(name=funcname, function_type=functy, visibility="private")
        params = $proto_type.params
        for index, arg in enumrate(myfunc.args):
            name = params[index][0]
            table.insert(name, arg)
        
        stmts = $func_body.stmts
        myfunc.body.block.add_ops(stmts)
    }
    ;
    
proto_type
[syn functy: FunctionType, funcname: str, params: List[Tuple[name, Type]]]
    : DEF func_name LP func_params RP func_return_ty {
        funcname = $func_name.name
        ps = $func_params.ps
        ty = $func_return_ty.ty
        functy = FunctionType([p[1] for p in ps], [] if ty is None else [ty])
    }
    ;

func_params
[syn ps: List[Tuple[name, Type]]]
    : params {ps = $params.ps}
    | {ps = []}
    ;

func_return_ty
[syn ty: Type]
    : RA type {ty = $type.ty}
    | {ty = None}
    ;

func_name
[syn name: str]
    : Identifier {return $Identifier.text}
    ;

params
[syn ps: List[Tuple[name, Type]]]
    : param (Comma param)* {
        ps = []
        for p in $param:
            ps.append(p.p)
    }
    ;
    
param
[syn p: Tuple[name, Type]]
    : type Identifier {
        name = $Identifier.text
        ty = $type.ty
        p = (name, ty)
    }
    ;
    
type
[syn ty: Type]
    : INT {ty = IntegerType(32)}
    | FLOAT {ty = FloatPoint32Type()}
    ;

func_body
[syn stmts: List[SSAValue]]
    : LB statement* RB {stmts = $statement.stmt}
    ;
    

statement
[syn stmt: SSAValue]
    : assign_stmt {stmt = $assign_stmt.stmt}
    | return_stmt {stmt = $return_stmt.stmt}
    ;

assign_stmt
[syn stmt: SSAValue]
    : variable_access Assign expression {
        name = $variable_access.name
        v = $expression.v
        table.insert(name, v)
        stmt = v
    }
    ;

variable_access
    [syn ssavalue: SSAValue, name: str]
    : name_access {
        ssavalue = $name_access.ssavalue
        name = $name_access.name
    }
    ;

name_access
    [syn ssavalue: SSAValue, name: str]
    : Identifier {
        name = $Identifier.text
        ssavalue = table.find(name)
    }
    ;
    
expression
    [syn v: Value]
    : prim_expr {v = $prim_expr.v}
    | add_expr {v = $add_expr.v}
    ;    
    
prim_expr
    [syn v: Value]
    : variable_access {
        v = $variable_access.ssavalue
        if v is None:
            name = $variable_access.name
            assert False && f"use of undefined variable: {name}"
    }
    | Number {v = arith.constantOp(int($Number.text), IntegerType.get(32))}
    ;
    
add_expr
    [syn v: Value]
    : prim_expr Add add_expr {v = arith.addi($prim_expr.v, $add_expr.v)}
    ;
    
return_stmt
    [syn v: SSAValue]
    : RETURN variable_access {
        var = $variable_access.ssavalue
        if var is None:
            name = $variable_access.name
            assert False && f"use of undefined variable: {name}"
        v = func.ReturnOp(var)
    }
    ;

RETURN: 'return';

Add: '+';

Assign: '=';

DEF: 'def';

LP: '(';

RP: ')';

LB: '{';

RB: '}';

RA: '->';

Comma: ',';

INT: 'int';

FLOAT: 'float';

Number: '0-9' | '1-9' '0-9'*;

Identifier: [a-zA-Z][a-zA-Z0-9];

"""

table : dict = {}


@attr_grammar
class MyGrammar(FeGenGrammar):
    @parser
    def function_defination(self):
        """
        function_defination
        [syn func_decl: SSAValue]
            : proto_type func_body {
                funcname = $proto_type.funcname
                functy = $proto_type.functy
                myfunc = func.FuncOp(name=funcname, function_type=functy, visibility="private")
                params = $proto_type.params
                for index, arg in enumrate(myfunc.args):
                    name = params[index][0]
                    table.insert(name, arg)
                
                stmts = $func_body.stmts
                myfunc.body.block.add_ops(stmts)
            }
            ;
        """
        g = newParserRule()
        proto_ty = self.proto_type()
        func_body = self.func_body()
        g.setProduction(concat(proto_ty, func_body))

        func_decl = g.new_attr("func_decl", SSAValue)
        funcname = proto_ty.get_attr("funcname")
        functy = proto_ty.get_attr("functy")
        myfunc = func.FuncOp(name=funcname, function_type=functy, visibility="private")
        func_decl.set(myfunc)
        params = proto_ty.get_attr("params")
        for index, arg in enumerate(myfunc.args):
            name = params[index][0]
            table.insert(name, arg)
        stmts = func_body.get_attr("stmts")
        myfunc.body.block.add_ops(stmts)
        return g

    @parser    
    def proto_type(self):
        """
        proto_type
        [syn functy: FunctionType, funcname: str, params: List[Tuple[name, Type]]]
            : DEF func_name LP func_params RP func_return_ty {
                funcname = $func_name.name
                ps = $func_params.ps
                ty = $func_return_ty.ty
                functy = FunctionType([p[1] for p in ps], [] if ty is None else [ty])
            }
            ;
        """
        g = newParserRule()
        func_name = self.func_name()
        params =  self.func_params()
        func_return_ty = self.func_return_ty() 
        g.setProduction(concat(self.DEF, func_name, self.LP(), params, self.RP(), func_return_ty))
        
        funcname = func_name.get_attr("name")
        ps = params.get_attr("ps")
        ty = func_return_ty.get_attr("ty")
        functy = func.FunctionType([p[1] for p in ps], [] if ty is None else [ty])
        g.new_attr("funcname", str, funcname)
        g.new_attr("params", List[Tuple[str, builtin.TypeAttribute]], ps)
        g.new_attr("functy", func.FunctionType, functy)
        return g

    @parser
    def func_name(self):
        """
        func_name
        [syn name: str]
            : Identifier {return $Identifier.text}
            ;
        """
        g = newParserRule()
        iden = self.Identifier()
        g.setProduction(iden)
        g.new_attr("name", str, iden.text())
        return g

    def func_params(self):
        """
        func_params
        [syn ps: List[Tuple[str, builtin.TypeAttribute]]]
            : params {ps = $params.ps}
            | {ps = []}
            ;
        """
        g = newParserRule()
        ps = g.new_attr("ps", List[Tuple[str, builtin.TypeAttribute]])
        def alt1():
            params = self.params()
            ps.set(params.get_attr("ps"))
            return params
        
        def alt2():
            ps.set([])
            return newParserRule()
        
        g.setProduction(alternate(alt1, alt2))
        return g
            

    @parser
    def func_return_ty(self):
        """
        func_return_ty
        [syn ty: Type]
            : RA type {ty = $type.ty}
            | {ty = None}
            ;
        """
        g = newParserRule()
        ty = g.new_attr("ty", builtin.TypeAttribute)
            
        def alt1():
            g_ty = self.type()
            ty.set(g_ty.get_attr("ty"))
            return concat(self.RA, g_ty)
        
        def alt2():
            ty.set(None)
            return newParserRule()
        
        g.setProduction(alternate(alt1, alt2))
        return g

    @parser
    def func_body(self):
        """
        func_body
        [syn stmts: List[SSAValue]]
            : LB statement* RB {stmts = $statement.stmt}
            ;
        """
        g = newParserRule()
        g_statement = self.statement()
        g_statements = zero_or_more(g_statement)
        g.setProduction(concat(self.LB(), g_statements, self.RB()))
        
        stmts = g.new_attr("stmts", List[SSAValue])
        init =  []
        for g_stmt in g_statements:
            init.append(g_stmt.get_attr("stmt"))
        stmts.set(init)
        return g

    @parser
    def statement(self):
        """
        statement
        [syn stmt: SSAValue]
            : assign_stmt {stmt = $assign_stmt.stmt}
            | return_stmt {stmt = $return_stmt.stmt}
            ;
        """
        g = newParserRule()
        stmt = g.new_attr("stmt", SSAValue)
        def alt1():
            g_assign_stmt = self.assign_stmt()
            stmt.set(g_assign_stmt.get_attr("stmt"))
            return g_assign_stmt
        def alt2():
            g_return_stmt = self.return_stmt()
            stmt.set(g_return_stmt.get_attr("stmt"))
            return g_return_stmt
        g.setProduction(alternate(alt1, alt2))
        return g
    
    
    @parser
    def assign_stmt(self):
        """
        assign_stmt
        [syn stmt: SSAValue]
            : variable_access Assign expression {
                name = $variable_access.name
                v = $expression.v
                table.insert(name, v)
                stmt = v
            }
            ;
        """
        
        # grammar define
        var_access = self.variable_access()
        expr = self.expression()
        g = newParserRule()
        g.setProduction(concat(var_access, self.Assign(), expr))
        
        # attribute define
        v: SSAValue = expr.get_attr("v")
        name = var_access.get_attr("name")
        table[name] = v
        stmt = g.new_attr("stmt", SSAValue)
        stmt.set(v)
        return g
    
    @parser
    def variable_access(self):
        """
        variable_access
            [syn ssavalue: SSAValue, name: str]
            : name_access {
                ssavalue = $name_access.ssavalue
                name = $name_access.name
            }
            ;
        """
        name_acc = self.name_access()
        g = newParserRule(name_acc)
        alloca = name_acc.get_attr("ssavalue")
        alloca_ = g.new_attr("ssavalue", SSAValue)
        alloca_.set(alloca)
        
        name = name_acc.get_attr("name")
        name_ = g.new_attr("name", SSAValue)
        name_.set(name)
        return g
    

    @parser
    def name_access(self):
        """
        name_access
            [syn ssavalue: SSAValue, name: str]
            : Identifier {
                name = $Identifier.text
                ssavalue = table.find(name)
            }
            ;
        """
        id = self.Identifier()
        g = newParserRule(id)
        name = id.text()
        g.new_attr("name", SSAValue).set(name)
        g.new_attr("ssavalue", SSAValue).set(table[name] if name in table else None)
        return g

    @parser
    def expression(self):
        """
        expression
            [syn v: Value]
            : prim_expr {v = $prim_expr.v}
            | add_expr {v = $add_expr.v}
            ;    
            
        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        def alt1():
            expr = self.prim_expr()
            v.set(expr.get_attr("v"))
            return expr
        
        def alt2():
            expr = self.add_expr()
            v.set(expr.get_attr("v"))
            return expr

        g.setProduction(alternate(alt1, alt2))
        return g
    
    @parser
    def prim_expr(self):
        """
        prim_expr
            [syn v: Value]
            : variable_access {
                v = $variable_access.ssavalue
                if v is None:
                    name = $variable_access.name
                    assert False && f"use of undefined variable: {name}"
            }
            | Number {v = arith.constantOp(int($Number.text), IntegerType.get(32))}
            ;
        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        def alt1():
            var_acc = self.variable_access()
            ssavalue = var_acc.get_attr("ssavalue")
            name = var_acc.get_attr("name")
            assert ssavalue is not None and f"use of undefined variable: {name}"
            v.set(ssavalue)
            return var_acc
        
        def alt2():
            num = self.Number()
            const = arith.ConstantOp(IntegerAttr(int(num.text()), IntegerType(32)))
            v.set(const)
            return num
        
        g.setProduction(alternate(alt1, alt2))
        return g

    @parser
    def add_expr(self):
        """

        add_expr
            [syn v: Value]
            : prim_expr Add add_expr {v = arith.addi($prim_expr.v, $add_expr.v)}
            ;

        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        prim = self.prim_expr()
        add = self.add_expr()
        lhs = prim.get_attr("v")
        rhs = add.get_attr("v")
        v.set(arith.AddiOp(lhs, rhs))
        g.setProduction(concat(prim, self.Add(), add))
        return g
    
    @parser
    def return_stmt(self):
        """
        return_stmt
            [syn stmt: SSAValue]
            : RETURN variable_access {
                var = $variable_access.ssavalue
                if var is None:
                    name = $variable_access.name
                    assert False && f"use of undefined variable: {name}"
                stmt = func.ReturnOp(var)
            }
            ;
        """
        g = newParserRule()
        g_var_access = self.variable_access()
        g.setProduction(concat(self.RETURN(), g_var_access))
        
        stmt = g.new_attr("stmt", SSAValue)
        var = g_var_access.get_attr("ssavalue")
        name = g_var_access.get_attr("name")
        assert var is not None and f"use of undefined variable: {name}"
        stmt.set(func.ReturnOp(var))
        return g
    
    @lexer
    def Add(self):
        """
            Add: '+';
        """
        return newTerminalRule("+")
    
    @lexer
    def Number(self):
        """
            Number: '0-9' | '1-9' '0-9'*;
        """
        no_zero = char_set("1-9")
        all_number = char_set("0-9")
        return newTerminalRule(alternate(lambda: all_number, lambda: concat(no_zero, zero_or_more(all_number))))
    
    @lexer
    def Dot(self):
        """
            Dot: '.';
        """
        return newTerminalRule(".")
    
    @lexer
    def Identifier(self):
        """
            Identifier: [a-zA-Z][a-zA-Z0-9]*;
        """
        noDigit = char_set("a-zA-Z")
        allcase = char_set("a-zA-Z0-9")
        # return newTerminalRule(concat(noDigit, zero_or_more(allcase)))
        return newTerminalRule(noDigit, zero_or_more(allcase))

    @lexer
    def Assign(self):
        """
            Assign: '=';
        """
        return newTerminalRule("=")
    
    @lexer
    def RETURN(self):
        return newTerminalRule("return")
    
    @lexer
    def DEF(self):
        return newTerminalRule("def")
    
    @lexer
    def LP(self):
        return newTerminalRule("(")
    
    @lexer
    def RP(self):
        return newTerminalRule(")")
    
    @lexer
    def LB(self):
        return newTerminalRule("{")
    
    @lexer
    def RB(self):
        return newTerminalRule("}")
    
    @skip
    def skip(self):
        return newTerminalRule(char_set("\n\t "))
    
if __name__ == "__main__":
    context = Context()
    context.load_dialect(arith.Arith)
    context.load_dialect(func.Func)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(memref.MemRef)
    code = """
        def add(int x, int y) -> int {
            return x + y
        }
    """
    grammar = MyGrammar()
    mylexer = grammar.lexer()
    myparser = grammar.parser()
    p = myparser(mylexer(code))
    tree = p.function_defination()
    myfunc = tree.get_attr("func_decl")
    printer = Printer()
    printer.print_op(myfunc)
    