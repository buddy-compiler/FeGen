from FeGen import *
import logging
import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin
import xdsl.dialects.arith as arith
from xdsl.printer import Printer


from SymbolTable import Variable, Table


logging.basicConfig(level=logging.DEBUG)

class AssignGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
        self.table = Table()

    @sema
    def create_func(self, stmts):
        functy = func.FunctionType.from_lists([], [])
        myfunc = func.FuncOp(name="test", function_type=functy, visibility="private")
        myfunc.body.block.add_ops(stmts)
        return myfunc

    @parser
    def statements(self):
        g = newParserRule()
        g_stmts = zero_or_more(self.assign_stmt())
        g.setProduction(g_stmts)
        all_stmts = []
        for g_stmt in g_stmts:
            all_stmts += g_stmt.get_attr("stmts")
        myfunc = self.create_func(all_stmts)
        g.set_attr("func", myfunc)
        return g
    
    @parser
    def assign_stmt(self):
        g = newParserRule()
        g_var = self.variable_access()
        g_expr = self.expression()
        g.setProduction(concat(g_var, self.ASSIGN(), g_expr))
        var_name = g_var.get_attr("name")
        value = g_expr.get_attr("value")
        var = self.table.lookup(var_name)
        if var is None:
            var = Variable(var_name, value)
            self.table.update(var)
        else:
            var.value = value
        g.set_attr("var", var)
        stmts = g_expr.get_attr("stmts")
        g.set_attr("stmts", stmts)
        return g
        
    @parser
    def variable_access(self):
        g = newParserRule()
        g_varname = self.IDEN()
        g.setProduction(g_varname)
        
        var_name = g_varname.getText()
        g.set_attr("name", var_name)
        value = self.table.lookup(var_name)
        g.set_attr("value", value)
        return g
    
    
    @parser
    def expression(self):
        g = newParserRule()
        g_add = self.add_expr()
        g.set_attr("value", g_add.get_attr("value"))
        g.set_attr("stmts", g_add.get_attr("stmts"))
        g.setProduction(g_add)
        return g
    
    @parser
    def add_expr(self):
        g = newParserRule()
        g_term = self.term_expr()
        lhs = g_term.get_attr("value")
        lhs_stmts = g_term.get_attr("stmts")
        def add():
            g_add = self.add_expr()
            rhs = g_add.get_attr("value")
            res = arith.AddiOp(lhs, rhs, builtin.i32)
            g.set_attr("value", res)
            rhs_stmts = g_add.get_attr("stmts")
            stmts : list = lhs_stmts + rhs_stmts
            stmts.append(res)
            g.set_attr("stmts", stmts)
            return concat(g_term, self.ADD(), g_add)
        
        def sub():
            g_add = self.add_expr()
            rhs = g_add.get_attr("value")
            res = arith.SubiOp(lhs, rhs, builtin.i32)
            g.set_attr("value", res)
            rhs_stmts = g_add.get_attr("stmts")
            stmts : list = lhs_stmts + rhs_stmts
            stmts.append(res)
            g.set_attr("stmts", stmts)
            return concat(g_term, self.SUB(), g_add)
        
        def term():
            g.set_attr("value", lhs)
            g.set_attr("stmts", lhs_stmts)
            return g_term
        
        alt = alternate(add, sub, term)
        alt.visit()
        g.setProduction(alt)
        return g
    
    
    @parser
    def term_expr(self):
        g = newParserRule()
        g_prim = self.prim_expr()
        lhs = g_prim.get_attr("value")
        lhs_stmts = g_prim.get_attr("stmts")
        def mul():
            g_term = self.term_expr()
            rhs = g_term.get_attr("value")
            ret = arith.MuliOp(lhs, rhs, builtin.i32)
            g.set_attr("value", ret)
            rhs_stmts = g_term.get_attr("stmts")
            stmts : list = lhs_stmts + rhs_stmts
            stmts.append(ret)
            g.set_attr("stmts", stmts)
            return concat(g_prim, self.MUL(), g_term)
        
        def div():
            g_term = self.term_expr()
            rhs = g_term.get_attr("value")
            ret = arith.DivSIOp(lhs, rhs, builtin.i32)
            g.set_attr("value", ret)
            rhs_stmts = g_term.get_attr("stmts")
            stmts : list = lhs_stmts + rhs_stmts
            stmts.append(ret)
            g.set_attr("stmts", stmts)
            return concat(g_prim, self.DIV(), g_term)
        
        def prim():
            g.set_attr("value", lhs)
            g.set_attr("stmts", lhs_stmts)
            return g_prim
        
        alt = alternate(mul, div, prim)
        alt.visit()
        g.setProduction(alt)
        return g
            
    @parser
    def prim_expr(self):
        g = newParserRule()  
        def alt1():
            g_var = self.variable_access()
            value: Variable = g_var.get_attr("value")
            if value is None:
                name = g_var.get_attr("name")
                print(f"Undefined reference to {name}")
                exit(0)
            g.set_attr("value", value.value)
            g.set_attr("stmts", [])
            return g_var
        
        def alt2():
            g_num = self.NUM()
            num = int(g_num.getText())
            value = arith.ConstantOp(builtin.IntegerAttr(num, builtin.i32))
            g.set_attr("value", value)
            g.set_attr("stmts", [value])
            return g_num
        
        def alt3():
            g_expr = self.expression()
            g.set_attr("value", g_expr.get_attr("value"))
            g.set_attr("stmts", g_expr.get_attr("stmts"))
            return concat(self.LP(), g_expr, self.RP())
      
        g_alt = alternate(alt1, alt2, alt3)
        g_alt.visit()
        g.setProduction(g_alt)
        return g

    
    @lexer
    def LP(self):
        return newTerminalRule("(")
    
    @lexer
    def RP(self):
        return newTerminalRule(")")
    
    @lexer
    def ADD(self):
        return newTerminalRule("+")
    
    @lexer
    def SUB(self):
        return newTerminalRule("-")

    @lexer
    def MUL(self):
        return newTerminalRule("*")
    
    @lexer
    def DIV(self):
        return newTerminalRule("/")
    

    @lexer
    def ASSIGN(self):
        return newTerminalRule("=")
    
    @lexer
    def IDEN(self):
        return newTerminalRule(regular_expr("[a-zA-Z_][a-zA-Z0-9_]*"))
    
    @lexer
    def NUM(self):
        return newTerminalRule(regular_expr("[1-9][0-9]+|[0-9]"))


code = """
a = 1 + 2
"""

code1 = """
a = 1 + 2
b = 2 * 3
c = (a + b)
d = (a + b) * c
"""
    
    
assign_gram = AssignGrammar()
lexer = assign_gram.lexer()
parser = assign_gram.parser(lexer, "statements")
root = parser.parse(code1)
print(root.getText())
myfunc = root.get_attr("func")
printer = Printer()
printer.print(myfunc)