from FeGen import *
import logging
logging.basicConfig(level=logging.DEBUG)

from codegen import Visitor
from xdsl.printer import Printer

class ToyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
        self.functionTable = {}
        self.funcmap : dict = {}
        self.visitor = Visitor()
        self.themodule = self.visitor.module
        
    @parser
    def module(self):
        """
        module: structDefine* funcDefine+
        """
        g = newParserRule()
        g_struct_def = self.structDefine()
        g_struct_defs = zero_or_more(g_struct_def)
        g_func_def = self.funcDefine()
        g_func_defs = one_or_more(g_func_def)
        g.setProduction(concat(g_struct_defs, g_func_defs))
        self.visitor.visit(g, g_func_defs)
        return g
    
    
    @parser
    def funcDefine(self):
        """
        funcDefine: prototype block
        """
        g = newParserRule()
        g_proto = self.prototype()
        g_block = self.block()
        g.setProduction(concat(g_proto, g_block))
        self.visitor.visit(g, g_proto, g_block)
        return g
    
    @parser
    def prototype(self):
        """
        prototype: Def Identifier ParentheseOpen params? ParentheseClose
        """
        g = newParserRule()
        g_id = self.Identifier()
        g_decl_List = zero_or_one(self.params())
        g.setProduction(concat(self.Def(), g_id, self.ParentheseOpen(), g_decl_List, self.ParentheseClose()))
        self.visitor.visit(g, g_id, g_decl_List)
        return g
    
    @parser
    def params(self):
        """
        params: param (Comma param)*
        """
        g = newParserRule()
        g_first = self.param()
        g_other = zero_or_more(self.Comma(), self.param())
        g.setProduction(g_first, g_other)
        self.visitor.visit(g, g_first, g_other)
        return g
    
    @parser
    def param(self):
        """
        param: Identifier | Identifier Identifier
        """
        g = newParserRule()
        
        def alt_id():
            g_id = self.Identifier()
            return g_id
        
        def alt_typed():
            g_id = self.Identifier()
            g_ty = self.Identifier()
            return concat(g_ty, g_id)
        
        g_alt = alternate(alt_id, alt_typed)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g

    
    @parser
    def block(self):
        """
        block: BracketOpen (blockExpr Semicolon)* BracketClose
        """
        g = newParserRule()
        g_blockexprs = zero_or_more(concat(self.blockExpr(), self.Semicolon()))
        g.setProduction(concat(self.BracketOpen(), g_blockexprs, self.BracketClose()))
        self.visitor.visit(g, g_blockexprs)
        return g
    
    @parser
    def blockExpr(self):
        """
        blockExpr: varDecl | returnExpr | func_call
        """
        g = newParserRule()
        def alt_var():
            g_var = self.varDecl()
            return g_var
        def alt_ret():
            g_ret = self.returnExpr()
            return g_ret
        def alt_func_call():
            g_call = self.func_call()
            return g_call
        g_alt = alternate(alt_var, alt_ret, alt_func_call)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    
    @parser
    def varDecl(self):
        """
        varDecl : Var Identifier type? (Equal expression)?
                | Identifier Identifier (Equal expression)?
        """
        g = newParserRule()
        def alt_tensor():
            return self.tensorVarDecl()
        
        def alt_struct():
            return self.structVarDecl()
        
        g_alt = alternate(alt_tensor, alt_struct)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    
    @parser
    def tensorVarDecl(self):
        g = newParserRule()
        g_id = self.Identifier()
        g_type = self.type()
        g_expr = self.expression()
        g.setProduction(self.Var(), g_id, zero_or_one(g_type), concat(self.Equal(), g_expr))
        self.visitor.visit(g, g_id, g_type, g_expr)
        return g
    
    @parser
    def type(self):
        """
        type: AngleBracketsOpen Number (Comma Number)* AngleBracketsClose
        """
        g = newParserRule()
        g_first_num = self.Number()
        g_other_num = zero_or_more(concat(self.Comma(), self.Number()))
        g.setProduction(concat(self.AngleBracketsOpen(), g_first_num, g_other_num, self.AngleBracketsClose()))
        self.visitor.visit(g, g_first_num, g_other_num)
        return g
    
    
    @parser
    def structVarDecl(self):
        g = newParserRule()
        g_struct_name = self.Identifier()
        g_var_name = self.Identifier()
        g_expr = self.expression()
        g.setProduction(g_struct_name, g_var_name, zero_or_one(concat(self.Equal(), g_expr)))
        return g
    
    @parser
    def returnExpr(self):
        g = newParserRule()
        def ret():
            return self.Return()
        
        def ret_expr():
            g_expr = self.expression()
            return concat(self.Return(), g_expr)
        
        g_alt = alternate(ret, ret_expr)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    

    
    @parser
    def expression(self):
        """
        expression: add_expr
        """
        g = newParserRule()
        
        g_add = self.add_expr()
        g.setProduction(g_add)
        self.visitor.visit(g, g_add)
        return g
    
    @parser
    def add_expr(self):
        """
        add_expr: term_expr | term_expr Add add_expr 
        """
        g = newParserRule()
        def alt_term():
            g_term = self.term_expr()
            return g_term
        
        def alt_add():
            g_term = self.term_expr()
            g_add = self.add_expr()
            return concat(g_term, self.Add(), g_add)
        
        g_alt = alternate(alt_term, alt_add)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g

    
    @parser
    def term_expr(self):
        """
        term_expr: prim_expr | prim_expr Mul term_expr 
        """
        g = newParserRule()
        def alt_prim():
            g_prim = self.prim_expr()
            return g_prim
        
        def alt_mul():
            g_prim = self.prim_expr()
            g_term = self.term_expr()
            return concat(g_prim, self.Mul(), g_term)
        
        g_alt = alternate(alt_prim, alt_mul)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    
    @parser
    def prim_expr(self):
        """
        prim_expr: Num | tensorLiteral | structLiteral | variable_access | func_call
        """
        g = newParserRule()
        def alt_num():
            g_num = self.Number()
            return g_num
        
        def alt_tensor():
            g_tensor = self.tensorLiteral()
            return g_tensor
    
        def alt_struct():
            g_stru = self.structLiteral()
            return g_stru
        
        def alt_var_access():
            g_acc = self.variable_access()
            return g_acc
        
        def alt_call():
            g_call = self.func_call()
            return g_call
        
        g_alt = alternate(alt_num, alt_tensor, alt_struct, alt_var_access, alt_call)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    
    
    @parser
    def tensorLiteral(self):
        """
        tensorLiteral: SbracketOpen (tensorLiteral (Comma tensorLiteral)*)? SbracketClose
        """
        g = newParserRule()
        def tensor():
            g_first = self.tensorLiteral()
            g_other = zero_or_more(concat(self.Comma(), self.tensorLiteral()))
            g_content = zero_or_one(concat(g_first, g_other))
            return concat(self.SbracketOpen(), g_content, self.SbracketClose())
        
        def num():
            g_num = self.Number()
            return g_num
    
        g_alt = alternate(tensor, num)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g
    

    
    @parser
    def variable_access(self):
        """
        variable_access: name_access | member_access
        """
        g = newParserRule()
        def alt_name():
            g_name = self.name_access()
            return g_name
        
        def alt_member():
            g_mem = self.member_access()
            return g_mem
        
        g_alt = alternate(alt_name, alt_member)
        g.setProduction(g_alt)
        self.visitor.visit(g, g_alt)
        return g    
    
    @parser
    def name_access(self):
        """
        name_access: Identifier
        """
        g = newParserRule()
        g.setProduction(self.Identifier())
        self.visitor.visit(g)
        return g
    
    @parser
    def member_access(self):
        """
        member_access: variable_access Dot Identifier
        """
        g = newParserRule()
        g_var = self.variable_access()
        g_id = self.Identifier()
        g.setProduction(g_var, self.Dot(), g_id)
        return g
    
    @parser
    def func_call(self):
        """
        func_call: Identifier ParentheseOpen (expression (Comma expression)*)? ParentheseClose
        """
        g = newParserRule()
        g_id = self.Identifier()
        g_first_expr = self.expression()
        g_other_expr = zero_or_more(concat(self.Comma(), self.expression()))
        g_param = zero_or_one(concat(g_first_expr, g_other_expr))
        g.setProduction(g_id, self.ParentheseOpen(), g_param, self.ParentheseClose())
        self.visitor.visit(g, g_id, g_first_expr, g_other_expr)
        return g
    

    


    
    @parser
    def literalList(self):
        g = newParserRule()
        def alt1():
            g_tensor = self.tensorLiteral()
            return g_tensor
        def alt2():
            g_tensor = self.tensorLiteral()
            g_list = self.literalList()
            return concat(g_tensor, self.Comma(), g_list)
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def structLiteral(self):
        g = newParserRule()
        def stru():
            g_stru = self.structLiteral()
            return g_stru
        def lit():
            g_lit = self.literalList()
            return g_lit
        # visit alt
        g_first = alternate(stru, lit)
        g_other = zero_or_more(concat(self.Comma(), alternate(stru, lit)))
        g.setProduction(zero_or_one(concat(self.BracketOpen(), g_first, g_other, self.BracketClose())))
        return g
    
    @parser
    def structDefine(self):
        g = newParserRule()
        g_id = self.Identifier()
        g_decls = zero_or_more(concat(self.varDecl(), self.Semicolon()))
        g.setProduction(self.Struct(), g_id, self.BracketOpen(), g_decls, self.BracketClose())
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
    def Identifier(self):
        return newTerminalRule(regular_expr("[a-zA-Z][a-zA-Z0-9_]*"))
    
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
    def Mul(self):
        return newTerminalRule("*")
    
    @lexer
    def Dot(self):
        return newTerminalRule(".")
    

import os
import sys


if __name__ == "__main__":
    d = os.path.dirname(sys.argv[0])
    example_file = "constant.toy"
    example_path = os.path.join(d, example_file)
    with open(example_path) as f:
        code = f.read()
        # print(code)
        toygram = ToyGrammar()
        toylexer = toygram.lexer()
        # tokens = toylexer.input(code)
        # print("\n".join([token.__str__() for token in tokens]))
        toyparser = toygram.parser(toylexer, "module")
        tree = toyparser.parse(code)
        tree.visit()
        printer = Printer()
        printer.print(toygram.themodule)