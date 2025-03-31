from FeGen import *
import logging

logging.basicConfig(level=logging.DEBUG)


class ToyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
        self.functionTable = {}
        self.funcmap : dict = {}
        
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
        for g_func in g_func_defs:
            func_name = g_func.get_attr("name")
            funcop = g_func.get_attr("funcop")
            self.funcmap.update({func_name, funcop})
        g.setProduction(concat(g_struct_defs, g_func_defs))
        return g
    
    @parser
    def expression(self):
        g = newParserRule()
        def num():
            g_num = self.Number()
            return g_num
        
        def tensor():
            g_tensor = self.tensorLiteral()
            return g_tensor
        
        def id():
            g_id = self.identifierExpr()
            return g_id
        
        def mul():
            g_lhs = self.expression()
            g_rhs = self.expression()
            return concat(g_lhs, self.Mul(), g_rhs)
        
        def add():
            g_lhs = self.expression()
            g_rhs = self.expression()
            return concat(g_lhs, self.Add(), g_rhs)
        
        def dot():
            g_lhs = self.expression()
            g_rhs = self.expression()
            return concat(g_lhs, self.Dot(), g_rhs)
        
        def struct():
            g_stru = self.structLiteral()
            return g_stru
        
        g_alt = alternate(num, tensor, id, mul, add, dot, struct)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def identifierExpr(self):
        g = newParserRule()
        def id():
            g_id = self.Identifier()
            return g_id
        
        def func_call():
            g_id = self.Identifier()
            g_first_expr = self.expression()
            g_other_expr = zero_or_more(concat(self.Comma(), self.expression()))
            g_param = zero_or_one(concat(g_first_expr, g_other_expr))
            return concat(g_id, self.ParentheseOpen(), g_param, self.ParentheseClose())
        
        g_alt = alternate(id, func_call)
        g_alt.visit()
        g.setProduction(g_alt)
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
        return g
    
    @parser
    def tensorLiteral(self):
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
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def varDecl(self):
        g = newParserRule()
        def decl_var():
            g_id = self.Identifier()
            g_type = self.type()
            g_expr = self.expression()
            return concat(self.Var(), g_id, zero_or_one(g_type), zero_or_one(concat(self.Equal(), g_expr)))
        
        def decl_struct():
            g_struct_name = self.Identifier()
            g_var_name = self.Identifier()
            g_expr = self.expression()
            return concat(g_struct_name, g_var_name, zero_or_one(concat(self.Equal(), g_expr)))
        
        g_alt = alternate(decl_var, decl_struct)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def type(self):
        g = newParserRule()
        g_first_num = self.Number()
        g_other_num = zero_or_more(concat(self.Comma(), self.Number()))
        g.setProduction(concat(self.AngleBracketsOpen(), g_first_num, g_other_num, self.AngleBracketsClose()))
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
        return g
    
    @parser
    def prototype(self):
        """
        prototype: Def Identifier ParentheseOpen declList? ParentheseClose
        """
        g = newParserRule()
        g_id = self.Identifier()
        g_decl_List = zero_or_one(self.declList())
        g.setProduction(concat(self.Def(), g_id, self.ParentheseOpen(), g_decl_List, self.ParentheseClose()))
        return g
    
    @parser
    def declList(self):
        g = newParserRule()
        def alt1():
            g_decl = self.varDecl()
            return g_decl
        def alt2():
            g_decl = self.varDecl()
            g_decl_list = self.declList()
            return concat(g_decl, self.Comma(), g_decl_list) 
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def block(self):
        g = newParserRule()
        g_blockexprs = zero_or_more(concat(self.blockExpr(), self.Semicolon()))
        g.setProduction(concat(self.BracketOpen(), g_blockexprs, self.BracketClose()))
        return g
    
    @parser
    def blockExpr(self):
        g = newParserRule()
        def var():
            g_var = self.varDecl()
            return g_var
        def ret():
            g_ret = self.returnExpr()
            return g_ret
        def expr():
            g_expr = self.expression()
            return g_expr
        g_alt = alternate(var, ret, expr)
        g_alt.visit()
        g.setProduction(g_alt)
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
        print(tree.getText())