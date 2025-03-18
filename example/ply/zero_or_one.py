import ply.lex as lex
import ply.yacc as yacc

# 词法分析器
tokens = (
    'A',
)

t_A = "A"

t_ignore = ' \t'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# 语法分析器

def p_test(p):
    'test : __zero_or_one_A'
    p[0] = {"__zero_or_one_A": p[1]}

def p___zero_or_one_A(p):
    """__zero_or_one_A : A 
                       |"""
    if len(p) == 1:
        p[0] = None
    elif len(p) == 2:
        p[0] = p[1]
    else:
        assert False

parser = yacc.yacc(start="test")

print(p___zero_or_one_A.__doc__)

result = parser.parse("", lexer)
print(result)