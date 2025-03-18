import ply.lex as lex
import ply.yacc as yacc

# 词法分析器
tokens = (
    # 'ID',
    "NUM",
)

t_NUM = r"([1-9][0-9]+)|[0-9]"

t_ignore = ' \t'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# 语法分析器

def p_number(p):
    "number : NUM"
    p[0] = p[1]

parser = yacc.yacc(start="number")

result = parser.parse("100", lexer)
print(result)