import ply.lex as lex
import ply.yacc as yacc

# 词法分析器
tokens = (
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
)

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

t_ignore = ' \t'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# 语法分析器
def p_expression_plus(p):
    'expression : expression PLUS term'
    p[0] = p[1] + p[3]
    
    for pi in p:
        print(pi)
    print("p_expression_plus")

def p_expression_minus(p):
    'expression : expression MINUS term'
    p[0] = p[1] - p[3]
    print("p_expression_minus")

def p_expression_term(p):
    'expression : term'
    p[0] = p[1]
    print("p_expression_term")

def p_term_times(p):
    'term : term TIMES factor'
    p[0] = p[1] * p[3]
    print("p_term_times")

def p_term_divide(p):
    'term : term DIVIDE factor'
    p[0] = p[1] / p[3]
    print("p_term_divide")

def p_term_factor(p):
    'term : factor'
    p[0] = p[1]
    print("p_term_factor")

def p_factor_number(p):
    'factor : NUMBER'
    p[0] = p[1]
    print("p_factor_number")

def p_error(p):
    print("Syntax error in input!")

parser = yacc.yacc()

# 测试
# while True:
#     try:
#         s = input('calc > ')
#     except EOFError:
#         break
#     if not s:
#         continue
#     result = parser.parse(s, lexer)
#     print(result)

result = parser.parse("1 + 2 + 3", lexer)
print(result)