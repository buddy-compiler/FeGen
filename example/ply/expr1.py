import ply.lex as lex
import ply.yacc as yacc

# 词法分析器
tokens = ('NUMBER', 'COMMA')
t_ignore = ' \t'

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

t_COMMA = r','

lexer = lex.lex()

# 语法分析器
def p_list(p):
    '''list : expr_list'''
    p[0] = p[1]

def p_expr_list(p):
    '''expr_list : expr
                 | expr_list COMMA expr'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_expr(p):
    'expr : NUMBER'
    p[0] = p[1]

parser = yacc.yacc()

# 测试
result = parser.parse("1, 2, 3")
print(result)  # 输出 [1, 2, 3]