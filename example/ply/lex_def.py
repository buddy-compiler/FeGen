import ply.lex as lex
import ply.yacc as yacc

# 词法分析器
tokens = (
    # 'ID',
    # "NUM",
    "backslash_LB",
    "Dot",
    # "backslash",
    # "LB",
    # "RB",
    "LP",
    "RP",
    "LP1",
    "RP1",
    "star",
    "D",
)

# t_NUM = r"([1-9][0-9]+)|[0-9]"

t_Dot = "\."

# t_backslash = "\\\\" # "\\"

# t_LB = "\(" # "("

# t_RB = "\)" # ")"

t_LP = "\[" # "["

t_RP = "\]" # "]"

t_LP1 = "\{" # "{"

t_RP1 = "\}" # "}"

t_star = "\*" # "*"

t_D = "TEST" # "\d"

t_backslash_LB = "\(" # "\("

t_ignore = ' \t'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

lexer.input("\(")
while True:
    tokens = lexer.token()
    if tokens is None:
        break
    
    print(tokens)