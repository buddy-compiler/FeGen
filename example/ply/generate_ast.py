from typing import Dict, List
import ply.lex as lex
import ply.yacc as yacc
import copy

def test():
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
        return t

    t_ignore = ' \t'

    def t_error(t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    lexer = lex.lex()

    def generate_concat(name:str, rule: str, elements: List[str]):
        def p_(p):
            assert len(p) - 1 == len(elements)
            d = dict()
            for i in range(len(elements)):
                elem = elements[i]
                pi = p[i + 1]
                d.update({elem: pi})
                
            p[0] = d
        p_.__name__ = p_.__name__ + name
        p_.__doc__ = rule + " : " + " ".join(elements)
        globals().update({p_.__name__: p_})
        # module.update({p_.__name__: p_})


    generate_concat("p_expression_plus", "expression", ["expression", "PLUS", "term"])

    generate_concat("p_expression_minus", "expression", ["expression", "MINUS", "term"])

    generate_concat("p_expression_term", "expression", ["term"])

    generate_concat("p_term_times", "term", ["term", "TIMES", "factor"])

    generate_concat("p_term_divide", "term", ["term", "DIVIDE", "factor"])

    generate_concat("p_term_factor", "term", ["factor"])

    generate_concat("p_factor_number", "factor", ["NUMBER"])

    def p_error(p):
        print("Syntax error in input!")

    parser = yacc.yacc()

    s = "1 + 2 * 3"
    result = parser.parse(s, lexer)
    print(result)
    
test()