from typing import Dict, List
import ply.lex as lex
import ply.yacc as yacc
import copy
from types import ModuleType

mymodule = ModuleType("mymodule", "generated in module.py")
setattr(mymodule, "__file__", __file__)
setattr(mymodule, "__package__", __package__)
d = mymodule.__dict__

d.update({
    "tokens": ('TEST', 'TEST1'),
    "t_TEST": r'test',
    "t_TEST1": r'test1',
    "t_ignore": ' \t',
    "t_error": lambda t: t.lexer.skip(1)
})

lexer = lex.lex(mymodule)

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
    d.update({p_.__name__: p_})
    # module.update({p_.__name__: p_})


generate_concat("p_test", "pptest", ["TEST", "TEST1"])

d.update({"p_error": lambda p: print("Syntax error in input!")})

parser = yacc.yacc(module=mymodule)

s = "test test1"
result = parser.parse(s, lexer)
print(result)