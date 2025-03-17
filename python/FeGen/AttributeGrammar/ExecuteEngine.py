from types import FunctionType
from typing import Type, Dict
from .Rule import *

class CodeGenError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class BaseCodeGen:
    def __init__(self):
        pass
    
    def __call__(self, prod):
        return self.visit(prod)
        
    def visit(self, prod) -> str:
        visitor_name = "visit_" + prod.__class__.__name__
        visitor = getattr(self, visitor_name)
        if visitor is None:
            raise CodeGenError(f"Can not find visit function: {visitor_name}")
        return visitor(prod)

class LexerProdGen(BaseCodeGen):   
    def visit_TerminalRule(self, rule: TerminalRule):
        return rule.name
    
    def visit_str(self, s: str):
        return s

    def visit_OneOrMore(self, prod: Concat):
        return self.visit(prod.rule) + "*"
    
class ParserProdGen(BaseCodeGen):
    def __init__(self, attr_dict):
        super().__init__()
        self.attr_dict : Dict[str, Any] = attr_dict
        
        
    def __call__(self, rule: ParserRule):
        """insert p functions like
        def p_{rulename}(p):
            "rulename: p_name"
            p[0] = p[1]
        """
        assert isinstance(rule, ParserRule)
        self.processing_rule = rule
        rulename = rule.name
        prod = rule.production
        p_name = self.visit(prod)
        def p_(p):
            p[0] = {p_name: p[1]}
        p_.__name__ += rulename
        p_.__doc__ = f"{rulename} : {p_name}"
        self.attr_dict.update({p_.__name__: p_}) 
    
    def visit_TerminalRule(self, prod: TerminalRule):
        return prod.name
    
    def visit_ParserRule(self, prod: ParserRule):
        return prod.name
    
    def visit_Concat(self, prod: Concat):
        # visit and get children prod names
        prod_children_names: List[str] = [self.visit(r) for r in prod.rules]
        rule_name = "concat_" + "_".join(prod_children_names)
        # return if rule name is already existed
        if rule_name in self.attr_dict:
            return rule_name
        # define function and insert to attr dict
        def p_(p):
            assert len(p) - 1 == len(prod_children_names)
            d = dict()
            for i in range(len(prod_children_names)):
                child_name = prod_children_names[i]
                pi = p[i + 1]
                d.update({child_name: pi})
            p[0] = d
        p_.__name__ += rule_name
        p_.__doc__ = "{rule_name} : {prod_children}".format(rule_name=rule_name, prod_children = " ".join(prod_children_names))
        self.attr_dict.update({p_.__name__: p_})
        return rule_name
    # def visit_(self, prod)