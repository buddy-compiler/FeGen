from types import FunctionType
from typing import Type, Dict
from .Rule import *

class CodeGenError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class CodeGen:
    def __init__(self, ):
        pass
        
        
    def __call__(self, prod):
        return self.visit(prod)
        
        
    def visit(self, prod) -> str:
        visitor_name = "visit_" + prod.__class__.__name__
        visitor = getattr(self, visitor_name)
        if visitor is None:
            raise CodeGenError(f"Can not find visit function: {visitor_name}")
        return visitor(prod)
    
    def visit_TerminalRule(self, rule: TerminalRule):
        return rule.name
    
    def visit_str(self, s: str):
        return f"\"{s}\""

    def visit_OneOrMore(self, prod: one_or_more):
        return self.visit(prod.rule) + "*"