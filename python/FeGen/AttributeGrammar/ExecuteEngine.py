from types import FunctionType
from typing import Type, Dict
from .Rule import *

class CodeGen:
    def __init__(self, ):
        self.__dispatch_dict : Dict[Type, FunctionType] = {
            TerminalRule: self.visitTerminalRule,
            str: self.visitStr
        }
        
        
    def __call__(self, prod):
        return self.visit(prod)
        
        
    def visit(self, prod):
        ty = type(prod)
        if not ty in self.__dispatch_dict:
            raise RuntimeError(f"Can not find visit function for {ty.__name__}")
        return self.__dispatch_dict[ty](prod)
    
    def visitTerminalRule(self, rule: TerminalRule):
        return rule.name
    
    def visitStr(self, s: str):
        return f"\"{s}\""
