import ast as py_ast
from typing import Tuple, Dict, List, Any, Type
from types import FunctionType


class SemaRelatedChecker:
    """check if the node expressed value is Sema Related 
    """
    def __init__(self, env, semaRelate):
        self.env: Dict[str, Any] = env
        self.semaRelate : List[Any] = semaRelate
        self.__dispatch_dict: Dict[Type, FunctionType] = {
            py_ast.Call, self.visit_Call
        }

    def __call__(self, node):
        return self.visit(node)

    def visit(self, node):
        return self.__dispatch_dict[type(node)]()
    
    def visit_Call(self, node: py_ast.Call):
        pass

    

class LexLexTransformer(py_ast.NodeTransformer):
    """lex rules generate lex definations
    """
    def __init__(self, file: str, start_lineno: int, start_column: int, env):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] = env
        self.semaRelate : List[Any] = []
        self.checker = SemaRelatedChecker(env, self.semaRelate)


    def visit_Module(self, node: py_ast.Module):
        for consist in node.body:
            # only change function body
            if isinstance(consist, py_ast.FunctionDef):
                self.generic_visit(consist)
        return node

    def visit_FunctionDef(self, node: py_ast.FunctionDef):
        for stmt in node.body:
            self.generic_visit(stmt)
        return node

    def visit_Assign(self, node: py_ast.Assign):
        targets = node.targets
        value = node.value
        if self.checker(value):
            self.semaRelate.append(targets)
            return None
        else:
            return node

    
class ParseParseTransformer(py_ast.NodeTransformer):
    """parse rules generate parse definations
    """
    def __init__(self, file: str, start_lineno: int, start_column: int, env):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] = env
    

class LexSemaTransformer(py_ast.NodeTransformer):
    """lex rules generate sema definations
    """
    def __init__(self, file: str, start_lineno: int, start_column: int, env):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] = env
    
    
    
class ParseSemaTransformer(py_ast.NodeTransformer):
    """parse rules generate sema defination
    """
    def __init__(self, file: str, start_lineno: int, start_column: int, env):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] = env
    