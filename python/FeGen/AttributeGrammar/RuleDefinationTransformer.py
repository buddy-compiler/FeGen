from __future__ import annotations
import ast as py_ast
from typing import Tuple, Dict, List, Any, Type, Optional
from types import FunctionType
import inspect
import copy
import logging

class ExecutionTimeError(Exception):
    def __init__(self, func: FunctionType, msg: str):
        self.func = func
        self.func_name = func.__name__
        self.func_loc = inspect.getfile(func)
        self.message = f"{msg}: Function '{self.func_name}' defined in '{self.func_loc}'"
        super().__init__(self.message)  # make this exception picklable


class LexLexTransformer(py_ast.NodeTransformer):
    """lex rules generate lex definations
    """
    def __init__(self, when: str, func_name: str, file: str, start_lineno: int, start_column: int, env):
        self.when: str = when
        self.func_name: str = func_name
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] =  env


    def visit_Module(self, node: py_ast.Module):
        for consist in node.body:
            # only change function body
            if isinstance(consist, py_ast.FunctionDef):
                self.visit_FunctionDef(consist)
        return node

    def visit_FunctionDef(self, node: py_ast.FunctionDef):
        newbody = []
        for stmt in node.body:
            if isinstance(stmt, py_ast.Return):
                newbody.append(stmt)
                break
            code  = compile(py_ast.Module(body=[stmt], type_ignores=[]), filename="fake_execution", mode="exec")
            try:
                exec(code, self.env)
                newbody.append(stmt)
            except ExecutionTimeError as e:
                codestr = py_ast.unparse(stmt)
                logging.debug(f"""
File "{self.file}", line {self.start_lineno}, col {self.start_column},
Function "{self.func_name}",
When generating {self.when}:
Remove statement: "{codestr}" 
                            """)
            except Exception as e:
                codestr = py_ast.unparse(stmt)
                logging.warning(f"""
File "{self.file}", line {self.start_lineno}, col {self.start_column},
Function "{self.func_name}",
When generating {self.when}:
Remove statement: "{codestr}".
It may caused by FeGen, or some error in code, exception details: 
{e}
                              """)
        node.body = newbody
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
    def __init__(self, func_name: str,  file: str, start_lineno: int, start_column: int, env):
        self.func_name = func_name
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
    