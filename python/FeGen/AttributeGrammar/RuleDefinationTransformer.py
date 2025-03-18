from __future__ import annotations
import ast as py_ast
from typing import Tuple, Dict, List, Any, Type, Optional
from types import FunctionType
import inspect
import copy
import logging
import traceback


class ExecutionTimeError(Exception):
    def __init__(self, func: FunctionType, msg: str):
        self.func = func
        self.func_name = func.__name__
        self.func_loc = inspect.getfile(func)
        self.message = f"{msg}: Function '{self.func_name}' defined in '{self.func_loc}'"
        super().__init__(self.message)  # make this exception picklable

class Scope:
    def __init__(self, table):
        self.table: Dict[str, Any] = table
    
    def insert(self, name: str, obj: Any):
        self.table.update({name: obj})
    
    def loopup(self, name: str):
        if name in self.table:
            return self.table[name]
        return None

class LexLexTransformer(py_ast.NodeTransformer):
    """lex rules generate lex definations
    """
    def __init__(self, when: str, func_name: str, file: str, start_lineno: int, start_column: int, env):
        self.when: str = when
        self.func_name: str = func_name
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.global_scope: Scope =  Scope(env)
        self.scopestack : List[Scope] = [self.global_scope]

    @property
    def current_scope(self):
        return self.scopestack[-1]

    def push(self):
        self.scopestack.append(Scope({}))

    def pop(self):
        self.scopestack.pop()

    def get_globals(self) -> Dict[str, Any]:
        scopes = self.scopestack[:-1]
        env = {}
        for scope in scopes:
            env.update(scope.table)
        return env
    
    def get_locals(self) -> Dict[str, Any]:
        return self.current_scope.table

    def try_exec(self, stmt) -> bool:
        """try execute stmt, return True if success, else return False
        
        """
        code  = compile(py_ast.Module(body=[stmt], type_ignores=[]), filename="fake_execution", mode="exec")
        try:
            exec(code, self.get_globals(), self.get_locals())
            return True
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
            traceinfo = traceback.format_exc()
            logging.warning(f"""
File "{self.file}", line {self.start_lineno}, col {self.start_column},
Function "{self.func_name}",
When generating {self.when}:
Remove statement: "{codestr}".
It may caused by FeGen, or some error in code, exception details: 
{traceinfo}
                        """)
        return False
    
    
    def visit_Module(self, node: py_ast.Module):
        for consist in node.body:
            # only change function body
            if isinstance(consist, py_ast.FunctionDef):
                assert len(self.scopestack) == 1
                self.push()
                _self = self.global_scope.loopup("self")
                assert _self is not None
                self.current_scope.insert("self", _self)
                self.visit_FunctionDef(consist)
                self.pop()
        return node

    def visit_FunctionDef(self, node: py_ast.FunctionDef):
        newbody = []
        for stmt in node.body:
            if isinstance(stmt, py_ast.Return):
                newbody.append(stmt)
                break
            elif isinstance(stmt, py_ast.FunctionDef):
                self.push()
                newfunc = self.visit(stmt)
                self.pop()
                flag = self.try_exec(newfunc)
                assert flag
                newbody.append(newfunc)
            else:
                if(self.try_exec(stmt)):
                    newbody.append(stmt)
                    
                    
        node.body = newbody
        return node





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
    