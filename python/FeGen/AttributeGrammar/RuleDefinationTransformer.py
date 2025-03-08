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


class Obj:
    def __init__(self, name, isfunc, semarelated):
        self.name = name
        self.isfunc = isfunc
        self.semarelated = semarelated
            
class Scope:
    def __init__(self, parent: Optional[Scope]):
        self.parent: Optional[Scope] = parent
        self.name2obj: Dict[str, Obj] = {}

    @staticmethod
    def default_top_level():
        scope = Scope(None)
        return scope

    def define_var(self, name: str, v: Obj):
        self.name2obj[name] = v

    def lookup(self, name: str, search_parents=True) -> Optional[Obj]:
        if name in self.name2obj:
            return self.name2obj[name]
        if search_parents and self.parent:
            return self.parent.lookup(name, search_parents)
        return None


class ScopeStack:
    def __init__(self):
        self.scopes: list[Scope] = [Scope.default_top_level()]

    def __enter__(self) -> Scope:
        parent = self.scopes[-1]
        scope = Scope(parent)
        self.scopes.append(scope)
        return scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()

class SemaRelatedChecker(py_ast.NodeVisitor):
    """return True if the node expressed value is Sema Related 
    """
    def __init__(self, env, semaRelate):
        self.env: Dict[str, Any] = env
        self.semaRelate : List[Any] = semaRelate
        self.scope: Scope = None
        
        
    def __call__(self, node, scope: Scope):
        self.scope = scope
        return self.visit(node)
    
    def visit_Call(self, node: py_ast.Call):
        if self.visit(node.func):
            return True
        for arg in node.args:
            if self.visit(arg):
                return True
        return False

    def visit_Constant(self, node: py_ast.Constant):
        return True
            
    def visit_Name(self, node: py_ast.Name):
        iden = node.id
        scope_res = self.scope.lookup(iden)
        if scope_res is not None and scope_res.semarelated:
            return True
        if iden in self.env:
            env_res = self.env[iden]
            if hasattr(env_res, "execute_when") and getattr(env_res, "execute_when") == "sema":
                return True
        return False

class LexLexTransformer(py_ast.NodeTransformer):
    """lex rules generate lex definations
    """
    def __init__(self, when: str, func_name: str, file: str, start_lineno: int, start_column: int, env):
        self.when: str = when
        self.func_name: str = func_name
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] =  copy.copy(env)

    def scope(self):
        """return scope stack
        with self.scope() as env_scope:
            ...
        """
        return self.scope_stack

    @property
    def current_scope(self) -> Scope:
        if len(self.scope_stack.scopes) == 0:
            raise ValueError('The scope stack is empty.')
        return self.scope_stack.scopes[-1]

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
When generating {self.when}er:
Remove statement: "{codestr}" 
                            """)
            except Exception as e:
                codestr = py_ast.unparse(stmt)
                logging.warning(f"""
File "{self.file}", line {self.start_lineno}, col {self.start_column},
Function "{self.func_name}",
When generating {self.when}er:
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
    