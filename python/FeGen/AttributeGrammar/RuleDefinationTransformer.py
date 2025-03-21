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

class GrammarCodeConvertor(py_ast.NodeTransformer):
    """lex rules generate lex definations
    """
    def __init__(self, when: str, func_name: str, file: str, start_lineno: int, start_column: int, global_env: dict):
        self.when: str = when
        self.func_name: str = func_name
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.global_scope: Scope =  Scope(global_env)
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
            logging.debug(f"""
File "{self.file}", line {self.start_lineno}, col {self.start_column},
Function "{self.func_name}",
When generating {self.when}:
Remove statement: "{codestr}".
It may caused by FeGen, or some error in code, exception details: 
{traceinfo}
                        """)
        return False
    
    
    def split_parse_sema(self, node: py_ast.AST) -> Tuple[py_ast.AST, py_ast.AST]:
        assert isinstance(node, py_ast.Module)
        parse_node = node
        sema_node = copy.deepcopy(node)
        return self.visit_Module(parse_node, sema_node)
        
    
    def visit_Module(self, parse_node: py_ast.Module, sema_node: py_ast.Module):
        for parse_consist, sema_consist in zip(parse_node.body, sema_node.body):
            # only change function body
            if isinstance(parse_consist, py_ast.FunctionDef):
                assert len(self.scopestack) == 1
                assert isinstance(sema_consist, py_ast.FunctionDef)
                
                self.push()
                _self = self.global_scope.loopup("self")
                assert _self is not None
                self.current_scope.insert("self", _self)
                self.visit_FunctionDef(parse_consist, sema_consist)
                self.pop()
                
        return parse_node, sema_node


    def visit_FunctionDef(self, parse_node: py_ast.FunctionDef, sema_node: py_ast.Module):
        parse_newbody = []
        sema_newbody = []

        for stmt in parse_node.body:
            if isinstance(stmt, py_ast.Return): # return stmt only insert to parse body
                parse_newbody.append(stmt)
                break
            elif isinstance(stmt, py_ast.FunctionDef):
                self.push()
                parse_func, sema_func = self.visit_FunctionDef(stmt, copy.deepcopy(stmt))
                self.pop()
                flag = self.try_exec(parse_func)
                assert flag
                parse_newbody.append(parse_func)
                sema_newbody.append(sema_func)
            elif isinstance(stmt, py_ast.Assign):
                # declare variables defined in parse/lex as global variable 
                if(self.try_exec(stmt)):
                    parse_newbody.append(stmt)
                    targets = stmt.targets
                    global_variables = []
                    for t in targets:
                        if isinstance(t, py_ast.Name):
                            global_variables.append(t.id)
                    global_stmt = py_ast.Global(global_variables)
                    sema_newbody.append(global_stmt)
                else:
                    sema_newbody.append(stmt)
            else:
                if(self.try_exec(stmt)):
                    parse_newbody.append(stmt)
                else:
                    sema_newbody.append(stmt)
                    
                    
        parse_node.body = parse_newbody
        if len(sema_newbody) == 0:
            sema_newbody.append(py_ast.Pass())
        sema_node.body = sema_newbody
        return parse_node, sema_node