import ast as py_ast
from ast import unparse
import os.path

from hidet import script
import inspect
from typing import Tuple, Any, Dict, List, Optional

def eliminate_indent(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    indent = len(source)
    for line in lines:
        if len(line.strip()) == 0:
            continue
        indent = min(indent, len(line) - len(line.lstrip()))
    source = '\n'.join([line[indent:] for line in lines])

    
    return source, indent


def eliminate_decorators(source: str) -> Tuple[str, int]:
    lines = source.split('\n')
    num_decorators = 0
    for line in lines:
        if len(line) > 0 and line[0] == '@':
            num_decorators += 1
        else:
            break
    source = '\n'.join(lines[num_decorators:])
    return source, num_decorators


class Scope:
    def __init__(self, parent: Optional["Scope"]):
        self.parent: Optional[Scope] = parent
        self.name2var = {}
        self.name2host_var: Dict[str, Any] = {}
        self.stmts = []
        self.attributes: dict[str, Any] = {}

    @staticmethod
    def default_top_level():
        scope = Scope(None)
        return scope

    def define_var(self, name: str, v):
        if name == '_':
            # ignore anonymous variable '_'
            return
        self.name2var[name] = v

    def define_host_var(self, name: str, value: Any):
        self.name2host_var[name] = value

    def lookup(self, name: str, search_parents=True):
        if name in self.name2var:
            return self.name2var[name]
        if name in self.name2host_var:
            return self.name2host_var[name]
        if search_parents and self.parent:
            return self.parent.lookup(name, search_parents)
        return None

    def annotate(self, name: str, value: Any):
        if name in self.attributes:
            raise ValueError('Attribute {} has already been annotated.'.format(name))
        self.attributes[name] = value

    def append(self, stmt):
        self.stmts.append(stmt)

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



class PythonAstFunctor:
    def __init__(self, file: str, start_lineno: int, start_column: int, env, func_annotations):
        self.file: str = file
        self.start_lineno: int = start_lineno
        self.start_column: int = start_column
        self.env: Dict[str, Any] = env
        self.func_annotations: Dict[str, Any] = func_annotations
        self.scope_stack: ScopeStack = ScopeStack()

        
    def __call__(self, node):
        return self.visit(node)

    def visit(self, node):
        from hidet.ir.library.tune import ScheduleError

        method = 'visit_' + node.__class__.__name__
        if hasattr(self, method):
            visitor = getattr(self, method)
        else:
            msg = 'The AST node {} is not supported.'.format(node.__class__.__name__)
            raise RuntimeError(msg)

        try:
            return visitor(node)
        except Exception as e:
            # import traceback
            raise RuntimeError('Internal exception occurred during transpiling this ast node.') from e

    def visit_Module(self, module: py_ast.Module):
        raise NotImplementedError()

    def visit_FunctionDef(self, func_def: py_ast.FunctionDef):
        raise NotImplementedError()

    def visit_Return(self, stmt: py_ast.Return):
        raise NotImplementedError()

    def visit_Assign(self, stmt: py_ast.Assign):
        raise NotImplementedError()

    def visit_AnnAssign(self, stmt: py_ast.AnnAssign):
        raise NotImplementedError()

    def visit_AugAssign(self, stmt: py_ast.AugAssign):
        raise NotImplementedError()

    def visit_For(self, stmt: py_ast.For):
        raise NotImplementedError()

    def visit_While(self, stmt: py_ast.While):
        raise NotImplementedError()

    def visit_If(self, stmt: py_ast.If):
        raise NotImplementedError()

    def visit_With(self, stmt: py_ast.With):
        raise NotImplementedError()

    def visit_Assert(self, stmt:py_ast.Assert):
        raise NotImplementedError()

    def visit_Expr(self, stmt: py_ast.Expr):
        raise NotImplementedError()

    def visit_Pass(self, stmt: py_ast.Pass):
        raise NotImplementedError()

    def visit_Break(self, stmt: py_ast.Break):
        raise NotImplementedError()

    def visit_Continue(self, stmt: py_ast.Continue):
        raise NotImplementedError()

    def visit_BoolOp(self, expr: py_ast.BoolOp):
        raise NotImplementedError()

    def visit_BinOp(self, expr: py_ast.BinOp):
        raise NotImplementedError()

    def visit_UnaryOp(self, expr: py_ast.UnaryOp):
        raise NotImplementedError()

    def visit_Lambda(self, expr: py_ast.Lambda):
        raise NotImplementedError()

    def visit_IfExp(self, expr: py_ast.IfExp):
        raise NotImplementedError()

    def visit_Compare(self, expr: py_ast.Compare):
        raise NotImplementedError()

    def visit_Call(self, expr: py_ast.Call):
        raise NotImplementedError()

    def visit_Constant(self, expr: py_ast.Constant):
        raise NotImplementedError()

    def visit_Num(self, expr: py_ast.Num):
        return self.visit(py_ast.copy_location(py_ast.Constant(expr.n), expr))

    def visit_Str(self, expr: py_ast.Str):
        return self.visit(py_ast.copy_location(py_ast.Constant(expr.s), expr))

    def visit_NameConstant(self, expr: py_ast.NameConstant):
        return self.visit(py_ast.copy_location(py_ast.Constant(expr.value), expr))

    def visit_Attribute(self, expr: py_ast.Attribute):
        raise NotImplementedError()

    def visit_Subscript(self, expr: py_ast.Subscript):
        raise NotImplementedError()

    def visit_Starred(self, expr: py_ast.Starred):
        raise NotImplementedError(unparse(expr))

    def visit_Name(self, expr: py_ast.Name):
        raise NotImplementedError()

    def visit_Tuple(self, expr: Tuple):
        raise NotImplementedError()

    def visit_List(self, expr: List):
        raise NotImplementedError()

    def visit_Slice(self, expr: py_ast.Slice):
        raise NotImplementedError()

    def visit_ExtSlice(self, expr: py_ast.ExtSlice):
        raise NotImplementedError()

    def visit_Index(self, expr: py_ast.Index):
        raise NotImplementedError()

    def visit_ListComp(self, expr: py_ast.ListComp):
        raise NotImplementedError()

    def visit_SetComp(self, expr: py_ast.SetComp):
        raise NotImplementedError()

    def visit_DictComp(self, expr: py_ast.DictComp):
        raise NotImplementedError()

    def visit_GeneratorExp(self, expr: py_ast.GeneratorExp):
        raise NotImplementedError()

    def visit_Nonlocal(self, stmt: py_ast.Nonlocal):
        raise NotImplementedError()

def mywarpper(func):
    lines, start_line = inspect.getsourcelines(func)
    file = inspect.getsourcefile(func)
    source = ''.join(lines)
    source, col_offset = eliminate_indent(source)
    source, inc_lineno = eliminate_decorators(source)
    start_line += inc_lineno
    parsed: py_ast.AST = py_ast.parse(source=source)
    print(parsed.__str__())
    # env: Dict[str, Any] = func.__globals__.copy()
    # func_freevar_names: List[str] = list(func.__code__.co_freevars)
    # func_freevar_cells: List[Any] = [v.cell_contents for v in func.__closure__] if func.__closure__ else []
    # assert len(func_freevar_names) == len(func_freevar_cells)
    # env.update(dict(zip(func_freevar_names, func_freevar_cells)))
    # func_annotations: Dict[str, Any] = func.__annotations__
    # translator = PythonAstFunctor(
    #     file=file, start_lineno=start_line, start_column=col_offset, env=env, func_annotations=func_annotations
    # )
    # new_function = translator(parsed)
    return func


@mywarpper
def add(a, b):
    return a + b

print(add(1, 2))

