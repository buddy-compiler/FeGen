from typing import Type, List, Dict, Literal, Callable, Tuple, Any
from types import FunctionType, MethodType
import inspect
import copy
import ast as py_ast
import astunparse
from .RuleDefinationTransformer import LexLexTransformer, ParseParseTransformer, LexSemaTransformer, ParseSemaTransformer, ExecutionTimeError
import logging



class FeGenGrammar:
    """
        base class of all grammar class
    """    
    def __update_rules(self):
        # declare temporary rules used in fake execution for rules generating
        temp_lexer = lambda self: TerminalRule()
        temp_parser = lambda self: ParserRule()
        for name in ExecutionEngine.lexerRule.keys():
            setattr(self, name, MethodType(temp_lexer, self))
        for name in ExecutionEngine.parserRule.keys():
            setattr(self, name, MethodType(temp_parser, self))
    
    def lexer(self):
        from .ExecuteEngine import CodeGen
       
        """return lexer of attribute grammar
        """
        
        # process source code and generate code for lexer
        ExecutionEngine.WHEN = "lex"
        for name, lexer_rule_defination in ExecutionEngine.lexerRule.items():
            # collect file, line and col
            lines, start_line = inspect.getsourcelines(lexer_rule_defination)
            file = inspect.getsourcefile(lexer_rule_defination)
            source = ''.join(lines)
            source, col_offset = eliminate_indent(source)
            source, inc_lineno = eliminate_decorators(source)
            start_line += inc_lineno
            
            # get ast of source code
            parsed: py_ast.AST = py_ast.parse(source=source)
            anotherParsed = copy.copy(parsed) 
            
            # collect env
            env: Dict[str, Any] = lexer_rule_defination.__globals__.copy()
            func_freevar_names: List[str] = list(lexer_rule_defination.__code__.co_freevars)
            func_freevar_cells: List[Any] = [v.cell_contents for v in lexer_rule_defination.__closure__] if lexer_rule_defination.__closure__ else []
            assert len(func_freevar_names) == len(func_freevar_cells)
            env.update(dict(zip(func_freevar_names, func_freevar_cells)))
            env.update({"self": self})
            self.__update_rules()
            
            # call transformer
            lexlextrans = LexLexTransformer(
                when=ExecutionEngine.WHEN, func_name=name, file=file, start_lineno=start_line, start_column=col_offset, env=env
            )
            lexlexfuncast = lexlextrans.visit(parsed)
            lexsemafunc_code_str = astunparse.unparse(lexlexfuncast)
            lexsemafunc_code = compile(lexsemafunc_code_str, f"lexer_rule_{name}", mode = "exec")
            exec(lexsemafunc_code, env)
            lex_func: FunctionType = env[name]
            
            # set name for return of lex_func
            def lex_func_decorator(lex_func: FunctionType, name: str):
                def warpper(*args, **kwargs):
                    g = lex_func(*args, **kwargs)
                    assert isinstance(g, TerminalRule) 
                    g.name = name
                    return g
                return warpper
            
            # store lex function to ExecutionEngine
            decorated_lex_func = lex_func_decorator(lex_func, name)
            ExecutionEngine.lexerRulesWhenLex[name] = decorated_lex_func
            logging.debug(f"Code generated for lex function {name}: " + lexsemafunc_code_str)
            
        # update lex functions of self
        for name, lex_func in ExecutionEngine.lexerRulesWhenLex.items():
            setattr(self, name, MethodType(lex_func, self))
        
        # generate lex rules
        for name, lexDef in ExecutionEngine.lexerRulesWhenLex.items():
            lexRule = lexDef(self)
            assert isinstance(lexRule, TerminalRule)
            prod = lexRule.production
            gen = CodeGen()
            res = gen(prod)
            print(f"{name}: {res}")
            # TODO
    
    def parser(self) -> str:
        """
        see defination in attr_grammar.parser
        """
        ...


class Production:
    pass


class ExecutionEngine:
    """Stores global variables
    """
    WHEN: Literal['lex', 'parse', 'sema'] = "lex"
    GRAMMAR_CLS = None
    # method that decorated by lexer
    lexerRule : Dict[str, Callable] = {}
    # method that decorated by parser
    parserRule : Dict[str, Callable] = {}
    
    parserRulesParse : Dict[str, Callable] = {}
    parserRulesWhenSema : Dict[str, Callable] = {}
    lexerRulesWhenLex : Dict[str, Callable] = {}
    lexerRulesWhenSema : Dict[str, Callable] = {}
    skipRules : Dict[str, Callable] = {}

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


def parser(parser_rule_defination: FunctionType):
    def warp(*args, **kwargs):
        name = parser_rule_defination.__name__
        ExecutionEngine.parserRulesParse[name] = parser_rule_defination
        return parser_rule_defination(*args, **kwargs)
    return warp


def lexer(lexer_rule_defination: FunctionType):
    """An decorator to mark a function in a subclass of FeGenGrammar that defines a lex rule, ExecutionEngine.WHEN decides action of function.

    Args:
        lexer_rule_defination (FunctionType): lex defining function
    """
    # assert function only have one parameter: self
    sig = inspect.signature(lexer_rule_defination).parameters
    assert len(sig) == 1 and f"lexer defining function should only have one parameter: self"
    name = lexer_rule_defination.__name__
    ExecutionEngine.lexerRule[name] = lexer_rule_defination
    return lexer_rule_defination


def skip(skip_rule_defination):
    def warp(*args, **kwargs):
        name = skip_rule_defination.__name__
        ExecutionEngine.lexerRulesWhenLex[name] = skip_rule_defination
        return skip_rule_defination(*args, **kwargs)
    return warp


class ExecutableWarpper:
    def __init__(self, executable: FunctionType, whenexecute: Literal['parse', 'sema']):
        self.executable = executable
        self.whenexecute = whenexecute

    def __call__(self, *args, **kwds):
        if ExecutionEngine.WHEN == self.whenexecute:
            return self.executable(*args, **kwds)
        else:
            raise ExecutionTimeError(self.executable, f"Function execute in wrong time, expected in {self.whenexecute} time, but now it is time to {ExecutionEngine.WHEN}")

def execute_when(when: Literal['lex', 'parse', 'sema']):
    """mark function to execute at correct time
    """
    def decorator(func):
        f = lambda *args, **kwds: ExecutableWarpper(func, when)(*args, **kwds)
        setattr(f, "execute_when", when)
        return f
    return decorator

class ChatSet(Production):
    """
        char_set("A-Z") --> [A-Z]
    """
    def __init__(self, charset: str):
        super().__init__()
        self.charset = charset

def char_set(charset: str):
    return ChatSet(charset)

class ZeroOrMore(Production):
    """
        zero_or_more(A) --> A*
    """
    def __init__(self, rule: "Rule"):
        super().__init__()
        self.rule = rule

def zero_or_more(rule: "Rule"):
    return ZeroOrMore(rule)


class OneOrMore(Production):
    """
        one_or_more(A) --> A+ 
    """
    def __init__(self, rule: "Rule"):
        super().__init__()
        self.rule = rule

def one_or_more(rule: "Rule"):
    return OneOrMore(rule)


class Concat(Production):
    """
        concat(A, B) -->  A B
    """
    def __init__(self, *args):
        super().__init__()
        self.rules : List[Rule] = args

def concat(*args):
    return Concat(*args)

class Alternate(Production):
    """
        alternate(A, B) --> A | B
    """
    def __init__(self, *args):
        super().__init__()
        self.alts : List[FunctionType] = args


def alternate(*args):
    return Alternate(*args)


class Attribute:
    def __init__(self, name: str, ty: Type, init = None):
        self.name = name
        self.ty = ty
        self.value = init

    def set(self, value):
        assert (isinstance(value, self.ty) or value is None) and f"mismatch type."
        self.value = value

class Rule:
    def __init__(self, production = None):
        self.production = None
        self.name = "UNKNOWN"
        self.setProduction(production)
        
    def setProduction(self, prod):
        self.production = prod


class ParserRule(Rule):
    
    def __init__(self, production = None):
        super().__init__(production)
        self.attributes : Dict[str, Attribute] = []
        self.visited = False
        self.children : List[Rule] = []
    
    @execute_when("parse")
    def setProduction(self, prod):
        super().setProduction(prod)
        #TODO: generate PLY parser function
        if isinstance(prod, "Production"):
            pass
        elif isinstance(prod, ParserRule):
            pass
        elif isinstance(prod, TerminalRule):
            pass
        else:
            raise RuntimeError("unknown prod")
    
    
    @execute_when("sema")
    def new_attr(self, name: str, ty: Type, init = None):
        assert name not in self.attributes and f"Attribute {name} already exists."
        attr = Attribute(name, ty, init)
        self.attributes[name] = attr
        return attr
    
    
    @execute_when("sema")
    def set_attr(self, name: str, value):
        assert name not in self.attributes and f"Attribute {name} does exist."
        attr = self.attributes[name]
        attr.set(value)
    
    
    @execute_when("sema")
    def get_attr(self, name: str):
        assert name in self.attributes and f"Attribute {name} does not exist."
        if not self.visited:
            self.visit()
        return self.attributes[name]

    
    @execute_when("sema")
    def visit(self):
        """
            visit tree node 
        """
        self.visited = True
        # TODO: VISIT

class TerminalRule(Rule):
    def __init__(self, production = None):
        super().__init__(production)

    
    @execute_when("lex")
    def setProduction(self, prod):
        return super().setProduction(prod)
        
        
    @execute_when("sema")
    def text(self) -> str:
        print("get text: not implemented.")

@execute_when("parse")
def newParserRule() -> ParserRule:
    return ParserRule()

@execute_when("lex")
def newTerminalRule(prod = None) -> TerminalRule:
    return TerminalRule(prod)
    
    



