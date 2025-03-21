from typing import Type, List, Dict, Literal, Callable, Tuple, Any
from types import FunctionType, MethodType, ModuleType, FrameType, CodeType
import inspect
import copy
import ast as py_ast
import astunparse
from .RuleDefinationTransformer import GrammarCodeConvertor, ExecutionTimeError
import logging  
import sys
import ply.lex as lex
import ply.yacc as yacc
import os
from functools import partial

class LexOrParseError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class FeGenLexer:
    def __init__(self, module):
        self.module = module
        outputdir = os.path.join(os.path.dirname(__file__), self.module.__name__)
        self.lexer = lex.lex(module=self.module, debug=False, outputdir=outputdir)



    def input(self, src: str):
        self.lexer.input(src)
        token_list = []
        while True:
            token = self.lexer.token()
            if not token:
                break
            token_list.append(token)
        return token_list



class ConcreteSyntaxTree:
    def __init__(self, grammar: "FeGenGrammar", root: "ParserRule", parser_locals_dict: Dict["ParserRule", Dict[str, Any]], lexer_locals_dict: Dict["TerminalRule", Dict[str, Any]] ):
        self.grammar = grammar
        self.root = root
        self.parser_locals_dict = parser_locals_dict
        self.lexer_locals_dict = lexer_locals_dict
        self._eval()


    def _eval(self):
        ExecutionEngine.WHEN = "sema"
        for rule, local_dict in self.parser_locals_dict.items():
            rule_name = rule.name
            sema_code = ExecutionEngine.semaCodeForParseRule[rule_name]
            # get globals from lex/parse function
            lexorparse_func: FunctionType = getattr(self.grammar, rule_name)
            assert lexorparse_func is not None
            global_dict = lexorparse_func.__globals__
            # execute to get sema function
            sema_func_global = {**global_dict, **local_dict, "self": self.grammar}
            exec(sema_code, sema_func_global)
            # `sema_func` has function parameter: `self`
            sema_func = sema_func_global[rule_name]
            # bind parameter `self` with self.grammar
            rule.visit_func = partial(sema_func, self=self.grammar)


    def getText(self):
        if ExecutionEngine.WHEN != "sema":
            logging.warning("Method `ConcreteSyntaxTree.getText` should be called after `eval`")
        return self.root.getText()

    def get_attr(self, name: str) -> Any:
        return self.root.get_attr(name)


    def visit(self):
        self.root.visit()

class FeGenParser:
    def __init__(self, module : ModuleType, lexer: FeGenLexer, grammar: "FeGenGrammar", start: str):
        self.module = module
        self.lexer = lexer
        self.grammar = grammar
        self.start = start
        self.start_func : MethodType = getattr(self.grammar, self.start)
        if self.start_func is None:
            raise LexOrParseError("\n\n\nCan not find parse function for start rule: `{start}`, maybe you forgot to decorate `{start}` by `@parser`".format(start=start))

        # create yacc parser
        outputdir = os.path.join(os.path.dirname(__file__), self.module.__name__)
        self.__parser = yacc.yacc(module=self.module, debug=True, outputdir=outputdir, start=start)

        
        
    def __capture_locals(func: FunctionType, *args, **kwargs):
        """execute and capture locals of func"""
        parser_locals_dict: Dict[ParserRule, Dict[str, Any]] = {}
        lexer_locals_dict: Dict[TerminalRule, Dict[str, Any]] = {}
        
        parser_rule_names = list(ExecutionEngine.parserRuleFunc.keys())
        lexer_rule_names = list(ExecutionEngine.lexerRuleFunc.keys())
        
        def trace_function(frame: FrameType, event: str, arg):
            if event == 'return':
                func_name = frame.f_code.co_name
                if func_name in parser_rule_names and isinstance(arg, ParserRule):
                    parser_locals_dict.update({arg: frame.f_locals})
                elif func_name in lexer_rule_names and isinstance(arg, TerminalRule):
                    lexer_locals_dict.update({arg: frame.f_locals})
            return trace_function
        
        
        original_trace = sys.gettrace()
        sys.settrace(trace_function)
        try:
            res = func(*args, **kwargs) # execute
        finally:
            sys.settrace(original_trace)  # resume trace function
        
        return res, parser_locals_dict, lexer_locals_dict
        
        
    def parse(self, code):
        from .ExecuteEngine import ParserTreeBuilder
        raw_data = self.__parser.parse(code)
        ExecutionEngine.WHEN = "gen_ast"
        # Execute self.start_func, generate ParserTree Node,
        # At the same time, collect local variables generated when calling parse/lex function.
        # Because of the true Alternative / ZeroOrMore / OneOrMore / ZeroOrOne matching contents are unknown,
        # so related ParserTree(s) will not generate.
        startrule, parser_locals_dict, lexer_locals_dict = FeGenParser.__capture_locals(self.start_func)
        # Complete Alternative / ZeroOrMore / OneOrMore / ZeroOrOne related ParserTree, including ParserTree Node and local variables generated when calling parse/lex function.
        builder = ParserTreeBuilder(self.grammar)
        builder(startrule, raw_data)
        
        parser_locals_dict.update(builder.parser_locals_dict)
        lexer_locals_dict.update(builder.lexer_locals_dict)
        
        return ConcreteSyntaxTree(self.grammar, startrule, parser_locals_dict, lexer_locals_dict)

        
class FeGenGrammar:
    """
        base class of all grammar class
    """    
    def __init__(self):
        self.plymodule = ModuleType("PLYModule", "Generated by FeGenGrammar")
        # mkdir for plymodule
        outputdir = os.path.join(os.path.dirname(__file__), self.plymodule.__name__)
        # set attr for plymodule
        setattr(self.plymodule, "__file__", outputdir)
        setattr(self.plymodule, "__package__", __package__+".PLYModule")
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        # initialize lexer and parser
        self.lexerobj = None
        self.parserobj = None

    def __update_rules(self, self_copy):
        # declare temporary rules used in fake execution for rules generating
        def generate_template_lexer(name: str):
            return lambda self: TerminalRule(name=name)

        def generate_template_parser(name: str):
            return lambda self: ParserRule(name=name)
            
        for name in ExecutionEngine.lexerRuleFunc.keys():
            setattr(self_copy, name, MethodType(generate_template_lexer(name), self_copy))
        for name in ExecutionEngine.parserRuleFunc.keys():
            setattr(self_copy, name, MethodType(generate_template_parser(name), self_copy))

    
    def __eliminate_indent(source: str) -> Tuple[str, int]:
        lines = source.split('\n')
        indent = len(source)
        for line in lines:
            if len(line.strip()) == 0:
                continue
            indent = min(indent, len(line) - len(line.lstrip()))
        source = '\n'.join([line[indent:] for line in lines])

        
        return source, indent


    def __eliminate_decorators(source: str) -> Tuple[str, int]:
        lines = source.split('\n')
        num_decorators = 0
        for line in lines:
            if len(line) > 0 and line[0] == '@':
                num_decorators += 1
            else:
                break
        source = '\n'.join(lines[num_decorators:])
        return source, num_decorators


    def __convert_functions_and_get_ruletree(self, when: Literal["lex", "parse"]) -> List["TerminalRule"] | List["ParserRule"]:
        """
            Execute to generate rule trees for lex/parse rule generating.
            Productions of ruletree_for_parse will be folded.
            For example, for rule `one_or_more_a_or_b` defined as follow and input A B:
            
            A: 'A'
            B: 'B'
            a_or_b: A | B
            one_or_more_a_or_b: a_or_b+
            
            unfolded ruletree: one_or_more_a_or_b: {'a_or_b': {'A': 'A'}, 'a_or_b': {'B': 'B'}}
            folded ruletree:  one_or_more_a_or_b: {'a_or_b': None, 'a_or_b': None}
            
            Content of prod are folded for avoiding RecursionError which happened when generating parse rules
        """

    
        if when == "lex":
            func_dict = ExecutionEngine.lexerRuleFunc
            code_dict = ExecutionEngine.semaCodeForLexRule
        elif when == "parse":
            func_dict = ExecutionEngine.parserRuleFunc
            code_dict = ExecutionEngine.semaCodeForParseRule

        rule_trees = []
        for name, rule_def_method in func_dict.items():
            # collect file, line and col
            lines, start_line = inspect.getsourcelines(rule_def_method)
            file = inspect.getsourcefile(rule_def_method)
            source = ''.join(lines)
            source, col_offset = FeGenGrammar.__eliminate_indent(source)
            source, inc_lineno = FeGenGrammar.__eliminate_decorators(source)
            start_line += inc_lineno
            
            # get ast of source code
            parsed: py_ast.AST = py_ast.parse(source=source)
            
            # collect env
            global_env: Dict[str, Any] = rule_def_method.__globals__.copy()
            self_copy = copy.copy(self)
            self.__update_rules(self_copy)
            local_env = {"self": self_copy}
            
            # call transformer
            convertor = GrammarCodeConvertor(
                when=ExecutionEngine.WHEN, func_name=name, file=file, start_lineno=start_line, start_column=col_offset, global_env={**global_env, **local_env}
            )
            
            cvted_parseorlex_func_ast, cvted_sema_func_ast = convertor.split_parse_sema(parsed)
            
            cvted_parseorlex_func_code_str = astunparse.unparse(cvted_parseorlex_func_ast)
            cvted_parseorlex_func_code = compile(cvted_parseorlex_func_code_str, f"{when}_rule_{name}", mode = "exec")
            exec(cvted_parseorlex_func_code, rule_def_method.__globals__, local_env)
            parseorlex_func: FunctionType = local_env[name]
            logging.debug(f"Code generated for {when} function {name}: " + cvted_parseorlex_func_code_str)
            
            # execute to generate folded ruletree
            rule_tree = parseorlex_func(self_copy)
            rule_trees.append(rule_tree)
            
            # update functions of self
            setattr(self, name, MethodType(parseorlex_func, self))
            
            cvted_sema_func_code_str = astunparse.unparse(cvted_sema_func_ast)
            cvted_sema_func_code = compile(cvted_sema_func_code_str, f"sema_rule_{name}", mode = "exec") 
            code_dict.update({name: cvted_sema_func_code})
            logging.debug(f"Code generated for sema function {name}: " + cvted_sema_func_code_str)
            
            
        return rule_trees
    
    
    def lexer(self) -> FeGenLexer:
        """return lexer of attribute grammar
        """
        if self.lexerobj is not None:
            return self.lexerobj
        from .ExecuteEngine import LexerProdGen
       
        
        # process source code and generate code for lexer
        ExecutionEngine.WHEN = "lex"
        # generated rule trees for parse rule generating
        ruletrees_for_lex : List[TerminalRule] = self.__convert_functions_and_get_ruletree(ExecutionEngine.WHEN)
        
        attr_dict = self.plymodule.__dict__
        # insert tokens tuple
        attr_dict.update({"tokens": tuple([r.name for r in ruletrees_for_lex])})
        gen = LexerProdGen(ruletrees_for_lex)
        # insert lex defination
        for lexRule in ruletrees_for_lex:
            name = lexRule.name
            prod = lexRule.production
            lexprod = gen(prod)
            logging.debug(f"Regular expression for rule '{name}' is {lexprod}")
            attr_dict.update({"t_" + name:  lexprod})
        # insert ignore and skip
        # TODO: skip
        def t_error(t):
            print(f"Illegal character '{t.value[0]}'")
            t.lexer.skip(1)

        attr_dict.update({"t_error": t_error, "t_ignore": ' \t'})

        # generate PLY lexer
        self.lexerobj = FeGenLexer(self.plymodule)
        return self.lexerobj
    
    def parser(self, lexer: FeGenLexer, start = None) -> FeGenParser:
        if self.parserobj is not None:
            return self.parserobj
        from .ExecuteEngine import ParserProdGen
        # process source code and generate code for lexer
        ExecutionEngine.WHEN = "parse"
        # generated rule trees for parse rule generating
        ruletrees_for_parse : List[ParserRule] = self.__convert_functions_and_get_ruletree(ExecutionEngine.WHEN)
        
        # generate PLY function
        attr_dict = self.plymodule.__dict__
        gen = ParserProdGen(attr_dict)
        for rule in ruletrees_for_parse:
            gen(rule)

        self.parserobj = FeGenParser(self.plymodule, lexer, self, start)
        return self.parserobj



class ExecutionEngine:
    """Stores global variables
    """
    WHEN: Literal['lex', 'parse', 'gen_ast', 'sema'] = "lex"
    GRAMMAR_CLS = None
    # method that decorated by lexer
    lexerRuleFunc : Dict[str, Callable] = {}
    # method that decorated by parser
    parserRuleFunc : Dict[str, Callable] = {}
    # sema methods
    semaCodeForLexRule : Dict[str, CodeType] = {}
    semaCodeForParseRule : Dict[str, CodeType] = {}




def parser(parser_rule_defination):
    """An decorator to mark a function in a subclass of FeGenGrammar that defines a parse rule, ExecutionEngine.WHEN decides action of function.

    Args:
        parser_rule_defination (FunctionType): lex defining function
    """
    # assert function only have one parameter: self
    sig = inspect.signature(parser_rule_defination).parameters
    assert len(sig) == 1, f"lexer defining function should only have one parameter: self"
    name = parser_rule_defination.__name__
    ExecutionEngine.parserRuleFunc[name] = parser_rule_defination
    return parser_rule_defination


def lexer(lexer_rule_defination):
    """An decorator to mark a function in a subclass of FeGenGrammar that defines a lex rule, ExecutionEngine.WHEN decides action of function.

    Args:
        lexer_rule_defination (FunctionType): lex defining function
    """
    # assert function only have one parameter: self
    sig = inspect.signature(lexer_rule_defination).parameters
    assert len(sig) == 1, f"lexer defining function should only have one parameter: self"
    name = lexer_rule_defination.__name__
    ExecutionEngine.lexerRuleFunc[name] = lexer_rule_defination
    return lexer_rule_defination


def skip(skip_rule_defination):
    def warp(*args, **kwargs):
        name = skip_rule_defination.__name__
        return skip_rule_defination(*args, **kwargs)
    return warp


class ExecutableWarpper:
    def __init__(self, executable: FunctionType, whenexecute: Tuple[Literal['lex', 'parse', 'get_ast', 'sema']]):
        self.executable = executable
        self.whenexecute = whenexecute

    def __call__(self, *args, **kwds):
        if ExecutionEngine.WHEN in self.whenexecute:
            return self.executable(*args, **kwds)
        else:
            raise ExecutionTimeError(self.executable, "Function execute in wrong time, expected in {when} time, but now it is time to {time}".format(when=self.whenexecute, time=ExecutionEngine.WHEN))


def execute_when(*when):
    """mark function to execute at correct time
        when can be:
        * lex
        * parse
        * get_ast
        * sema
    """
    for item in when:
        assert isinstance(item, str), "function `execute_when` accepts only string values."
    def decorator(func):
        f = lambda *args, **kwds: ExecutableWarpper(func, when)(*args, **kwds)
        setattr(f, "execute_when", when)
        return f
    return decorator


class ProductionSemaError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)
        
        
class Production:
    def __init__(self):
        self.content = None
        # Production may not exist if it is subprod of ZeroOrOne/ZeroOrMore production.
        self._ifexist = True

    def getText(self):
        raise NotImplementedError()

    def visit(self):
        raise NotImplementedError()

    def get_attr(self, name: str):
        return None

    def set_attr(self, name: str, value: Any):
        raise NotImplementedError()

class RegularExpression(Production):
    """
        regular_expr("[A-Z]") --> [A-Z]
    """
    def __init__(self, re_expr: str):
        super().__init__()
        self.re_expr = re_expr

def regular_expr(re_expr: str):
    return RegularExpression(re_expr)


class OneOrMore(Production):
    """
        one_or_more(A) --> A+ 
    """
    def __init__(self, prod: Production):
        super().__init__()
        self.template_prod = prod
        self.children : List[Production] = [prod]

    @execute_when("sema")
    def getText(self):
        if len(self.children) == 0:
            return ""
        else:
            texts = [child.getText() for child in self.children]
            return " ".join(texts)


    @execute_when("sema")
    def __getitem__(self, index) -> Production:
        if not self._ifexist:
            raise IndexError("Production OneOrMore matches nothing.")
        return self.children[index]


    @execute_when("sema")
    def __iter__(self):
        if not self._ifexist:
            return iter([])
        return iter(self.children)


    @execute_when("sema")
    def get_attr(self, name):
        """Collect attributes from children, if attribute dose not exist in any child, get None.\n
        For example, obj = OneOrMore{has children: [child1{has attr: v=1}, child2{has no attr}, child3{has attr: v=2}]}, then `obj.get_attr("v") == [1, None, 2]`.
        Args:
            name (_str_): Name of attribute.
        """
        if not self._ifexist:
            return None
        res = []
        for child in self.children:
            res.append(child.get_attr(name))
        return res


    @execute_when("sema")
    def set_attr(self, name, value):
        raise ProductionSemaError("OneOrMore instance should have no attributes.")


    @execute_when("sema")
    def visit(self):
        """Visit all children of OneOrMore
        """
        if not self._ifexist:
            return
        for child in self.children:
            child.visit()

def one_or_more(prod: Production):
    return OneOrMore(prod)


class ZeroOrMore(Production):
    """
        zero_or_more(A) --> A*
    """
    def __init__(self, prod: Production):
        super().__init__()
        self.template_prod = prod
        self.children : List[Production] = []


    @execute_when("sema")
    def getText(self):
        texts = [child.getText() for child in self.children]
        return " ".join(texts)


    @execute_when("sema")
    def __getitem__(self, index) -> Production:
        """Return the index-th child

        Args:
            index (_int_): index of child
        """
        if not self._ifexist:
            raise IndexError("Production ZeroOrMore matches nothing.")
        return self.children[index]


    @execute_when("sema")
    def __iter__(self):
        if not self._ifexist:
            return iter([])
        return iter(self.children)


    @execute_when("sema")
    def get_attr(self, name):
        """Collect attributes from children, if attribute dose not exist in any child, get None.\n
        For example, obj = ZeroOrMore{has children: [child1{has attr: v=1}, child2{has no attr}, child3{has attr: v=2}]}, then `obj.get_attr("v") == [1, None, 2]`.

        Args:
            name (_str_): Name of attribute.
        """
        if not self._ifexist:
            return None
        res = []
        for child in self.children:
            res.append(child.get_attr(name))
        return res

    @execute_when("sema")
    def set_attr(self, name, value):
        raise ProductionSemaError("ZeroOrMore instance should have no attributes.")


    @execute_when("sema")
    def visit(self):
        """Visit all children of ZeroOrMore
        """
        if not self._ifexist:
            return
        for child in self.children:
            child.visit()


def zero_or_more(prod: Production):
    return ZeroOrMore(prod)


class ZeroOrOne(Production):
    """
        zero_or_one(A) --> A?
    """
    def __init__(self, prod: Production):
        super().__init__()
        self.prod = prod


    @execute_when("sema")
    def getText(self):
        if self.prod.content is None:
            return ""
        return self.prod.getText()
    
    
    @execute_when("sema")
    def get_attr(self, name):
        """Get Attribute from child if exist, otherwise return None.
        """
        if not self._ifexist:
            return None
        return self.prod.get_attr(name)


    @execute_when("sema")
    def set_attr(self, name, value):
        """Set Attribute for child if exist, otherwise will do nothing.
        """
        if not self._ifexist:
            return
        return self.prod.set_attr(name, value)


    @execute_when("sema")
    def exist(self):
        return self._ifexist


    @execute_when("sema")
    def visit(self):
        """Visit if exist
        """
        if not self._ifexist:
            return
        self.prod.visit()


def zero_or_one(prod: Production):
    return ZeroOrOne(prod)


class Concat(Production):
    """
        concat(A, B) -->  A B
    """
    def __init__(self, *args):
        super().__init__()
        self.rules : List[Production] = args


    @execute_when("sema")
    def getText(self):
        texts = [rule.getText() for rule in self.rules]
        return " ".join(texts)


    @execute_when("sema")
    def __getitem__(self, index) -> Production:
        """Return the index-th child

        Args:
            index (_int_): index of child
        """
        if not self._ifexist:
            raise IndexError("Production Concat matches nothing.")
        return self.rules[index]


    @execute_when("sema")
    def __iter__(self):
        if not self._ifexist:
            return iter([])
        return iter(self.rules)


    @execute_when("sema")
    def set_attr(self, name, value):
        raise ProductionSemaError("Concat instance should have no attributes.")
    
    
    @execute_when("sema")
    def get_attr(self, name):
        """Concat have no attribute, return None
        """
        return None


    @execute_when("sema")
    def visit(self):
        """Visit children of Concat
        """
        if not self._ifexist:
            return
        for r in self.rules:
            # skip terminal
            if isinstance(r, TerminalRule):
                continue
            r.visit()


def concat(*args):
    # replact regular expression keywords
    if ExecutionEngine.WHEN == "lex":
        args = list(args)
        for idx, arg in enumerate(args):
            if isinstance(arg, str):
                for keyword in TerminalRule.re_keywords:
                    arg = arg.replace(keyword, "\\" + keyword)
                args[idx] = arg
    return Concat(*args)

class Alternate(Production):
    """
        alternate(A, B) --> A | B
    """
    def __init__(self, *args):
        super().__init__()
        self.template_alt_funcs : Tuple[FunctionType] = args
        # ensure that alt functions have zero parameters
        for alt_func in self.template_alt_funcs:
            sig = inspect.signature(alt_func).parameters
            assert len(sig) == 0 and f"alt functions {alt_func.__name__} should not have any parameter"
 
        # During lex and parse time, template_alt_func will generate folded rule tree to avoid recursion
        self.template_alts : List[Production] = []
        if ExecutionEngine.WHEN in ("lex", "parse"):
            self.template_alts = [func() for func in self.template_alt_funcs]
        # the actual matched sub prod
        self.idx : int = None
        self.alt_locals: Dict[str, Any] = {}
        # set in ParserTreeBuilder.visit_Alternate
        # the real matched prod
        self.prod : Production = None
        
        # if visited in sema time
        self.visited = False
    
    @execute_when("get_ast", "sema")
    def visit(self):
        """Execute function of matched alt branch
        """
        if not self._ifexist:
            return
                
        # _getframe(3): visit <-- ExecutableWarpper.__call__ <-- execute_when.decorator
        caller_frame : FrameType = sys._getframe(3)
        if caller_frame.f_code.co_name == "get_attr":
            # _getframe(4): visit <-- ExecutableWarpper.__call__ <-- execute_when.decorator <-- get_attr <-- ExecutableWarpper.__call__ <-- execute_when.decorator 
            caller_frame = sys._getframe(6)
            
        local_dict = caller_frame.f_locals
        alt_func_name = self.template_alt_funcs[self.idx].__name__
        sema_alt_func : FunctionType = local_dict[alt_func_name]
        sema_alt_func.__globals__.update(self.alt_locals)
        if ExecutionEngine.WHEN == "gen_ast":
            return sema_alt_func()
        else:
            self.visited = True
            sema_alt_func()
            return None
    
    @execute_when("sema")
    def get_actual_alt(self):
        """Get actual matched rule.
        """
        if not self._ifexist:
            return None
        return self.prod


    @execute_when("sema")
    def get_actual_alt_index(self):
        """Get actual matched alt function index.
        """
        if not self._ifexist:
            return None
        return self.idx


    @execute_when("sema")
    def getText(self):
        return self.prod.getText()
    
    
    @execute_when("sema")
    def set_attr(self, name, value):
        """Set attribute to matched branch
        """
        if not self._ifexist:
            return
        return self.prod.set_attr(name, value)
    
    
    @execute_when("sema")
    def get_attr(self, name):
        """Get attribuet from matched branch
        """
        if not self._ifexist:
            return None
        if not self.visited:
            self.visit()
        return self.prod.get_attr(name)
    

def alternate(*args):
    return Alternate(*args)



class Rule(Production):
    def __init__(self, production = None, name = "UNKNOWN"):
        super().__init__()
        self.production = None
        self.name = name
        self.setProduction(production)
        
    def setProduction(self, prod):
        self.production = prod


class ParserRule(Rule):
    def __init__(self, production = None, name = "UNKNOWN"):
        super().__init__(production, name)
        self.attributes : Dict[str, Any] = {}
        self.visited = False
        self.visit_func : FunctionType = None
        
        
    @execute_when("parse", "gen_ast")
    def setProduction(self, prod):
        super().setProduction(prod)

    
    @execute_when("sema")
    def set_attr(self, name: str, value):
        self.attributes.update({name: value})
    
    
    @execute_when("sema")
    def get_attr(self, name: str):
        if not self._ifexist:
            return None
        if not self.visited:
            self.visit()
        return self.attributes.get(name, None)
    
    @execute_when("sema")
    def visit(self):
        """
            visit tree node 
        """
        if not self._ifexist:
            return
        self.visited = True
        self.visit_func()

    @execute_when("sema")
    def getText(self):
        return self.production.getText()
            
class TerminalRule(Rule):
    re_keywords = (
        ".",
        "*",
        "+",
        "?",
        "{",
        "}",
        "^",
        "$",
        "(",
        ")",
        "[",
        "]"
    )


    def __init__(self, production = None, name = "UNKNOWN"):
        super().__init__(production, name)

    
    @execute_when("lex", "parse", "gen_ast")
    def setProduction(self, prod):
        
        if isinstance(prod, str):
            for keyword in TerminalRule.re_keywords:
                prod = prod.replace(keyword, "\\" + keyword)
        return super().setProduction(prod)
            

    @execute_when("sema")
    def getText(self):
        if self.content is None:
            return ""
        assert isinstance(self.content, str)
        return self.content



@execute_when("parse", "gen_ast")
def newParserRule(prod = None) -> ParserRule:
    """Create ParserRule, can only used in parser function.

    Args:
        prod (_TerminalRule_, _ParserRule_, _Production_): Set the production of parser rule, Defaults to None, production can be set by calling TerminalRule.setProduction later.
        `prod` can be return of parser or lexer method directly or EBNF create by function:
        * concat
        * alternate
        * zero_or_one
        * zero_or_more
        * one_or_more
        see document of method for more details.
        
    Returns:
        ParserRule: _description_
    """
    g = ParserRule(prod)
    g.name = sys._getframe(3).f_code.co_name
    return g


@execute_when("lex", "parse", "gen_ast")
def newTerminalRule(prod = None) -> TerminalRule:
    """Create TerminalRule, can only used in lexer function.

    Args:
        prod (_str_, _Production_): Set the production of terminal rule. Defaults to None, production can be set by calling TerminalRule.setProduction later.
        If prod is instance of `str`, it matched string that equal to prod.
        If prod is instance of `Production`, it will generate a regular expression, and matches string that match the regular pattern. Lexer production can be create by function:
        * concat
        * alternate
        * zero_or_one
        * zero_or_more
        * one_or_more
        * regular_expr: create regular expression
        
    Returns:
        TerminalRule: return of lexer rule.
    """
    g = TerminalRule(prod)
    g.name = sys._getframe(3).f_code.co_name
    return g
    
    



