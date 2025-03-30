from typing import Type, List, Dict, Literal, Callable, Tuple, Any, Set
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
import traceback
from datetime import datetime


class LexOrParseError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class SemaError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class FeGenLexer:
    def __init__(self, regular_expression: Dict[str, str], ignore_re: str):
        # self.module = module
        # outputdir = self.module.__file__
        # self.lexer = lex.lex(module=self.module, debug=False, outputdir=outputdir)
        self.regular_expression = regular_expression
        self.tokens = list(regular_expression.keys())
        self.lexer = self.__build(self.regular_expression, ignore_re)

    def input(self, src: str):
        self.lexer.input(src)
        token_list = []
        while True:
            token = self.lexer.token()
            if not token:
                break
            token_list.append(token)
        return token_list

    def __build(self, regular_exprs: Dict[str, str], ignore_re: str):
        """generate ply lexer

        Args:
            regular_exprs (Dict[str, str]): Regular expr dict, for example:
            ```
            regular_exprs = {
                "NUM": "[0-9]+",
                "VAR": "var",
                "ID": "[a-zA-Z_][a-zA-Z0-9]*",
            } 
            ```

        Returns:
            _ply.lex.Lex_: ply lex instance
        """
        from ply.lex import Lexer, PlyLogger, _form_master_re
        import re
        lextab = 'lextab'
        stateinfo  = {'INITIAL': 'inclusive'}
        lexobj = Lexer()
        lexobj.optimize = False
        errorlog = PlyLogger(sys.stderr)
        tokens = tuple(regular_exprs.keys())
        
        lexobj.lextokens = set(tokens)
        lexobj.lexliterals = ""
        lexobj.lextokens_all = lexobj.lextokens | set(lexobj.lexliterals)
        
        regex_template = "(?P<t_{tokenname}>{prod})"
        regexs = {
            "INITIAL": [
                regex_template.format(tokenname=name, prod=regular_exprs[name]) for name in tokens
            ]
        }
        reflags = re.VERBOSE
        ldict = {"t_{}".format(key): value for key, value in regular_exprs.items()}
        
        tokennames = {"t_{}".format(key): key for key in regular_exprs.keys()}
        tokennames.update({"t_ignore": "ignore", "t_error": "error"})
        
        for state in regexs:
            lexre, re_text, re_names = _form_master_re(regexs[state], reflags, ldict, tokennames)
            lexobj.lexstatere[state] = lexre
            lexobj.lexstateretext[state] = re_text
            lexobj.lexstaterenames[state] = re_names

        for state, stype in stateinfo.items():
            if state != 'INITIAL' and stype == 'inclusive':
                lexobj.lexstatere[state].extend(lexobj.lexstatere['INITIAL'])
                lexobj.lexstateretext[state].extend(lexobj.lexstateretext['INITIAL'])
                lexobj.lexstaterenames[state].extend(lexobj.lexstaterenames['INITIAL'])
                
        lexobj.lexstateinfo = stateinfo
        lexobj.lexre = lexobj.lexstatere['INITIAL']
        lexobj.lexretext = lexobj.lexstateretext['INITIAL']
        lexobj.lexreflags = reflags

        # Set up ignore variables
        ignore = {'INITIAL': ignore_re}
        lexobj.lexstateignore = ignore
        lexobj.lexignore = lexobj.lexstateignore.get('INITIAL', '')

        # Set up error functions
        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)
        
        errorf = {'INITIAL': t_error}
        lexobj.lexstateerrorf = errorf
        lexobj.lexerrorf = errorf.get('INITIAL', None)
        if not lexobj.lexerrorf:
            errorlog.warning('No t_error rule is defined')

        # Set up eof functions
        eoff = {}
        lexobj.lexstateeoff = eoff
        lexobj.lexeoff = eoff.get('INITIAL', None)
        # Check state information for ignore and error rules
        for s, stype in stateinfo.items():
            if stype == 'exclusive':
                if s not in errorf:
                    errorlog.warning("No error rule is defined for exclusive state '%s'", s)
                if s not in ignore and lexobj.lexignore:
                    errorlog.warning("No ignore rule is defined for exclusive state '%s'", s)
            elif stype == 'inclusive':
                if s not in errorf:
                    errorf[s] = errorf.get('INITIAL', None)
                if s not in ignore:
                    ignore[s] = ignore.get('INITIAL', '')
        return lexobj



class ConcreteSyntaxTree:
    def __init__(self, grammar: "FeGenGrammar", root: "ParserRule", parser_locals_dict: Dict["ParserRule", Dict[str, Any]], lexer_locals_dict: Dict["TerminalRule", Dict[str, Any]] ):
        self.grammar = grammar
        self.root = root
        self.parser_locals_dict = parser_locals_dict
        self.lexer_locals_dict = lexer_locals_dict
        self._eval()


    def __generate_sema_func(self):
        # compile and execute function code to get function object
        sema_func_dict : Dict[str, FunctionType] = {}
        sema_target_file = self.grammar.sema_target_file
        sema_func_names = list(ExecutionEngine.parserRuleFunc.keys())
        with open(sema_target_file, "r") as f:
            target_code_str = f.read()
            target_code = compile(target_code_str, sema_target_file, mode = "exec")
            local_env = {}
            exec(target_code, self.grammar.__init__.__globals__.copy(), local_env)
            for local_name, local_obj in local_env.items():
                if local_name in sema_func_names and isinstance(local_obj, FunctionType):
                    sema_func_dict.update({local_name: local_obj})
        return sema_func_dict
            
        

    def _eval(self):
        ExecutionEngine.WHEN = "sema"
        sema_func_dict = {}
        for rule, local_dict in self.parser_locals_dict.items():
            rule_name = rule.name
            sema_func_dict = self.__generate_sema_func()
            sema_func = sema_func_dict[rule_name]
            sema_func.__globals__.update(local_dict)
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
    def __init__(self, p_funcs: Dict[str, FunctionType], prods: List[Tuple[str, List[str], str]], rules: Set[str], lexer: FeGenLexer, grammar: "FeGenGrammar", start: str):
        self.p_funcs = p_funcs
        self.prods = prods
        self.rules = rules
        self.lexer = lexer
        self.grammar = grammar
        self.start = start
        self.start_func : MethodType = getattr(self.grammar, self.start)
        if self.start_func is None:
            raise LexOrParseError("\n\n\nCan not find parse function for start rule: `{start}`, maybe you forgot to decorate `{start}` by `@parser`".format(start=start))

        # create yacc parser
        self.__parser = self.__build(self.lexer.tokens, self.p_funcs, self.prods, self.start)

        
        
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
        raw_data = self.__parser.parse(code, lexer=self.lexer.lexer)
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


    def __build(self, terminals: List[str], p_funcs: Dict[str, FunctionType], prods: List[Tuple[str, List[str], str]], start: str):
        """_generate ply parser_

        Args:
            terminals (List[str]): lexer tokens
            p_funcs (Dict[str, FunctionType]): rule functions
            prods (Dict[str, List[str]]): rule productions
            start (str): start rule name
            ```
            terminals = ["NUM", "VAR", "ID"]
            p_funcs = {
                "p_num": p_num,
                "p_var": p_var,
                "p_id": p_id,
                "p_module": p_module
            }
            prods = [
                ("num", ["NUM"], "p_num"),
                ("var": ["VAR"], "p_var"),
                ("id": ["ID"], "p_id"),
                ("module": ["num", "var", "id"], "p_module")
            ]
            ```
        Returns:
            _LRParser_: ply parser
        """
        from ply.yacc import Grammar, LRGeneratedTable, LRParser

        g = Grammar(terminals)

        for idx, t in enumerate(terminals):
            g.set_precedence(t, "left", idx)

        for name, prod, pfunc_name in prods:
            g.add_production(name, prod, pfunc_name)

        g.set_start(start)

        # report error for undefined symbols
        if len(g.undefined_symbols()) != 0:
            raise Exception()

        # report unused terminals
        unused_terminals = g.unused_terminals()
        if unused_terminals:
            logging.warning(f"unused terminals: {unused_terminals}")

        # Find unused non-terminals
        unused_rules = g.unused_rules()
        if unused_rules:
            logging.warning(f"unused rules: {unused_rules}")

        # find unreachable
        unreachable = g.find_unreachable()
        for u in unreachable:
            logging.warning('Symbol {} is unreachable'.format(u))

        # find infinite cycles
        infinite = g.infinite_cycles()
        for inf in infinite:
            logging.warning('Infinite recursion detected for symbol {}'.format(inf))

        # find unused precedence
        unused_prec = g.unused_precedence()
        for term, assoc in unused_prec:
            logging.warning('Precedence rule {assoc} defined for unknown symbol {term}}'.format(assoc=assoc, term=term))
            
        lr = LRGeneratedTable(g)

        # Build the parser
        lr.bind_callables(p_funcs)
        parser = LRParser(lr, None)
        return parser
    
class FeGenGrammar:
    """
        base class of all grammar class
    """    
    def __init__(self, output_dir_name = ".fegen"):
        import hashlib
        self.output_dir_name = output_dir_name
        # mkdir for plymodule
        # calc hash code of src file
        src_filepath = inspect.getsourcefile(self.__class__)
        if src_filepath is None:
            raise LexOrParseError("Can not find filepath of class: `{}`".format(self.__class__))
        chunksize = 8192
        sha1_func = hashlib.sha1()
        with open(src_filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunksize), b""):
                sha1_func.update(chunk)
        hash_code = sha1_func.hexdigest()
        
        # set attr for plymodule
        src_dir = os.path.dirname(src_filepath)
        outputdir = os.path.join(src_dir, self.output_dir_name, hash_code)
        self.do_generate = False
        if not os.path.exists(outputdir):
            self.do_generate = True
            os.makedirs(outputdir, exist_ok=True)
        # initialize lexer and parser
        self.lexerobj = None
        self.parserobj = None

        # create empty lex, parser and sema function file
        self.lexer_target_file = os.path.join(outputdir, "lexer_functions.py")
        self.parser_target_file = os.path.join(outputdir, "parser_functions.py")
        self.sema_target_file = os.path.join(outputdir, "sema_functions.py")
        
        
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
            global_sema_dict = ExecutionEngine.semaCodeForLexRule
            target_file = self.lexer_target_file
        elif when == "parse":
            func_dict = ExecutionEngine.parserRuleFunc
            global_sema_dict = ExecutionEngine.semaFuncForParseRule
            target_file = self.parser_target_file

        
        lex_parser_codestr_dict : Dict[str, str] = {}
        sema_codestr_dict : Dict[str, str] = {}
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
            
            # unparse and remove redundant link break
            cvted_parseorlex_func_code_str = astunparse.unparse(cvted_parseorlex_func_ast)
            cvted_parseorlex_func_code_str = cvted_parseorlex_func_code_str.strip("\n")
            cvted_sema_func_code_str = astunparse.unparse(cvted_sema_func_ast)
            cvted_sema_func_code_str = cvted_sema_func_code_str.strip("\n")
            # collect codes
            lex_parser_codestr_dict.update({name: cvted_parseorlex_func_code_str})
            sema_codestr_dict.update({name: cvted_sema_func_code_str})
        
        
        # handle lexer / parser function
        # dump lex/parse function codes
        if when in ("lex", "parse") and self.do_generate:
            with open(target_file, "a") as f:
                for func_code in lex_parser_codestr_dict.values():
                    f.write(func_code)
                    f.write("\n\n\n")
        
        # compile and execute function code to get function object
        lex_parser_func_dict : Dict[str, FunctionType] = {}
        with open(target_file, "r") as f:
            lex_parser_func_names = list(lex_parser_codestr_dict.keys())
            target_code_str = f.read()
            target_code = compile(target_code_str, target_file, mode = "exec")
            local_env = {}
            exec(target_code, self.__init__.__globals__, local_env)
            for local_name, local_obj in local_env.items():
                if local_name in lex_parser_func_names and isinstance(local_obj, FunctionType):
                    lex_parser_func_dict.update({local_name: local_obj})
            assert len(lex_parser_codestr_dict) == len(lex_parser_func_dict)
            
        # execute to get ruletree
        rule_trees = []
        for name, lex_parser_func in lex_parser_func_dict.items():
            self_copy = copy.copy(self)
            self.__update_rules(self_copy)
            # execute to generate folded ruletree
            rule_tree = lex_parser_func(self_copy)
            rule_trees.append(rule_tree)
            
            # update functions of self
            setattr(self, name, MethodType(lex_parser_func, self))
        
        
        # handle sema function
        if self.do_generate:
            # dump sema function codes
            with open(self.sema_target_file, "a") as f:
                for func_code in sema_codestr_dict.values():
                    f.write(func_code)
                    f.write("\n\n\n")
        
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
        
        
        gen = LexerProdGen(ruletrees_for_lex)
        
        # insert lex defination
        regular_expressions = {}
        for lexRule in ruletrees_for_lex:
            name = lexRule.name
            prod = lexRule.production
            lexprod = gen(prod)
            logging.debug(f"Regular expression for rule '{name}' is {lexprod}")
            regular_expressions.update({name:  lexprod})
        # insert skip
        ignore_re = self.skip()
        # generate PLY lexer
        self.lexerobj = FeGenLexer(regular_expressions, ignore_re)
        return self.lexerobj


    def parser(self, lexer: FeGenLexer, start = None) -> FeGenParser:
        if self.parserobj is not None:
            return self.parserobj
        from .ExecuteEngine import ParserProdGen
        # process source code and generate code for lexer
        ExecutionEngine.WHEN = "parse"
        
        # generated rule trees for parse rule generating
        # generate function code to function file
        ruletrees_for_parse : List[ParserRule] = self.__convert_functions_and_get_ruletree(ExecutionEngine.WHEN)
        
        # generate PLY function
        gen = ParserProdGen()
        for rule in ruletrees_for_parse:
            gen(rule)
        p_funcs = gen.p_funcs
        prods = gen.prods
        rules = gen.rules
        self.parserobj = FeGenParser(p_funcs, prods, rules, lexer, self, start)
        return self.parserobj


    def skip(self):
        """lex rule to skip

        Returns:
            _str_: regular expression to skip.
        """
        return " \t\n" 

        

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
    semaCodeForLexRule : Dict[str, str] = {}
    semaFuncForParseRule : Dict[str, FunctionType] = {}




def parser(parser_rule_defination):
    """A decorator to mark a function in a subclass of FeGenGrammar that defines a parse rule, ExecutionEngine.WHEN decides action of function.

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
    """A decorator to mark a function in a subclass of FeGenGrammar that defines a lex rule, ExecutionEngine.WHEN decides action of function.

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



def execute_when(*when):
    """mark function to execute at correct time
        when can be:
        * lex
        * parse
        * get_ast
        * sema
    """
    legal_when = ("lex", "parse", "gen_ast", "sema")
    for item in when:
        assert isinstance(item, str), "Function `execute_when` accepts only string values."
        assert item in legal_when, "Function `execute_when` accepts only strings from `{}`.".format(", ".join(legal_when))
    def wrapper(func):
        def check_when(*argc, **kwargs):
            now = ExecutionEngine.WHEN
            if now not in when:
                msg = "Function {name} execute in wrong time: expected in `{times}`, but not is {now}.".format(name=func.__name__, times=", ".join(when), now=now)
                raise Exception(msg)
            else:
                return func(*argc, **kwargs)
        return check_when
    return wrapper


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
    
    @execute_when("gen_ast", "sema")
    def visit(self):
        """Execute function of matched alt branch
        """
        if not self._ifexist:
            return
                
        # _getframe(2): visit(this)(0) <-- check_when(1) <-- <caller>(2)
        caller_frame : FrameType = sys._getframe(2)
        if caller_frame.f_code.co_name == "get_attr":
            # _getframe(4): visit(this)(0) <-- check_when(1) <-- get_attr(2) <-- check_when(3) <-- <caller>(4) 
            caller_frame = sys._getframe(4)
            
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
        attr = self.attributes.get(name, None)
        if attr is None:
            raise SemaError("parser rule `{rulename}` has no attribute {attrname}".format(rulename = self.name, attrname = name))
        return attr
    
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
    # 2: newParserRule(this)(0) <-- check_when(1) <-- <caller>(2)
    g.name = sys._getframe(2).f_code.co_name
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
    # 2: newTerminalRule(this)(0) <-- check_when(1) <-- <caller>(2)
    g.name = sys._getframe(2).f_code.co_name
    return g
    
    



