from types import FunctionType
from typing import Type, Dict, Optional, Set
from .Rule import *

class CodeGenError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class BaseVisitor:
    def __init__(self):
        pass
    
    def __call__(self, prod, *args, **kwargs):
        return self.visit(prod, *args, **kwargs)
        
    def visit(self, prod, *args, **kwargs) -> str:
        visitor_name = "visit_" + prod.__class__.__name__
        visitor = getattr(self, visitor_name)
        if visitor is None:
            raise CodeGenError(f"Can not find visit function: {visitor_name}")
        return visitor(prod, *args, **kwargs)

class LexerProdGen(BaseVisitor):
    """Generate regular expression for terminal rule
    """
    def __init__(self, rule_trees: List[TerminalRule]):
        super().__init__()
        self.rule_trees_dict = {rule.name : rule for rule in rule_trees}
    
    def visit_TerminalRule(self, rule: TerminalRule):
        rule_name = rule.name
        actual_rule = self.rule_trees_dict.get(rule_name)
        assert actual_rule is not None
        return self.visit(actual_rule.production)
    
    def visit_str(self, s: str):
        return s

    def visit_RegularExpression(self, prod: RegularExpression):
        return prod.re_expr

    def visit_Concat(self, prod: Concat):
        prod_exprs = [self.visit(p) for p in prod.rules]
        sur_prod_exprs = [f"({prod_expr})" for prod_expr in prod_exprs]
        return "{}".format("".join(sur_prod_exprs))
    
    def visit_Alternate(self, prod: Alternate):
        prod_exprs = [self.visit(p) for p in prod.template_alts]
        return "{}".format("|".join(prod_exprs))
    
    def visit_ZeroOrOne(self, prod: ZeroOrOne):
        prod_expr = self.visit(prod.prod)
        return "{}?".format(prod_expr)

    def visit_ZeroOrMore(self, prod: ZeroOrMore):
        prod_expr = self.visit(prod.template_prod)
        return "{}*".format(prod_expr)
        
    def visit_OneOrMore(self, prod: OneOrMore):
        prod_expr = self.visit(prod.template_prod)
        return "{}+".format(prod_expr)
    
    
    
class ParserProdGen(BaseVisitor):
    def __init__(self):
        super().__init__()
        self.p_funcs : Dict[str, FunctionType] = {}
        self.prods : List[Tuple[str, List[str], str]] = []
        self.rules : Set[str] = set()
        
    def __call__(self, rule: ParserRule):
        """
        prod: rulename: p_name
        p_funcs: p_rulename: template
        """
        assert isinstance(rule, ParserRule)
        self.processing_rule = rule
        rulename = rule.name
        prod = rule.production
        p_name = self.visit(prod)
        def template(p):
            p[0] = (p_name, p[1])
        
        self.p_funcs.update({f"p_{rulename}": template})
        self.prods.append((rulename, [p_name], f"p_{rulename}"))
    
    def visit_TerminalRule(self, prod: TerminalRule):
        return prod.name
    
    def visit_ParserRule(self, prod: ParserRule):
        return prod.name
    
    def visit_Concat(self, prod: Concat):
        """
        a_and_b: A B
            -->
        a_and_b: __concat_A_B     # in function p_a_and_b
        __concat_A_B: A B         # in function p___concat_A_B
        """
        # visit and get children prod names
        prod_children_names: List[str] = [self.visit(r) for r in prod.rules]
        rule_name = "__concat_" + "_".join(prod_children_names)
        # return if rule name is already existed
        if rule_name in self.rules:
            return rule_name
        # define function and insert to attr dict
        def template(p):
            assert len(p) - 1 == len(prod_children_names)
            d = list()
            for i in range(len(prod_children_names)):
                child_name = prod_children_names[i]
                pi = p[i + 1]
                d.append((child_name, pi))
            p[0] = tuple(d)
        self.p_funcs.update({f"p_{rule_name}": template})
        self.prods.append((rule_name, prod_children_names, f"p_{rule_name}"))
        self.rules.add(rule_name)
        return rule_name
    
    def visit_Alternate(self, prod: Alternate):
        """
        a_or_b: A | B
            -->
        a_or_b: __alt_A_B     # in function p_a_or_b
        __alt_A_B: A          # in function p___alt_A_B_0
        __alt_A_B: B          # in function p___alt_A_B_1
        """
        # visit and get alt names
        alt_names = [self.visit(alt) for alt in prod.template_alts]
        rule_name = "__alt_" + "_".join(alt_names)
        # return rule_name if exist
        if rule_name in self.rules:
            return rule_name

        def gen_template(idx, alt_name):
            def template(p):
                p[0] = (idx, alt_name, p[1])
            return template
        # traverse alt names and create functions
        for idx, alt_name in enumerate(alt_names):
            alt_rule_name = f"p_{rule_name}_{idx}"
            self.prods.append((rule_name, [alt_name], alt_rule_name))
            
            template = gen_template(idx, alt_name)
            self.p_funcs.update({alt_rule_name: template})
        self.rules.add(rule_name)
        return rule_name
    
    def visit_ZeroOrMore(self, prod: ZeroOrMore):
        """
        multi_A: A*
            --> 
        multi_A: __zero_or_more_A             # in function p_multi_A
        __zero_or_more_A : __zero_or_more_A A   # in function p___zero_or_more_A
                       | A
                       |
                       
        """
        child_name = self.visit(prod.template_prod)
        rule_name = f"__zero_or_more_{child_name}"
        # return rule_name if exist
        if rule_name in self.rules:
            return rule_name
        def template(p):
            p_len = len(p)
            if p_len == 1:
                p[0] = []
            elif p_len == 2:
                p[0] = [p[1]]
            elif p_len == 3:
                p[0] = p[1] + [p[2]]
            else:
                assert False
        self.prods.append((rule_name, [rule_name, child_name], f"p_{rule_name}"))
        self.prods.append((rule_name, [child_name], f"p_{rule_name}"))
        self.prods.append((rule_name, [], f"p_{rule_name}"))
        self.p_funcs.update({f"p_{rule_name}": template})
        self.rules.add(rule_name)
        return rule_name
        
    def visit_OneOrMore(self, prod: OneOrMore):
        """
        multi_A: A+
            -->
        multi_A: __one_or_more_A                # in function p_multi_A
        __one_or_more_A : __one_or_more_A A     # in function p___one_or_more_A
                      | A
        """
        child_name = self.visit(prod.template_prod)
        rule_name = f"__one_or_more_{child_name}"
        # return rule_name if exist
        if rule_name in self.rules:
            return rule_name
        def template(p):
            p_len = len(p)
            if p_len == 2:
                p[0] = [p[1]]
            elif p_len == 3:
                p[0] = p[1] + [p[2]]
            else:
                assert False
        self.prods.append((rule_name, [rule_name, child_name], f"p_{rule_name}"))
        self.prods.append((rule_name, [child_name], f"p_{rule_name}"))
        self.p_funcs.update({f"p_{rule_name}": template})
        self.rules.add(rule_name)
        return rule_name
    

    def visit_ZeroOrOne(self, prod: ZeroOrOne):
        """
        opt_a : A?
            --> 
        opt_a : __zero_or_one_A     # in function p_opt_a
        __zero_or_one_A : A         # in function p___zero_or_one_A
                        |
        """
        child_name = self.visit(prod.prod)
        rule_name = f"__zero_or_one_{child_name}"
        # return rule_name if exist
        if rule_name in self.rules:
            return rule_name
        def template(p):
            p_len = len(p)
            if p_len == 1:
                p[0] = None
            elif p_len == 2:
                p[0] = p[1]
            else:
                assert False
        self.prods.append((rule_name, [child_name], f"p_{rule_name}"))
        self.prods.append((rule_name, [], f"p_{rule_name}"))
        self.p_funcs.update({f"p_{rule_name}": template})
        self.rules.add(rule_name)
        return rule_name
    
    
class ParserTreeBuilder(BaseVisitor):
    """Execute when: get_ast
    """
    def __init__(self, grammar: FeGenGrammar):
        super().__init__()
        self.parser_locals_dict: Dict[ParserRule, Dict[str, Any]] = {}
        self.lexer_locals_dict: Dict[TerminalRule, Dict[str, Any]] = {}
        self.grammar = grammar
        
        
    def __call__(self, start_rule: ParserRule, data: dict):
        self.visit(start_rule, data)
    
    def __set_not_exist(self, rule: Production):
        rule._ifexist = False
        if isinstance(rule, ParserRule):
            self.__set_not_exist(rule.production)
        elif isinstance(rule, Concat):
            for r in rule.rules:
                self.__set_not_exist(r)
        elif isinstance(rule, ZeroOrOne):
            self.__set_not_exist(rule.prod)
        elif isinstance(rule, ZeroOrMore):
            for child in rule.children:
                self.__set_not_exist(child)
        elif isinstance(rule, OneOrMore):
            for child in rule.children:
                self.__set_not_exist(child)
        else:
            return
        
    def visit(self, rule: Production, data):
        # if no data matches, set rule not exist
        if data is None:
            self.__set_not_exist(rule)
            rule.content = None
        else:
            return super().visit(rule, data)
        
    def visit_ParserRule(self, rule: ParserRule, data: Tuple[str, Any]):
        """
        data = (p_name, p[1])
        """
        rule.content = data
        assert len(data) == 2
        self.visit(rule.production, data[1])
        
    def visit_TerminalRule(self, rule: TerminalRule, data: str):
        rule.content = data

    def visit_Concat(self, prod: Concat, data: Tuple[Tuple[str, Any]]):
        prod.content = data
        assert len(prod.rules) == len(data)
        for p, d in zip(prod.rules, data):
            assert len(d) == 2
            self.visit(p, d[1])
    
    def __capture_alt_locals(self, func: FunctionType):
        """execute to capture local variables of the actual alt of Alternate Object"""
        parser_locals_dict: Dict[ParserRule, Dict[str, Any]] = {}
        lexer_locals_dict: Dict[TerminalRule, Dict[str, Any]] = {}
        locals_dict : Dict[str, Any] = {}
        actual_func_name = func.__name__
        
        parser_rule_names = list(ExecutionEngine.parserRuleFunc.keys())
        lexer_rule_names = list(ExecutionEngine.lexerRuleFunc.keys())
        
        def trace_function(frame: FrameType, event: str, arg):
            if event == 'return':
                func_name = frame.f_code.co_name
                if func_name in parser_rule_names and isinstance(arg, ParserRule):
                    parser_locals_dict.update({arg: frame.f_locals})
                elif func_name in lexer_rule_names and isinstance(arg, TerminalRule):
                    lexer_locals_dict.update({arg: frame.f_locals})
                elif func_name == actual_func_name:
                    locals_dict.update(frame.f_locals)
            return trace_function
        
        
        original_trace = sys.gettrace()
        sys.settrace(trace_function)
        try:
            res = func() # execute
        finally:
            sys.settrace(original_trace)  # resume trace function
        
        return (res, parser_locals_dict, lexer_locals_dict, locals_dict)
        
    def __capture_locals(self, func: FunctionType):
        """execute to capture local variables"""
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
            res = func() # execute
        finally:
            sys.settrace(original_trace)  # resume trace function
        
        return (res, parser_locals_dict, lexer_locals_dict)
    
    
    def __copyParserTree(self, src: Production):
        if isinstance(src, Concat):
            src_children = src.rules
            res_children = []
            # copy children
            for child in src_children:
                res_child = self.__copyParserTree(child)
                res_children.append(res_child)
            return concat(*res_children)
        elif isinstance(src, Alternate):
            return alternate(*src.template_alt_funcs)
        elif isinstance(src, ZeroOrOne):
            return zero_or_one(self.__copyParserTree(src.prod))
        elif isinstance(src, ZeroOrMore):
            return zero_or_more(self.__copyParserTree(src.template_prod))
        elif isinstance(src, OneOrMore):
            return one_or_more(self.__copyParserTree(src.template_prod))
        elif isinstance(src, ParserRule):
            name = src.name
            parser_func = getattr(self.grammar, name)
            res, parser_locals_dict, lexer_locals_dict = self.__capture_locals(parser_func)
            self.parser_locals_dict.update(parser_locals_dict)
            self.lexer_locals_dict.update(lexer_locals_dict)
            return res
        elif isinstance(src, TerminalRule):
            name = src.name
            lex_func = getattr(self.grammar, name)
            res, parser_locals_dict, lexer_locals_dict = self.__capture_locals(lex_func)
            self.parser_locals_dict.update(parser_locals_dict)
            self.lexer_locals_dict.update(lexer_locals_dict)
            return res
        else:
            return copy.deepcopy(src)
    
    
    def visit_Alternate(self, prod: Alternate, data: Tuple[int, str, Any]):
        prod.content = data
        assert len(data) == 3
        # set actual matched child
        idx = data[0]
        prod.idx = idx
        actual_func = prod.template_alt_funcs[idx]
        # execute and collect local variables
        res, parser_locals_dict, lexer_locals_dict, locals_dict = self.__capture_alt_locals(actual_func)
        
        prod.prod = res
        prod.alt_locals = locals_dict
        # store local variables
        self.parser_locals_dict.update(parser_locals_dict)
        self.lexer_locals_dict.update(lexer_locals_dict)
        # visit child
        self.visit(prod.prod, data[2])
    
    
    def visit_ZeroOrOne(self, prod: ZeroOrOne, data: Optional[Any]):
        prod.content = data
        if data is not None:
            self.visit(prod.prod, data)
            
            
    def visit_ZeroOrMore(self, prod: ZeroOrMore, data: List[Any]):
        prod.content = data
        if len(data) == 0: # if prod match no content
            prod.template_prod.content = None
        else:
            prod.children.append(prod.template_prod)
            # copy children
            for _ in range(len(data) - 1):
                newchild = self.__copyParserTree(prod.template_prod)
                prod.children.append(newchild)
                
            assert len(prod.children) == len(data)
            # visit children
            for child, child_data in zip(prod.children, data):
                self.visit(child, child_data)
    
    def visit_OneOrMore(self, prod: OneOrMore, data: List[Any]):
        prod.content = data
        # copy children
        for _ in range(len(data) - 1):
            newchild = self.__copyParserTree(prod.template_prod)
            prod.children.append(newchild)
            
        assert len(prod.children) == len(data)
        # visit children
        for child, child_data in zip(prod.children, data):
            self.visit(child, child_data)


class ParserTreeVisitor(BaseVisitor):
    def __init__(self):
        super().__init__()