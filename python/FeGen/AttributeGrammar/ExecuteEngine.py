from types import FunctionType
from typing import Type, Dict, Optional
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
    def visit_TerminalRule(self, rule: TerminalRule):
        return rule.name
    
    def visit_str(self, s: str):
        return s

    def visit_ChatSet(self, prod: ChatSet):
        return prod.charset

    def visit_Concat(self, prod: Concat):
        prod_exprs = [self.visit(p) for p in prod.rules]
        sur_prod_exprs = [f"({prod_expr})" for prod_expr in prod_exprs]
        return R"{}".format("".join(sur_prod_exprs))
    
    def visit_Alternate(self, prod: Alternate):
        prod_exprs = [self.visit(p) for p in prod.template_alts]
        return R"{}".format("|".join(prod_exprs))
    
    def visit_ZeroOrOne(self, prod: ZeroOrOne):
        prod_expr = self.visit(prod.prod)
        return R"{}?".format(prod_expr)

    def visit_ZeroOrMore(self, prod: ZeroOrMore):
        prod_expr = self.visit(prod.template_prod)
        return R"{}*".format(prod_expr)
        
    def visit_OneOrMore(self, prod: OneOrMore):
        prod_expr = self.visit(prod.template_prod)
        return R"{}+".format(prod_expr)
    
    
    
class ParserProdGen(BaseVisitor):
    def __init__(self, attr_dict):
        super().__init__()
        self.attr_dict : Dict[str, Any] = attr_dict
        
        
    def __call__(self, rule: ParserRule):
        """insert p functions like
        def p_{rulename}(p):
            "rulename: p_name"
            p[0] = p[1]
        """
        assert isinstance(rule, ParserRule)
        self.processing_rule = rule
        rulename = rule.name
        prod = rule.production
        p_name = self.visit(prod)
        def template(p):
            p[0] = (p_name, p[1])
        template.__name__ = f"p_{rulename}"
        template.__doc__ = f"{rulename} : {p_name}"
        self.attr_dict.update({template.__name__: template}) 
    
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
        if rule_name in self.attr_dict:
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
        template.__name__ = f"p_{rule_name}"
        template.__doc__ = "{rule_name} : {prod_children}".format(rule_name=rule_name, prod_children = " ".join(prod_children_names))
        self.attr_dict.update({template.__name__: template})
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
        if rule_name in self.attr_dict:
            return rule_name
        
        def generate_template(rule_name, idx, alt_name):
            def template(p):
                p[0] = (idx, alt_name, p[1])
            template.__name__ = f"p_{rule_name}_{idx}"
            template.__doc__ = f"{rule_name} : {alt_name}"
            return template
        
        # traverse alt names and create functions
        for idx, alt_name in enumerate(alt_names):
            template = generate_template(rule_name, idx, alt_name)
            self.attr_dict.update({template.__name__: template})
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
        if rule_name in self.attr_dict:
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
        template.__name__ = f"p_{rule_name}"
        template.__doc__ = f"""{rule_name} : {rule_name} {child_name}
                                           | {child_name}
                                           |"""
        self.attr_dict.update({template.__name__: template})
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
        if rule_name in self.attr_dict:
            return rule_name
        def template(p):
            p_len = len(p)
            if p_len == 2:
                p[0] = [p[1]]
            elif p_len == 3:
                p[0] = p[1] + [p[2]]
            else:
                assert False
        template.__name__ = f"p_{rule_name}"
        template.__doc__ = f"""{rule_name} : {rule_name} {child_name}
                                           | {child_name}"""
        self.attr_dict.update({template.__name__: template})
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
        if rule_name in self.attr_dict:
            return rule_name
        def template(p):
            p_len = len(p)
            if p_len == 1:
                p[0] = None
            elif p_len == 2:
                p[0] = p[1]
            else:
                assert False
        template.__name__ = f"p_{rule_name}"
        template.__doc__ = f"""{rule_name} : {child_name}
                                           |"""
        self.attr_dict.update({template.__name__: template})
        return rule_name
    
    
class ParserTreeBuilder(BaseVisitor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, start_rule: ParserRule, data: dict):
        self.visit(start_rule, data)
        
    def visit_ParserRule(self, rule: ParserRule, data: Tuple[str, Any]):
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
    
    def visit_Alternate(self, prod: Alternate, data: Tuple[int, str, Any]):
        prod.content = data
        assert len(data) == 3
        # set actual matched child
        idx = data[0]
        prod.prod = prod.template_alts[idx]
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
            # copy children
            for _ in range(len(data) - 1):
                newchild = copy.deepcopy(prod.template_prod)
                prod.children.append(newchild)
                
            assert len(prod.children) == len(data)
            # visit children
            for child, child_data in zip(prod.children, data):
                self.visit(child, child_data)
    
    def visit_OneOrMore(self, prod: OneOrMore, data: List[Any]):
        prod.content = data
        # copy children
        for _ in range(len(data) - 1):
            newchild = copy.deepcopy(prod.template_prod)
            prod.children.append(newchild)
            
        assert len(prod.children) == len(data)
        # visit children
        for child, child_data in zip(prod.children, data):
            self.visit(child, child_data)
    