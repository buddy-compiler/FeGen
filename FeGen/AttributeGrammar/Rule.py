from typing import Type, List, Dict

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
        self.setProduction(production)
        
    def setProduction(self, prod):
        self.production = prod

class ParserRule(Rule):
    def __init__(self, production = None):
        super().__init__(production)
        self.attributes : Dict[str, Attribute] = []
        
    def new_attr(self, name: str, ty: Type, init = None):
        assert name not in self.attributes and f"Attribute {name} already exists."
        attr = Attribute(name, ty, init)
        self.attributes[name] = attr
        return attr
        
    def set_attr(self, name: str, value):
        assert name not in self.attributes and f"Attribute {name} does exist."
        attr = self.attributes[name]
        attr.set(value)

    def get_attr(self, name: str):
        assert name in self.attributes and f"Attribute {name} does not exist."
        return self.attributes[name]

class TerminalRule(Rule):
    def __init__(self, production = None):
        super().__init__(production)
        
    def text(self) -> str:
        return "TODO"
        