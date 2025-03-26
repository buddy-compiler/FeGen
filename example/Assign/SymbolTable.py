from typing import Dict

class Variable:
    def __init__(self, name: str, value = None):
        self.name = name
        self.value = value


class Table:
    def __init__(self):
        self.t : Dict[str, Variable] = {}
        
        
    def lookup(self, name):
        """return None if not exist
        """
        return self.t.get(name)
    

    def update(self, v: Variable):
        self.t.update({v.name: v})
        
        