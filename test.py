from types import FunctionType
import copy

class test:
    def __init__(self):
        self.x = [i for i in range(10)]
        
    def __getitem__(self, index):
        return self.x[index]

t = test()
for i in t:
    print(i)