d = {}

def deco1(cls):
    print(cls.__name__)
    print(d)
    return cls

def deco2(func):
    print(func.__name__)
    d[func.__name__] = func
    return func

@deco1
class clazz:
    @deco2
    def func(self):
        return
    
    @deco2
    def func1(self):
        return 
    
x = clazz()