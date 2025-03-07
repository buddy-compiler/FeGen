def deco1(cls):
    print(cls.__name__)
    return cls

def deco2(func):
    print(func.__name__)
    return func

@deco1
class clazz:
    @deco2
    def func(self):
        return
    
x = clazz()