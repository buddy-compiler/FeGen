from types import MethodType, FunctionType
from typing import Callable

CURR_TIME = "run_time"

class func_warp(Callable):
    def __init__(self, func, exe_time):
        self.func = func
        self.exe_time = exe_time
        
    def __call__(self, *args, **kwds):
        curr_time = CURR_TIME
        if curr_time == self.exe_time:
            if hasattr(self.func, '__self__'):
                self.func.__func__(self.func.__self__, *args, **kwds)
            else:
                self.func(*args, **kwds)


def execute_on(whichtime: str):
    def decorator(func):
        # return func_warp(func, whichtime)
        f = lambda *args, **kwds: func_warp(func, whichtime)(*args, **kwds)
        setattr(f, "execute_on", whichtime)
        return f
    return decorator

class clazz:
    # @execute_on("compile_time")
    def __init__(self):
        self.x = 10       
    
    @execute_on("run_time")
    def test(self):
        print(self.x)
    
    @execute_on("compile_time")
    def test1(self):
        print(self.x)
        
def execute():
    y = clazz()
    a = y.test
    b = y.test1
    print(a is b)
    
execute()


# class myclass:
#     def test(self):
#         pass
    
# y = myclass()
# assert y.test.__self__ is y
# assert not hasattr(myclass, "__self__")