import sys

def capture_locals(func):
    """捕获函数返回前的所有局部变量"""
    locals_dict = {}
    original_trace = sys.gettrace()
    
    def trace_function(frame, event, arg):
        if event == 'return' and frame.f_code == func.__code__:
            locals_dict.update(frame.f_locals)
        return trace_function
    
    sys.settrace(trace_function)
    try:
        func()  # 执行目标函数
    finally:
        sys.settrace(original_trace)  # 恢复原始跟踪函数
    
    return locals_dict

# 示例使用

class test:
    pass

def example_function():
    t = test()
    print("id(t): ", hex(id(t)))
    a = 10
    b = 20
    c = a + b
    return c

locals_before_return = capture_locals(example_function)
print(locals_before_return)  # 输出: {'a': 10, 'b': 20, 'c': 30}