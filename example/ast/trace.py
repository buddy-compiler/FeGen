import sys
from types import FunctionType, FrameType

def A():
    a = 10
    return "A"

def B():
    b = 10
    return "B"

def C():
    c = 30
    return "C"

def A_B():
    ab = 40
    return A() + B()

def capture_locals(func: FunctionType, *args, **kwargs):
    """execute and capture locals of func"""
    locals_dict = {}
    original_trace = sys.gettrace()
    frame_list = []
    def trace_function(frame: FrameType, event, arg):
        if event == "return":
            # locals_dict.update(frame.f_locals)
            print(f"Event: {event}, Function: {frame.f_code.co_name}, Line: {frame.f_lineno}, args: {arg}")
        frame_list.append(frame)
        return trace_function
    
    sys.settrace(trace_function)
    try:
        res = func(*args, **kwargs) # execute
    finally:
        sys.settrace(original_trace)  # resume trace function
    
    return frame_list

f = capture_locals(A_B)
for i in f:    
    print(i)