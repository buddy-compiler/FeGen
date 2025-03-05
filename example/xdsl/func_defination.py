import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin
import xdsl.dialects.arith as arith
from xdsl.printer import Printer


printer = Printer()
functy = func.FunctionType.from_lists([builtin.IntegerType(32), builtin.IntegerType(32)], [builtin.IntegerType(32)])
myfunc = func.FuncOp(name="test", function_type=functy, visibility="private")
lhs = myfunc.args[0]
rhs = myfunc.args[1]
addi = arith.AddiOp(lhs, rhs, builtin.i32)
ret = func.ReturnOp(addi)
myfunc.body.block.add_ops([addi, ret])
printer.print(myfunc)