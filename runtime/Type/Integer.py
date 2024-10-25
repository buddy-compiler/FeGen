from .Type import Type
from .Value import Value
from typing import Dict, Callable, Tuple
from typing import Type as CLS
from mlir import ir
from mlir.dialects import arith


def addInteger(lhs: Value, rhs: Value):
    if not lhs.isVariable:
        lhs = Value.convertToVariable(lhs)
    if not rhs.isVariable:
        rhs = Value.convertToVariable(rhs)

    res = arith.AddIOp(lhs.content, rhs.content)
    return Value(res, Integer(), True)


class Integer(Type):
    pass


class Integer(Type):
    ADD_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    def __init__(self, size: int = 32) -> None:
        self.size = size
        super().__init__()

    @property
    def mlirType(self):
        return ir.IntegerType.get_signless(self.size)

    def createMLIRConstant(self, constant):
        return arith.ConstantOp(self.mlirType, constant)


Integer.ADD_MAP = {
    Integer: (int.__add__, addInteger)
}
