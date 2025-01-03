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
    resTy = Type.create(res.result.type)
    return Value(res, resTy, True)


class Integer(Type):
    ADD_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    def __init__(self, size: int = 32) -> None:
        self.size = size
        super().__init__()

    def createMLIRConstant(self, constant):
        return arith.ConstantOp(self.mlirType, constant)

    def _materialize(self, content):
        self.mlirType = ir.IntegerType.get_signless(self.size)


Integer.ADD_MAP = {
    Integer: (int.__add__, addInteger)
}
