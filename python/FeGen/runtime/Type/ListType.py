from .Type import Type
from .Value import Value
from mlir import ir
from mlir.dialects import memref, arith
import numpy as np


class ListType(Type):
    def __init__(self, elementType: Type) -> None:
        super().__init__()
        self.elementType = elementType

    def createMLIRConstant(self, constant):
        assert isinstance(constant, list)
        mem = memref.AllocaOp(self.mlirType, [], [])
        for idx, c in enumerate(constant):
            index = arith.ConstantOp(ir.IndexType.get(), idx)
            const = arith.ConstantOp(self.elementType.mlirType, c)
            memref.StoreOp(const, mem, [index])
        return mem

    def _materialize(self, content):
        assert isinstance(content, list)
        size = len(content)
        self.elementType._materialize(content[0])
        self.mlirType = ir.MemRefType.get(
            [size], self.elementType.mlirType)
