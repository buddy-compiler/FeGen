from ..Type import Value, Type
from typing import List
from mlir.dialects import scf
from mlir.ir import InsertionPoint, Operation, Block


class IfElseStmt:
    def __init__(self, cond: Value, initialValues: List[Value]) -> None:
        assert cond.isVariable
        self.cond = cond
        self.resultTypes: List[Type] = [v.valuetype for v in initialValues]
        self.initialValues = initialValues
        self.op = scf.IfOp(cond=cond.mlirValue, results_=[
                           v.mlirType for v in initialValues], hasElse=True)

    @property
    def thenBody(self) -> InsertionPoint:
        return InsertionPoint(self.op.then_block)

    @property
    def elseBody(self) -> InsertionPoint:
        return InsertionPoint(self.op.else_block)

    def yieldVar(self, vars: List[Value]):
        scf.YieldOp([v.mlirValue for v in vars])

    @property
    def results(self) -> List[Value]:
        op: Operation = self.op.operation
        return [Value(v, ty, True) for v, ty in zip(op.results, self.resultTypes)]
