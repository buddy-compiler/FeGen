from ..Type import Type, Value
from typing import List
from mlir import ir
from mlir.dialects import scf, arith, memref


class ForStmt:
    def __init__(self, iteringObj: Value, initialValues: List[Value]) -> None:
        assert iteringObj.isVariable
        for v in initialValues:
            assert v.isVariable
        self.iteringObj = iteringObj
        self.initialValues = initialValues
        self.resultTypes = [v.valuetype for v in self.initialValues]
        indexTy = ir.IndexType.get()
        c0 = arith.ConstantOp(indexTy, 0)
        c1 = arith.ConstantOp(indexTy, 1)
        lb = c0
        ub = memref.DimOp(self.iteringObj.mlirValue, c0)
        self.op = scf.ForOp(lower_bound=lb, upper_bound=ub,
                            step=c1, iter_args=[v.mlirValue for v in self.initialValues])
        self.induction_variable = self.op.induction_variable

    @property
    def body(self) -> ir.InsertionPoint:
        return ir.InsertionPoint(self.op.body)

    @property
    def iterVar(self):
        ld = memref.LoadOp(self.iteringObj.mlirValue,
                           [self.induction_variable])
        retTy = Type.create(ld.result.type)
        return Value(ld, retTy, True)

    @property
    def iterArgs(self):
        vars = []
        for arg in self.op.inner_iter_args:
            argTy = Type.create(arg.type)
            vars.append(Value(arg, argTy, True))
        return vars

    def yieldVar(self, vars: List[Value]):
        scf.YieldOp([v.mlirValue for v in vars])

    @property
    def results(self) -> List[Value]:
        op: ir.Operation = self.op.operation
        return [Value(v, ty, True) for v, ty in zip(op.results, self.resultTypes)]
