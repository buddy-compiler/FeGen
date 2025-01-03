from . import Type
from typing import Optional, List, Dict
from mlir.dialects import arith


class Value:
    def create(content, valuetype: Type = None) -> 'Value':
        from .Integer import Integer
        from .ListType import ListType

        def inferType(content) -> Type:
            # TODO: infer type according to content
            if isinstance(content, int):
                valuetype = Integer()
            elif isinstance(content, list):
                element = content[0]
                elementType = inferType(element)
                elementType._materialize(element)
                valuetype = ListType(elementType)

            return valuetype

        # content is another value or expression consist of other values
        if isinstance(content, Value):
            # TODO: do some type transform
            pass
        else:
            if valuetype is None:
                valuetype = inferType(content)
            return Value(content, valuetype)

    def convertToVariable(notVariable: 'Value') -> 'Value':
        v = Value.create(notVariable.content, notVariable.valuetype)
        v.setVariable()
        return v

    def __init__(self, content, valuetype: Type, isVariable: bool = False) -> None:
        self.isVariable = isVariable
        self.valuetype: Type = valuetype
        self.content = content

    def setVariable(self):
        if self.isVariable:
            return
        self.isVariable = True
        # generate mlir constant
        self.content = self.valuetype.createMLIRConstant(self.content)

    @property
    def mlirValue(self):
        assert self.isVariable
        return self.content

    @property
    def mlirType(self):
        assert self.isVariable
        return self.valuetype.mlirType

    def __add__(self, another):
        if not isinstance(another, Value):
            another = Value.create(another)
        addMap = self.valuetype.ADD_MAP
        funcPair = addMap[type(another.valuetype)]
        addFunc = funcPair[int(self.isVariable or another.isVariable)]
        if addFunc is not None:
            return addFunc(self, another)
        else:
            raise TypeError(
                "TypeError: unsupported operand type(s) for +: '{}' and '{}'".format(str(self), str(another)))

    def __bool__(self):
        if self.isVariable:
            # TODO: turn content to i1 mlir value
            pass
        else:
            return bool(self.content)
