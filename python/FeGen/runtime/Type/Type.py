from typing import Dict, Callable, Tuple
from typing import Type as CLS
from mlir import ir


class Type:
    """ 
        Dict[anotherClass, Tuple(notVariableCase, isVariableCase)]
    """
    # ARITHMETIC
    # +
    ADD_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # -
    SUB_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # *
    MUL_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # /
    DIV_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    # COMPARE
    # ==
    EQ_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # !=
    NE_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # <
    LT_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # <=
    LE_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # >
    GT_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # >=
    GE_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    # SET
    # in
    CONTAINS_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    # INDEX
    # []
    GETITEM_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # []=
    SETITEM_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}
    # BOOL
    # bool
    BOOL_MAP: Dict[CLS, Tuple[Callable, Callable]] = {}

    def __init__(self) -> None:
        self.mlirType = None

    def create(mlirType) -> 'Type':
        """
        Get Type by mlir type.
        """
        from .Integer import Integer
        from .ListType import ListType

        if isinstance(mlirType, ir.IntegerType):
            ty = Integer()
            ty.mlirType = mlirType
            ty.size = mlirType.width
            return ty
        elif isinstance(mlirType, ir.MemRefType):
            elemTy = Type.create(mlirType.element_type)
            ty = ListType(elemTy)
            ty.mlirType = mlirType
            return ty
        else:
            raise ValueError("unknow mlirType: " + str(mlirType))

    def createMLIRConstant(self, constant):
        raise NotImplementedError()

    def _materialize(self, content):
        """
            infer mlir type according to content, content should be python value.
        """
        raise NotImplementedError()
