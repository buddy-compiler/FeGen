from typing import Dict, Callable, Tuple
from typing import Type as CLS
from mlir.ir import IntegerType


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
        pass

    @property
    def mlirType(self):
        # TODO
        return IntegerType.get_signless(32)

    def createMLIRConstant(self, constant):
        raise NotImplementedError()
