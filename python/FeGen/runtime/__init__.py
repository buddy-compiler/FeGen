import sys
import os
from .Statements import IfElseStmt, ForStmt
from .Type import Value, Integer

# load fegen_mlir
import importlib.util
file_names = [f for f in os.listdir(os.path.dirname(__file__)) if f.startswith("fegen_mlir") and f.endswith(".so")]
assert(len(file_names) == 1)
spec = importlib.util.spec_from_file_location("FeGen.runtime.fegen_mlir", os.path.dirname(__file__) + "/" + file_names[0])
fegen_mlir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fegen_mlir)