from xdsl.dialects.builtin import IntegerType, SSAValue, Region, Block, IntegerAttr
from xdsl.context import Context
from xdsl.printer import Printer
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.memref as memref
import xdsl.dialects.func as func

from FeGen import *
from FeGen.AttributeGrammar import *



table : dict = {}

"""
    example: a = 1 + 2
-->
    %1 = arith.constantOp(1, i32)
    %2 = arith.constantOp(2, i32)
    %add = arith.addi(%1, %2)
    %a = memref.alloca(i32)
    memref.store(%a, %add)

    module
        [syn module: Value]
        : assign_stmt+ {
            ops = []
            for stmt in $assign_stmt:
                ops.append(stmt.stmt)
            module = FuncOp.from_region("my_function", [], [], Region([Block(ops)]))
        }
        ;

    assign_stmt
        [syn stmt: Value]
        : variable_access Assign expression {
            alloca = $variable_access.alloca
            name = $variable_access.name
            v = $expression.v
            if alloca is None:
                alloca = memref.AllocaOp.create([v], MemrefType(v.type, [1]))
                table.insert(name, alloca)
            stmt = memref.store(alloca, v)
        }
        ;
    
    variable_access
        [syn alloca: Value, name: str]
        : name_access {
            alloca = $name_access.alloca
            name = $name_access.name
        }
        ;
    
    name_access
        [syn alloca: Value, name: str]
        : Identifier {
            name = $Identifier.text
            alloca = table.find(name)
        }
        ;
        
    expression
        [syn v: Value]
        : prim_expr {v = $prim_expr.v}
        | add_expr {v = $add_expr.v}
        ;    
        
    prim_expr
        [syn v: Value]
        : variable_access {v = memref.load($variable_access.alloca)}
        | Number {v = arith.constantOp(int($Number.text), IntegerType.get(32))}
        ;
        
    add_expr
        [syn v: Value]
        : prim_expr Add add_expr {v = arith.addi($prim_expr.v, $add_expr.v)}
        ;
    
    DEF: 'def';
    
    RETURN: 'return';
    
    LB: '{';
    
    RB: '}';
    
    LP: '(';
    
    RP: ')';
    
    Add: '+';
    
    Assign: '=';
    
    Dot: '.';
    
    Number: '0-9' | '1-9' '0-9'*;
    
    Identifier: [a-zA-Z][a-zA-Z0-9]*;
"""


@attr_grammar
class MyGrammar(FeGenGrammar):
    @parser
    def module(self):
        """
        module
            [syn module: Value]
            : assign_stmt+ {
                ops = []
                for stmt in $assign_stmt:
                    ops.append(stmt.stmt)
                module = FuncOp.from_region("my_function", [], [], Region([Block(ops)]))
            }
            ;
        """
        g = newParserRule()
        stmts = one_or_more(self.assign_stmt())
        g.production = stmts
        ops = []
        for s in stmts:
            ops.append(s.get_attr("stmt"))
        g.new_attr("module", SSAValue).set_attr(func.FuncOp.from_region("my_function", [], [], Region([Block(ops)])))
        return g
    
    @parser
    def assign_stmt(self):
        """
        assign_stmt
            [syn stmt: Value]
            : variable_access Assign expression {
                alloca = $variable_access.alloca
                name = $variable_access.name
                v = $expression.v
                if alloca is None:
                    alloca = memref.AllocaOp.create([v], MemrefType(v.type, [1]))
                    table.insert(name, alloca)
                stmt = memref.store(alloca, v)
            }
            ;
        """
        
        # grammar define
        var_access = self.variable_access()
        expr = self.expression()
        g = newParserRule()
        g.production = concat(var_access, self.Assign(), expr)
        
        # attribute define
        v: SSAValue = expr.get_attr("v")
        alloca = var_access.get_attr("alloca")
        name = var_access.get_attr("name")
        if alloca is None:
            ty = memref.MemRefType(v.type, [1])
            alloca = memref.AllocaOp.create(operands = [1], result_types=ty)
            table.insert(name, alloca)
        stmt = g.new_attr("stmt", SSAValue)
        stmt.set(memref.StoreOp.create(operands = [alloca.result[0], v.result[0]]))
        return g
    
    @parser
    def variable_access(self):
        """
        variable_access
            [syn alloca: Value, name: str]
            : name_access {
                alloca = $name_access.alloca
                name = $name_access.name
            }
            ;
        """
        name_acc = self.name_access()
        g = newParserRule(name_acc)
        alloca = name_acc.get_attr("alloca")
        alloca_ = g.new_attr("alloca", SSAValue)
        alloca_.set(alloca)
        
        name = name_acc.get_attr("name")
        name_ = g.new_attr("name", SSAValue)
        name_.set(name)
        return g
    

    @parser
    def name_access(self):
        """
        name_access
            [syn alloca: Value, name: str]
            : Identifier {
                name = $Identifier.text
                alloca = table.find(name)
            }
            ;
        """
        id = self.Identifier()
        g = newParserRule(id)
        name = id.text()
        g.new_attr("alloca", SSAValue).set(table.find(name))
        g.new_attr("name", SSAValue).set(name)
        return g

    @parser
    def expression(self):
        """
            expression
                [syn v: Value]
                : prim_expr {v = $prim_expr.v}
                | add_expr {v = $add_expr.v}
                ;
        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        def alt1():
            expr = self.prim_expr()
            v.set(expr.get_attr("v"))
            return expr
        
        def alt2():
            expr = self.add_expr()
            v.set(expr.get_attr("v"))
            return expr

        g.production = alternate(alt1, alt2)
        return g
    
    @parser
    def prim_expr(self):
        """
        prim_expr
            [syn v: Value]
            : variable_access {v = memref.load($variable_access.alloca)}
            | Number {v = arith.constantOp(int($Number.text), IntegerType.get(32))}
            ;
        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        def alt1():
            var_acc = self.variable_access()
            alloca = var_acc.get_attr("alloca")
            load = memref.LoadOp.create([alloca])
            v.set(load)
            return var_acc
        
        def alt2():
            num = self.Number()
            const = arith.ConstantOp(IntegerAttr(int(num.text()), IntegerType(32)))
            v.set(const)
            return num
        
        g.production = alternate(alt1, alt2)
        return g

    @parser
    def add_expr(self):
        """
        add_expr
            [syn v: Value]
            : prim_expr Add add_expr {v = arith.addi($prim_expr.v, $add_expr.v)}
            ;
        """
        g = newParserRule()
        v = g.new_attr("v", SSAValue)
        prim = self.prim_expr()
        add = self.add_expr()
        lhs = prim.get_attr("v")
        rhs = add.get_attr("v")
        v.set(arith.AddiOp(lhs, rhs))
        g.production = concat(prim, self.Add(), add)
        return g
    
    @lexer
    def Add(self):
        """
            Add: '+';
        """
        return TerminalRule("+")
    
    @lexer
    def Number(self):
        """
            Number: '0-9' | '1-9' '0-9'*;
        """
        no_zero = char_set("1-9")
        all_number = char_set("0-9")
        return TerminalRule(alternate(lambda: all_number, lambda: concat(no_zero, zero_or_more(all_number))))
    
    @lexer
    def Dot(self):
        """
            Dot: '.';
        """
        return TerminalRule(".")
    
    @lexer
    def Identifier(self):
        """
            Identifier: [a-zA-Z][a-zA-Z0-9]*;
        """
        noDigit = char_set("a-zA-Z")
        allcase = char_set("a-zA-Z0-9")
        # return TerminalRule(concat(noDigit, zero_or_more(allcase)))
        return TerminalRule(noDigit, zero_or_more(allcase))

    @lexer
    def Assign(self):
        """
            Assign: '=';
        """
        return TerminalRule("=")
    
    @skip
    @lexer
    def skip(self):
        return TerminalRule(char_set("\n\t "))
    
if __name__ == "__main__":
    context = Context()
    context.load_dialect(arith.Arith)
    context.load_dialect(func.Func)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(memref.MemRef)
    code = """
        a = 1 + 2
        b = 2 + 3
        c = a + b
        d = b + c
    """
    grammar = MyGrammar()
    mylexer = grammar.lexer
    myparser = grammar.parser
    p = myparser(mylexer(code))
    tree = p.module()
    myfunc = tree.get_attr("module")
    printer = Printer()
    printer.print_op(myfunc)
    
"""
expected output:

```mlir
func.func @my_function() {
// a = 1 + 2
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %add0 = arith.addi %c1, %c2 : i32
  %a = memref.alloca() : memref<1xi32>
  memref.store %add0, $a[0] : memref<1xi32>
// b = 2 + 3
  %c3 = arith.constant 1 : i32
  %add1 = arith.addi %c2, %c3 : i32
  %b = memref.alloca() : memref<1xi32>
  memref.store %add1, $b[0] : memref<1xi32>
// c = a + b
  %a_ = memref.load %a[0] : memref<1xi32>
  %b_ = memref.load %b[0] : memref<1xi32>
  %add2 = arith.addi %a_, %b_ : i32
  %c = memref.alloca() : memref<1xi32>
  memref.store %add2, $c[0] : memref<1xi32>
// d = b + c
  %b_ = memref.load %b[0] : memref<1xi32>
  %c_ = memref.load %c[0] : memref<1xi32>
  %add3 = arith.addi %b_, %c_ : i32
  %d = memref.alloca() : memref<1xi32>
  memref.store %add3, $d[0] : memref<1xi32>
}
```

"""