from grammar import FeGenParserVisitor
from grammar.FeGenParser import FeGenParser
from .Context import Context, RuleBuilder, VisitorBuilder
from .Rule import *
from .Variable import Variable
from typing import List, Callable, Tuple
from antlr4 import ParserRuleContext


class FeGenStmtBlockVisitor(FeGenParserVisitor):
    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context: Context = context
        self.visitorBuilder: VisitorBuilder = self.context.visitorBuilder
        self.symbolTable: List[Variable] = []

    def yieldOf(ctx) -> List[str]:
        if isinstance(ctx, FeGenParser.StatementBlockContext):
            ret = []
            for stmt in ctx.statement():
                ret += FeGenStmtBlockVisitor.yieldOf(stmt.children[0])
            return list(set(ret))
        elif isinstance(ctx, FeGenParser.AssignStmtContext):
            return [ctx.identifier().getText()]
        elif isinstance(ctx, FeGenParser.IfStmtContext):
            yieldsOfIf = FeGenStmtBlockVisitor.yieldOf(
                ctx.ifBlock().statementBlock())
            # handle elif(s)
            for elifblock in ctx.elifBlock():
                yieldsOfElif = FeGenStmtBlockVisitor.yieldOf(
                    elifblock.statementBlock())
                yieldsOfIf.extend(yieldsOfElif)
            # handle else
            if ctx.elseBlock() is not None:
                yieldsOfElse = FeGenStmtBlockVisitor.yieldOf(
                    ctx.elseBlock().statementBlock())
                yieldsOfIf.extend(yieldsOfElse)
            return list(set(yieldsOfIf))
        elif isinstance(ctx, FeGenParser.ForStmtContext):
            iterVar = ctx.identifier().getText()
            yieldsOfFor = FeGenStmtBlockVisitor.yieldOf(ctx.statementBlock())
            yieldsOfFor.append(iterVar)
            return list(set(yieldsOfFor))

    def visitActionAlt(self, ctx: FeGenParser.ActionAltContext):
        def visitBody(prod: RuleProd):
            root = prod.ruleTree
            root.dumpVisitor(self.visitorBuilder)
            self.visit(ctx.statementBlock())

        if ctx.statementBlock():
            prod = self.context.getProd(ctx.alternative())
            self.visitorBuilder.createStatementBlock(
                head="def visit{0}(self, ctx: {1}.{0}Context):".format(
                    prod.uniName.capitalize(), self.context.moduleName),
                body=lambda: visitBody(prod),
                tail="\n"
            )

    def visitStatementBlock(self, ctx: FeGenParser.StatementBlockContext):
        self.symbolTable.clear()
        super().visitStatementBlock(ctx)

    def visitAssignStmt(self, ctx: FeGenParser.AssignStmtContext):
        def body():
            self.visitorBuilder.write(ctx.expression().getText())
            if ctx.typeSpec():
                self.visitorBuilder.write(", ")
                self.visit(ctx.typeSpec())

        varName = ctx.identifier().getText()
        self.visitorBuilder.writeNewLine("{} = Value.create(".format(varName))
        body()
        self.visitorBuilder.write(")")
        # self.visitorBuilder.createStatementBlock(
        #     head="{} = Value.create(".format(varName),
        #     body=body,
        #     tail=")"
        # )
        self.symbolTable.append(Variable(varName))

    def visitIfStmt(self, ctx: FeGenParser.IfStmtContext):
        condAndBodies: List[Tuple[str, Callable[[None], None]]] = []
        ifCond = ctx.ifBlock().expression().getText()
        def ifBody(): self.visit(ctx.ifBlock().statementBlock())
        condAndBodies.append((ifCond, ifBody))
        for elifStmt in ctx.elifBlock():
            elifCond = elifStmt.expression().getText()
            def elifBody(): self.visit(elifStmt.statementBlock())
            condAndBodies.append((elifCond, elifBody))
        if ctx.elseBlock() is not None:
            def elseBody(): self.visit(ctx.elseBlock().statementBlock())
        else:
            elseBody = None
        usingVariable = FeGenStmtBlockVisitor.yieldOf(ctx)
        self.visitorBuilder.createIFStmt(
            condAndBodies, elseBody, usingVariable)

    def visitForStmt(self, ctx: FeGenParser.ForStmtContext):
        iterVar = ctx.identifier().getText()
        iterable = ctx.expression().getText()
        def forBody(): self.visit(ctx.statementBlock())
        self.visitorBuilder.createForStmt(
            iterVar, iterable, forBody
        )

    def visitReturnStmt(self, ctx: FeGenParser.ReturnStmtContext):
        returnVar = ctx.expression().getText()
        self.visitorBuilder.writeNewLine(
            "return {}".format(returnVar)
        )

    def visitFunctionCall(self, ctx: FeGenParser.FunctionCallContext):
        funcName = ctx.funcName().getText()
        if funcName == "visit":
            child = ctx.expression(0).getText()
            self.visitorBuilder.writeNewLine(
                "self.visit(ctx.{}())".format(child))
        else:
            self.visitorBuilder.writeNewLine(ctx.getText())

    def visitVariableDeclStmt(self, ctx: FeGenParser.VariableDeclStmtContext):
        self.visitorBuilder.writeNewLine(
            "{}.setVariable()".format(ctx.identifier().getText()))
