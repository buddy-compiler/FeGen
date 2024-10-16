from grammar import FeGenParserVisitor
from grammar.FeGenParser import FeGenParser
from .Context import Context, RuleBuilder, VisitorBuilder
from .Rule import *


class FeGenStmtBlockVisitor(FeGenParserVisitor):
    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context: Context = context
        self.visitorBuilder: VisitorBuilder = self.context.visitorBuilder

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
        self.visitorBuilder.writeNewLine("visit statement block.")
