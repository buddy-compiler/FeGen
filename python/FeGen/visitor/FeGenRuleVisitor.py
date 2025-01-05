
from FeGen.grammar import FeGenParserVisitor
from FeGen.grammar import FeGenParser
from .Context import Context, RuleBuilder
from .Rule import *


class FeGenRuleVisitor(FeGenParserVisitor):
    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context: Context = context
        self.ruleBuilder: RuleBuilder = self.context.ruleBuilder

    def visitFegenDecl(self, ctx: FeGenParser.FegenDeclContext):
        self.context.moduleName = ctx.identifier().getText()

    def visitParserRuleSpec(self, ctx: FeGenParser.ParserRuleSpecContext):
        r = self.ruleBuilder.createRule(ctx.ParserRuleName().getText())
        self.ruleBuilder.setInsertPoint(r)
        self.visit(ctx.ruleBlock())

    def visitActionAlt(self, ctx: FeGenParser.ActionAltContext):
        altProd = self.ruleBuilder.createProd(ctx.alternative())
        self.ruleBuilder.setInsertPoint(altProd)
        self.visit(ctx.alternative())

    def visitAlternative(self, ctx: FeGenParser.AlternativeContext):
        if (not isinstance(ctx.parentCtx, FeGenParser.ActionAltContext)) and len(ctx.element()) > 1:
            r = self.ruleBuilder.createConcat()
            self.ruleBuilder.setInsertPoint(r)
        point = self.ruleBuilder.insertPoint()
        for e in ctx.element():
            self.visit(e)
            self.ruleBuilder.setInsertPoint(point)

    def visitElement(self, ctx: FeGenParser.ElementContext):
        if ctx.ebnfSuffix() is not None:
            suffix = ctx.ebnfSuffix().getText()
            isopt = False
            match suffix:
                case '?':
                    isopt = True
                    r = self.ruleBuilder.createOpt()
                case '*':
                    r = self.ruleBuilder.createStar()
                case '+':
                    r = self.ruleBuilder.createPlus()
                case _:
                    assert False
            self.ruleBuilder.setInsertPoint(r)
            if ctx.bindElement():
                bindName = ctx.bindElement().identifier().getText()
                if isopt:
                    element = BindElement(bindName)
                else:
                    element = BindElement(bindName)
                self.ruleBuilder.bindElement(element, r)

        self.visit(ctx.atomOrGroup())

    def visitTerminalDef(self, ctx: FeGenParser.TerminalDefContext):
        if ctx.LexerRuleName():
            terminalRefTree = self.ruleBuilder.createTerminalRef(
                ctx.LexerRuleName().getText())
            if ctx.bindElement():
                bindName = ctx.bindElement().identifier().getText()
                element = BindElement(bindName)
                self.ruleBuilder.bindElement(element, terminalRefTree)
        else:
            self.ruleBuilder.createTerminal(ctx.StringLiteral().getText())

    def visitRuleref(self, ctx: FeGenParser.RulerefContext):
        ruleRef = self.ruleBuilder.createRuleRef(
            ctx.ParserRuleName().getText())
        if ctx.bindElement():
            bindName = ctx.bindElement().identifier().getText()
            element = BindElement(bindName)
            self.ruleBuilder.bindElement(element, ruleRef)

    def visitNotSet(self, ctx: FeGenParser.NotSetContext):
        print("not implemented yet")

    def visitGroup(self, ctx: FeGenParser.GroupContext):
        r = self.ruleBuilder.createGroup()
        self.ruleBuilder.setInsertPoint(r)
        self.visit(ctx.altList())
        if ctx.bindElement():
            bindName = ctx.bindElement().identifier().getText()
            element = BindElement(bindName)
            self.ruleBuilder.bindElement(element, r)

    def visitAltList(self, ctx: FeGenParser.AltListContext):
        if len(ctx.alternative()) == 1:
            self.visit(ctx.alternative(0))
        else:
            r = self.ruleBuilder.createAlt()
            self.ruleBuilder.setInsertPoint(r)
            for alt in ctx.alternative():
                self.visit(alt)
                self.ruleBuilder.setInsertPoint(r)

    def visitLexerRuleSpec(self, ctx: FeGenParser.LexerRuleSpecContext):
        r = self.ruleBuilder.createRule(ctx.LexerRuleName().getText(), True)
        self.ruleBuilder.setInsertPoint(r)
        prod = self.ruleBuilder.createProd()
        self.ruleBuilder.setInsertPoint(prod)
        self.visit(ctx.lexerRuleBlock())

    def visitLexerAltList(self, ctx: FeGenParser.LexerAltListContext):
        if len(ctx.lexerAlt()) == 1:
            self.visit(ctx.lexerAlt(0))
        else:
            alt = self.ruleBuilder.createAlt()
            self.ruleBuilder.setInsertPoint(alt)
            for e_lexerAlt in ctx.lexerAlt():
                self.visit(e_lexerAlt)
                self.ruleBuilder.setInsertPoint(alt)

    def visitLexerAlt(self, ctx: FeGenParser.LexerAltContext):
        self.visit(ctx.lexerElements())
        # handle lexer commands
        if ctx.lexerCommands() is not None:
            cmd_text = [cmd_ctx.getText()
                        for cmd_ctx in ctx.lexerCommands().lexerCommand()]
            for cmd in cmd_text:
                if cmd in LexRule.COMMANDS_MAP:
                    point = self.ruleBuilder.insertPoint()
                    rule = point.belongTo.rule
                    rule.addCommand(LexRule.COMMANDS_MAP[cmd])

    def visitLexerElements(self, ctx: FeGenParser.LexerElementsContext):
        point = self.ruleBuilder.insertPoint()
        # if point is not the root of tree, and len(ctx.lexerElement()>1), create concat
        if (not point.isRoot()) and (len(ctx.lexerElement()) > 1):
            point = self.ruleBuilder.createConcat()
        for e in ctx.lexerElement():
            self.visit(e)
            self.ruleBuilder.setInsertPoint(point)

    def visitLexerElement(self, ctx: FeGenParser.LexerElementContext):
        if ctx.ebnfSuffix() is not None:
            suffix = ctx.ebnfSuffix().getText()
            match suffix:
                case '?':
                    r = self.ruleBuilder.createOpt()
                case '*':
                    r = self.ruleBuilder.createStar()
                case '+':
                    r = self.ruleBuilder.createPlus()
                case _:
                    assert False
            self.ruleBuilder.setInsertPoint(r)
        self.visit(ctx.lexerAtomOrGroup())

    def visitCharacterRange(self, ctx: FeGenParser.CharacterRangeContext):
        self.ruleBuilder.createTerminal(ctx.getText())

    def visitLexerGroup(self, ctx: FeGenParser.LexerGroupContext):
        t = self.ruleBuilder.createGroup()
        self.ruleBuilder.setInsertPoint(t)
        self.visit(ctx.lexerAltList())
