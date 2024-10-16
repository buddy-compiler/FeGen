from __future__ import annotations
from .Rule import *
from typing import Dict, Callable
import sys
from io import StringIO
from antlr4 import ParserRuleContext


class Context:
    def __init__(self) -> None:
        self.moduleName: str = None
        self.rules: Dict[str, Rule] = {}
        self.prods: Dict[ParserRuleContext, RuleProd] = {}

        self.ruleBuilder: RuleBuilder = RuleBuilder(self)
        self.visitorBuilder: VisitorBuilder = VisitorBuilder(self)

        self.g4File: TextIO = self.ruleBuilder.file
        self.visitorFile: TextIO = self.visitorBuilder.file

    def addRule(self, rule: Rule):
        self.rules[rule.name] = rule

    def addProd(self, prod: RuleProd, ctx: ParserRuleContext):
        self.prods[ctx] = prod

    def getProd(self, ctx: ParserRuleContext) -> RuleProd:
        return self.prods[ctx]

    def ruleFusion(self):
        """
            fuse tree like: group-concat => groupAndConcat, group-alt => groupAndAlt
        """
        for rule in self.rules.values():
            if rule.isTerminal:
                continue
            for prod in rule.alts:
                tree = prod.ruleTree
                stack = []
                stack.append(tree)
                while stack:
                    curr: RuleTree = stack.pop()
                    exchange = False
                    if isinstance(curr, RuleGroup):
                        child = curr.children[0]
                        if isinstance(child, RuleConcat):
                            newRule = RuleGroupAndConcat()
                            exchange = True
                        elif isinstance(child, RuleAlt):
                            newRule = RuleGroupAndAlt()
                            exchange = True
                    if exchange:
                        # set parent
                        parent = curr.parent
                        index = parent.children.index(curr)
                        parent.changeChild(newRule, index)
                        # set child
                        for childchild in child.children:
                            newRule.addChild(childchild)
                        # set bind
                        newRule.bindElement = curr.bindElement

                        curr = newRule

                    stack += curr.children

    def handleBind(self):
        """
            generate bind elements
        """
        for rule in self.rules.values():
            if rule.isTerminal:
                continue
            for prod in rule.alts:
                tree: RuleConcat = prod.ruleTree
                stack = []
                stack += tree.children
                while stack:
                    currentTreeNode: RuleTree = stack.pop()
                    if currentTreeNode.isBinded():
                        if isinstance(currentTreeNode, (RulePlus, RuleStar, RuleOpt)):
                            child = currentTreeNode.children[0]
                            elementName = "element_of_" + currentTreeNode.bindElement.name
                            element = BindElement(elementName, True)
                            child.bindElement = element
                        elif isinstance(currentTreeNode, RuleGroupAndAlt):
                            for idx, child in enumerate(currentTreeNode.children):
                                if not child.isBinded():
                                    elementName = "alt" + str(idx)
                                    element = BindElement(elementName, True)
                                    child.bindElement = element
                    stack += currentTreeNode.children

    def handleChildElement(self):
        """
            collect element from children
        """
        def handle(tree: RuleTree):
            # process children first
            for child in tree.children:
                handle(child)
            # then process parent
            for child in tree.children:
                childRet = child.returnList()
                if childRet:
                    tree.elementFromChild[child] = childRet

        for rule in self.rules.values():
            if rule.isTerminal:
                continue
            for prod in rule.alts:
                tree = prod.ruleTree
                handle(tree)

    def expandRules(self):
        rules = list(self.rules.values())
        for rule in rules:
            if rule.isTerminal:
                continue
            for prod in rule.alts:
                tree = prod.ruleTree
                stack = [tree]
                while stack:
                    curr: RuleTree = stack.pop()
                    if not curr.isRoot():
                        r = self.ruleBuilder.createRule(curr.uniName)
                        r.isGenerated = True
                        self.ruleBuilder.setInsertPoint(r)
                        prod = self.ruleBuilder.createProd()
                        self.ruleBuilder.setInsertPoint(prod)
                        self.ruleBuilder.appendRuleTree(curr)
                    for idx, child in enumerate(curr.children):
                        if not isinstance(child, (RuleTermi, RuleRefTermi, RuleRef)) and (child.elementFromChild or child.bindElement is not None):
                            newChild = RuleRef(str(child))
                            if child.isBinded():
                                newChild.bindElement = child.bindElement
                            curr.changeChild(newChild, idx)
                            stack.append(child)

    def processRules(self):
        self.ruleFusion()
        self.handleBind()
        self.handleChildElement()
        self.expandRules()

    def dumpRule(self, output: TextIO = sys.stdout):
        for rule in self.rules.values():
            rule.dumpG4(self.ruleBuilder)
        output.write(self.g4File.getvalue())

    def initVisitorFile(self):
        self.visitorFile.write(
            "class FeGen{0}Visitor({0}Visitor):".format(self.moduleName))
        self.visitorBuilder.enterBlock()

        self.visitorBuilder.createStatementBlock(
            "def __init__(self) -> None:",
            lambda: self.visitorBuilder.writeNewLine("super().__init__()")
        )

        def genBodyForGeneratedRule(rule: Rule):
            prod = rule.alts[0]
            root = prod.ruleTree
            if len(root.children) == 1:
                child = root.children[0]
                child.dumpVisitor(self.visitorBuilder)
            else:
                root.dumpVisitor(self.visitorBuilder)

        rules = self.rules.values()
        for rule in rules:
            if rule.isGenerated:
                assert len(rule.alts) == 1
                self.visitorBuilder.createStatementBlock(
                    head="def visit{0}(self, ctx: {1}.{0}Context):".format(
                        rule.name.capitalize(), self.moduleName),
                    body=lambda: genBodyForGeneratedRule(rule)
                )

    def dumpVisitor(self, output: TextIO = sys.stdout):
        output.write(self.visitorFile.getvalue())


class Builder:
    def __init__(self, context: Context) -> None:
        self.context: Context = context
        self.file: TextIO = StringIO()
        self.tab: int = 0

    def enterBlock(self):
        self.tab += 1

    def exitBlock(self):
        assert self.tab != 0
        self.tab -= 1

    def writeNewLine(self, code: str):
        self.file.write("\n")
        for _ in range(self.tab):
            self.file.write("\t")
        self.file.write(code)

    def write(self, code: str):
        self.file.write(code)

    def createStatementBlock(self, head: str, body: Callable, tail: str = ""):
        self.writeNewLine(head)
        self.enterBlock()
        body()
        self.exitBlock()
        if tail != "":
            self.writeNewLine(tail)


class RuleBuilder(Builder):
    def __init__(self, context: Context) -> None:
        self.processingRule: Rule = None
        self.processingProd: RuleProd = None
        self.processingRuleTree: RuleTree = None
        self.tailTreeRule: RuleTree = None
        super().__init__(context)

    def createRule(self, ruleName: str, isTerminal: bool = False) -> Rule:
        if isTerminal:
            r = LexRule(ruleName)
        else:
            r = Rule(ruleName, isTerminal)
        self.context.addRule(r)
        return r

    def setInsertPoint(self, point):
        if isinstance(point, RuleTree):
            assert not isinstance(point, (RuleRef, RuleTermi))
            self.processingRule = point.belongTo.rule
            self.processingProd = point.belongTo
            self.processingRuleTree = point
        elif isinstance(point, RuleProd):
            self.processingRule = point.rule
            self.processingProd = point
            self.processingRuleTree = point.ruleTree
        elif isinstance(point, Rule):
            self.processingRule = point
            self.processingProd = None
            self.processingRuleTree = None
        else:
            assert False

    def insertPoint(self) -> RuleTree:
        return self.processingRuleTree

    def createProd(self, ctx: ParserRuleContext = None) -> RuleProd:
        p = RuleProd(self.processingRule)
        self.processingRule.addRuleProd(p)
        self.tailTreeRule = p.ruleTree
        if ctx is not None:
            self.context.addProd(p, ctx)
        return p

    def createConcat(self) -> RuleConcat:
        t = RuleConcat()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createAlt(self) -> RuleAlt:
        t = RuleAlt()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createGroup(self) -> RuleGroup:
        t = RuleGroup()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createStar(self) -> RuleStar:
        t = RuleStar()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createPlus(self) -> RulePlus:
        t = RulePlus()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createOpt(self) -> RuleOpt:
        t = RuleOpt()
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createTerminal(self, content: str) -> RuleTermi:
        t = RuleTermi(content)
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createTerminalRef(self, ruleName: str) -> RuleRefTermi:
        t = RuleRefTermi(ruleName)
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    def createRuleRef(self, ruleName: str) -> RuleRef:
        t = RuleRef(ruleName)
        self.processingRuleTree.addChild(t)
        self.tailTreeRule = t
        return t

    # append an exist RuleTree as the child of RuleTree
    def appendRuleTree(self, tree: RuleTree) -> None:
        self.processingRuleTree.addChild(tree)

    def bindElement(self, element: BindElement, tree: RuleTree) -> None:
        tree.bindElement = element


class VisitorBuilder(Builder):
    def __init__(self, context: Context) -> None:
        super().__init__(context)

    def visitChild(self, child: RuleTree, childRetList: List[str]) -> None:
        if isinstance(child, RuleRefTermi):
            fstr = "{} = ctx.{}({}).getText()"
        else:
            fstr = "{} = self.visit(ctx.{}({}))"

        name = str(child)
        # count of str(ch) == str(child)
        count = 0
        # index of child
        index = 0
        for ch in child.parent.children:
            if str(ch) == name:
                if ch != child:
                    count += 1
                else:
                    index = count
        self.writeNewLine(
            fstr.format(", ".join(childRetList), str(
                child), str(index) if count > 0 else "")
        )
