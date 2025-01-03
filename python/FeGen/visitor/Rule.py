from __future__ import annotations
from typing import List, TextIO, Dict
from enum import Enum


class BindElement:
    def __init__(self, name: str, isGenerated: bool = False) -> None:
        self.name = name
        self.isGenerated = isGenerated
        self.consistOf: List[BindElement] = []


class RuleTree:
    def __init__(self) -> None:
        self.bindName: str = None
        self.bindElement: BindElement = None
        self.elementFromChild: Dict[RuleTree, List[BindElement]] = {}

        self.parent: RuleTree = None
        self.belongTo: RuleProd = None
        self.children: List[RuleTree] = []
        self.uniName: str = "EMPTY"

    def returnList(self) -> List[BindElement]:
        collected = []
        for elements in self.elementFromChild.values():
            if elements[-1].isGenerated:
                collected += elements[0:-1]
            else:
                collected += elements
        if self.isBinded():
            collected.append(self.bindElement)
        return collected

    def isRoot(self) -> bool:
        return self.parent is None

    def isBinded(self) -> bool:
        return self.bindElement is not None

    def addChild(self, child):
        child.parent = self
        child.belongTo = self.belongTo
        child.uniName = self.uniName + "_" + \
            type(child).NAME + str(len(self.children))

        self.children.append(child)
        self._guardAddChild()

    def changeChild(self, child, index):
        child.parent = self
        child.belongTo = self.belongTo
        child.uniName = self.uniName + "_" + type(child).NAME + str(index)

        oldChild = self.children[index]
        elems = self.elementFromChild.pop(oldChild, None)
        if elems is not None:
            self.elementFromChild[child] = elems

        self.children[index] = child

    def __str__(self) -> str:
        return self.uniName

    def dumpG4(self, ruleBuilder) -> None:
        raise NotImplementedError()

    def _guardAddChild(self):
        raise NotImplementedError()

    def dumpVisitor(self, builder):
        raise NotImplementedError()


class RuleConcat(RuleTree):
    NAME = "concat"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self) -> None:
        return

    def dumpG4(self, file: TextIO) -> None:
        for child in self.children:
            child.dumpG4(file)
            file.write(" ")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        for child in self.children:
            child.dumpG4(ruleBuilder)
            ruleBuilder.write(" ")

    def dumpVisitor(self, builder) -> None:
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        returnNames = []
        for child, elements in self.elementFromChild.items():
            elementNames = [e.name for e in elements]
            returnNames += elementNames
            builder.visitChild(child, elementNames)
        if self.isBinded():
            builder.writeNewLine(
                "{} = ({})".format(
                    self.bindElement.name, ", ".join(returnNames))
            )
        if not self.isRoot():
            builder.writeNewLine(
                "return ({})".format(
                    ", ".join([e.name for e in self.returnList()]))
            )


class RuleAlt(RuleTree):
    NAME = "alt"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self):
        return

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        for idx, child in enumerate(self.children):
            child.dumpG4(ruleBuilder)
            if idx != 1:
                ruleBuilder.write(" | ")


class RuleGroup(RuleTree):
    NAME = "group"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self):
        if len(self.children) > 1:
            raise ValueError(
                "Rule terminal should have no more than one child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write("( ")
        self.children[0].dumpG4(ruleBuilder)
        ruleBuilder.write(" )")

    def dumpVisitor(self, builder):
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        child = self.children[0]
        elements = self.elementFromChild[child]
        elementNames = [e.name for e in elements]
        # visit child and get return
        builder.visitChild(child, elementNames)

        def createMap():
            if child.bindElement is not None:
                builder.writeNewLine(
                    '"{0}": {0}'.format(child.bindElement.name)
                )

        if self.bindElement is not None:
            builder.createStatementBlock(
                head="{}".format(self.bindElement.name) + " = {",
                body=createMap,
                tail="}"
            )

        builder.writeNewLine(
            "return ({})".format(
                ", ".join([e.name for e in self.returnList()]))
        )


class RuleTermi(RuleTree):
    NAME = "terminal"

    def __init__(self, content: str) -> None:
        self.content: str = content
        super().__init__()

    def __str__(self) -> str:
        return self.content

    def _guardAddChild(self):
        if len(self.children) > 0:
            raise ValueError("Rule terminal should have zero child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write(self.content)


class RuleRefTermi(RuleTree):
    NAME = "terminalRef"

    def __init__(self, ruleName: str) -> None:
        super().__init__()
        self.ruleName: str = ruleName

    def __str__(self) -> str:
        return self.ruleName

    def _guardAddChild(self):
        if len(self.children) > 0:
            raise ValueError("Rule terminal ref should have zero child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write(self.ruleName)


class RuleRef(RuleTree):
    NAME = "ruleRef"

    def __init__(self, ruleName: str) -> None:
        super().__init__()
        self.ruleName: str = ruleName

    def __str__(self) -> str:
        return self.ruleName

    def _guardAddChild(self):
        if len(self.children) > 0:
            raise ValueError("Rule ref should have zero child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write(self.ruleName)


class RuleStar(RuleTree):
    NAME = "star"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self):
        if len(self.children) > 1:
            raise ValueError("Rule star should have no more than one child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        self.children[0].dumpG4(ruleBuilder)
        ruleBuilder.write("*")

    def dumpVisitor(self, builder):
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        child = self.children[0]
        elements = self.elementFromChild[child]
        elementNames = [e.name for e in elements]
        for name in elementNames:
            builder.writeNewLine(
                "{}_list = []".format(name)
            )

        def createList():
            builder.writeNewLine(
                "{} = self.visit(child)".format(", ".join(elementNames))
            )
            for name in elementNames:
                builder.writeNewLine(
                    "{0}_list.append({0})".format(name)
                )

        builder.createStatementBlock(
            head="for child in ctx.{}():".format(str(child)),
            body=createList
        )
        for name in elementNames:
            builder.writeNewLine(
                "{0} = {0}_list".format(name)
            )
        if self.isBinded():
            childBindElem = child.bindElement
            builder.writeNewLine(
                "{} = {}".format(self.bindElement.name, childBindElem.name)
            )

        builder.writeNewLine(
            "return ({})".format(
                ", ".join([e.name for e in self.returnList()]))
        )


class RulePlus(RuleTree):
    NAME = "plus"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self):
        if len(self.children) > 1:
            raise ValueError("Rule plus should have no more than one child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        self.children[0].dumpG4(ruleBuilder)
        ruleBuilder.write("+")

    def dumpVisitor(self, builder):
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        returnNames = []
        child = self.children[0]
        elements = self.elementFromChild[child]
        elementNames = [e.name for e in elements]
        returnNames += elementNames
        for name in elementNames:
            builder.writeNewLine(
                "{}_list = []".format(name)
            )

        def createList():
            builder.writeNewLine(
                "{} = self.visit(child)".format(", ".join(elementNames))
            )
            for name in elementNames:
                builder.writeNewLine(
                    "{0}_list.append({0})".format(name)
                )

        builder.createStatementBlock(
            head="for child in ctx.{}():".format(str(child)),
            body=createList
        )
        for name in elementNames:
            builder.writeNewLine(
                "{0} = {0}_list".format(name)
            )
        if self.isBinded():
            childBindElem = child.bindElement
            builder.writeNewLine(
                "{} = {}".format(self.bindElement.name, childBindElem.name)
            )

        builder.writeNewLine(
            "return ({})".format(
                ", ".join([e.name for e in self.returnList()]))
        )


class RuleOpt(RuleTree):
    NAME = "opt"

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return super().__str__()

    def _guardAddChild(self):
        if len(self.children) > 1:
            raise ValueError("Rule opt should have no more than one child.")

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        self.children[0].dumpG4(ruleBuilder)
        ruleBuilder.write("?")

    def dumpVisitor(self, builder):
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        # initialize returns of child to None
        child = self.children[0]
        elements = self.elementFromChild[child]
        elementNames = [e.name for e in elements]
        for name in elementNames:
            builder.writeNewLine(
                "{} = None".format(name)
            )

        # assign if child exist
        builder.createStatementBlock(
            head="if ctx.{}() is not None:".format(str(child)),
            body=lambda: builder.visitChild(child, elementNames)
        )
        # if self is binded, assign self.bindElement
        if self.isBinded():
            childBindElem = child.bindElement
            builder.writeNewLine(
                "{} = {}".format(self.bindElement.name, childBindElem.name)
            )

        builder.writeNewLine(
            "return ({})".format(
                ", ".join([e.name for e in self.returnList()]))
        )


class RuleCombined(RuleTree):
    def __init__(self) -> None:
        super().__init__()


class RuleGroupAndConcat(RuleCombined):
    NAME = "group_concat"

    def __init__(self) -> None:
        super().__init__()

    def _guardAddChild(self) -> None:
        return

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write("( ")
        for idx, child in enumerate(self.children):
            if idx > 0:
                ruleBuilder.write(" ")
            child.dumpG4(ruleBuilder)
        ruleBuilder.write(" )")

    def dumpVisitor(self, builder) -> None:
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        # visit child and get returns
        for child, elements in self.elementFromChild.items():
            elementNames = [e.name for e in elements]
            builder.visitChild(child, elementNames)

        def createMap(mapped_names):
            for name in mapped_names:
                builder.writeNewLine(
                    '"{0}": {0}, '.format(name)
                )

        # if self is Binded by a element, get it
        if self.isBinded():
            mapped_names = []
            for child in self.children:
                if child.isBinded():
                    mapped_names.append(child.bindElement.name)
            builder.createStatementBlock(
                head="{}".format(self.bindElement.name) + " = {",
                body=lambda: createMap(mapped_names),
                tail="}"
            )

        # return values
        builder.writeNewLine(
            "return ({})".format(
                ", ".join([e.name for e in self.returnList()]))
        )


class RuleGroupAndAlt(RuleCombined):
    NAME = "group_alt"

    def __init__(self) -> None:
        super().__init__()

    def _guardAddChild(self) -> None:
        return

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        ruleBuilder.write("( ")
        for idx, child in enumerate(self.children):
            if idx > 0:
                ruleBuilder.write(" | ")
            child.dumpG4(ruleBuilder)
        ruleBuilder.write(" )")

    def dumpVisitor(self, builder):
        from .Context import VisitorBuilder
        builder: VisitorBuilder = builder

        # initialize returns of child to None
        for elements in self.elementFromChild.values():
            elementNames = [e.name for e in elements]
            builder.writeNewLine(
                "{} = {}".format(", ".join(elementNames),
                                 ", ".join(["None"] * len(elementNames)))
            )

        returnNames = [e.name for e in self.returnList()]

        def ifBody(child, childRetNames):
            # visit branch
            builder.visitChild(child, childRetNames)
            # assign element of self
            if self.isBinded():
                builder.writeNewLine(
                    "{} = {}".format(self.bindElement.name,
                                     child.bindElement.name)
                )
            # return elements
            builder.writeNewLine(
                "return ({})".format(", ".join(returnNames))
            )

        # visit exist branch and return
        for idx, (child, elements) in enumerate(self.elementFromChild.items()):
            elementNames = [e.name for e in elements]
            if idx == 0:
                builder.createStatementBlock(
                    head="if ctx.{}() is not None:".format(str(child)),
                    body=lambda: ifBody(child, elementNames)
                )
            else:
                builder.createStatementBlock(
                    head="elif ctx.{}() is not None".format(str(child)),
                    body=lambda: ifBody(child, elementNames)
                )


class RuleProd:
    def __init__(self, rule: Rule) -> None:
        self.ruleTree: RuleConcat = RuleConcat()
        self.ruleTree.belongTo = self
        self.rule: Rule = rule
        self.uniName: str = ""

    def __str__(self) -> str:
        return self.uniName

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder
        self.ruleTree.dumpG4(ruleBuilder)

    def setUniName(self, name: str):
        self.uniName = name
        self.ruleTree.uniName = name


class Rule:
    def __init__(self, ruleName: str, isTerminal=False) -> None:
        self.isTerminal = isTerminal
        self.isGenerated = False
        self.name = ruleName
        self.alts: List[RuleProd] = []

    def addRuleProd(self, alt: RuleProd) -> None:
        if len(self.alts) == 0:
            uniName = self.name
        else:
            uniName = self.name + str(len(self.alts))

        if len(self.alts) == 1:
            self.alts[0].setUniName(self.name + "0")

        alt.setUniName(uniName)
        self.alts.append(alt)
        self._guardAddRuleProd()

    def dumpG4(self, ruleBuilder) -> None:
        from .Context import RuleBuilder
        ruleBuilder: RuleBuilder = ruleBuilder

        def ruleBody():
            doSpilt = (len(self.alts) > 1)
            for index, alt in enumerate(self.alts):
                if index == 0:
                    ruleBuilder.writeNewLine(": ")
                else:
                    ruleBuilder.writeNewLine("| ")
                alt.dumpG4(ruleBuilder)
                if doSpilt:
                    ruleBuilder.write("\t#")
                    ruleBuilder.write(alt.uniName)
            ruleBuilder.writeNewLine(";\n")

        ruleBuilder.createStatementBlock(
            head=self.name,
            body=ruleBody
        )

    def _guardAddRuleProd(self):
        pass


class LexRule(Rule):
    COMMANDS_MAP = {
        "skip": 0
    }

    def __init__(self, ruleName: str) -> None:
        super().__init__(ruleName, True)
        self.commands: List[int] = []

    def addCommand(self, cmd):
        if cmd not in self.commands:
            self.commands.append(cmd)

    def dumpG4(self, file: TextIO) -> None:
        file.write(self.name)
        file.write(": ")
        self.alts[0].dumpG4(file)
        file.write("; \n")

    def _guardAddRuleProd(self):
        if len(self.alts) > 1:
            raise ValueError(
                "Terminal Rule should have no more than one prod.")
