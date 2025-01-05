from FeGen import grammar
from FeGen.grammar import FeGenLexer, FeGenParser, FeGenParserVisitor
from FeGen.visitor import *
import sys
from antlr4 import *


def main(argv):
    file_path = "example/for.fegen"
    input_stream = FileStream(file_path)
    lexer = FeGenLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = FeGenParser(stream)
    parserTreeRoot = parser.fegenSpec()

    ctx = Context()
    # visit and build rule tree
    ruleVisitor = FeGenRuleVisitor(ctx)
    ruleVisitor.visit(parserTreeRoot)
    # fuse tree node, handle element bind, collect return value of tree node, and expand rules
    ctx.processRules()
    # dump rule (default to stdout)
    ctx.dumpRule()
    # initialize visitor file accroding to rule tree
    ctx.initVisitorFile()
    # visit and build visitor file
    stmtBlockVisitor = FeGenStmtBlockVisitor(ctx)
    stmtBlockVisitor.visit(parserTreeRoot)
    # dump visitor file
    ctx.dumpVisitor(sys.stdout)


if __name__ == '__main__':
    main(sys.argv)
