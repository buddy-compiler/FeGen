import sys
from antlr4 import *
from grammar import MojoLexer
from grammar import MojoParser

def main(argv):
    input_stream = FileStream(argv[1])
    lexer = MojoLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = MojoParser(stream)
    tree = parser.var_assign()
    print(tree.toStringTree())


if __name__ == '__main__':
    main(sys.argv)