def attr_grammar(clazz):
    pass


def parser(func):
    pass


def lexer(func):
    pass


def skip(func):
    pass


def char_set(arg):
    """
        char_set("A-Z") --> [A-Z]
    """
    pass


def zero_or_more(arg):
    """
        zero_or_more(A) --> A*
    """
    pass


def one_or_more(arg):
    """
        one_or_more(A) --> A+ 
    """
    pass


def concat(*args):
    """
        concat(A, B) -->  A B
    """
    pass


def alternate(*arg):
    """
        alternate(A, B) --> A | B
    """


class FeGenGrammar:
    pass
