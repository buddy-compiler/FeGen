lexer grammar FeGenLexer;

fragment NONDIGIT: [a-zA-Z_];

fragment UPPERCASE: [A-Z];

fragment LOWERCASE: [a-z];

fragment ALLCASE: [a-zA-Z0-9_];

// string literal
fragment SHORT_STRING_LITERAL:
	'\'' SHORT_STRING_ITEM_FOR_SINGLE_QUOTE* '\''
	| '"' SHORT_STRING_ITEM_FOR_DOUBLE_QUOTE* '"';

fragment SHORT_STRING_ITEM_FOR_SINGLE_QUOTE:
	SHORT_STRING_CHAR_NO_SINGLE_QUOTE
	| STRING_ESCAPE_SEQ;
fragment SHORT_STRING_ITEM_FOR_DOUBLE_QUOTE:
	SHORT_STRING_CHAR_NO_DOUBLE_QUOTE
	| STRING_ESCAPE_SEQ;

fragment SHORT_STRING_CHAR_NO_SINGLE_QUOTE: ~[\\\r\n'];

fragment STRING_ESCAPE_SEQ: '\\' OS_INDEPENDENT_NL | '\\' .;

fragment SHORT_STRING_CHAR_NO_DOUBLE_QUOTE: ~[\\\r\n"];

fragment OS_INDEPENDENT_NL: '\r'? '\n';

fragment LONG_STRING_LITERAL:
	'\'\'\'' LONG_STRING_ITEM*? '\'\'\''; // | '"""' LONG_STRING_ITEM*? '"""'

fragment LONG_STRING_ITEM: LONG_STRING_CHAR | STRING_ESCAPE_SEQ;

fragment LONG_STRING_CHAR: ~'\\';

// integer literal
fragment INTEGER:
	DEC_INTEGER
	| BIN_INTEGER
	| OCT_INTEGER
	| HEX_INTEGER;
fragment DEC_INTEGER:
	NON_ZERO_DIGIT ('_'? DIGIT)*
	| '0'+ ('_'? '0')*;
fragment BIN_INTEGER: '0' ('b' | 'B') ('_'? BIN_DIGIT)+;
fragment OCT_INTEGER: '0' ('o' | 'O') ('_'? OCT_DIGIT)+;
fragment HEX_INTEGER: '0' ('x' | 'X') ('_'? HEX_DIGIT)+;
fragment NON_ZERO_DIGIT: [1-9];
fragment DIGIT: [0-9];
fragment BIN_DIGIT: '0' | '1';
fragment OCT_DIGIT: [0-7];
fragment HEX_DIGIT: DIGIT | [a-f] | [A-F];

// floatpoint literal
fragment FLOAT_NUMBER: POINT_FLOAT | EXPONENT_FLOAT;
fragment POINT_FLOAT: DIGIT_PART? FRACTION | DIGIT_PART '.';
fragment EXPONENT_FLOAT: (DIGIT_PART | POINT_FLOAT) EXPONENT;
fragment DIGIT_PART: DIGIT ('_'? DIGIT)*;
fragment FRACTION: '.' DIGIT_PART;
fragment EXPONENT: ('e' | 'E') ('+' | '-')? DIGIT_PART;

// boolean literal
fragment TRUE: 'True';
fragment FALSE: 'False';

// key words

FEGEN: 'fegen';

DEF: 'def';

INPUTS: 'inputs';

RETURNS: 'returns';

ACTIONS: 'actions';

IR: 'ir';

OPERAND_VALUE: 'operandValue';

ATTRIBUTE_VALUE: 'attributeValue';

CPP_VALUE: 'cppValue';

OPERATION: 'operation';

FUNCTION: 'function';

TYPEDEF: 'typedef';

OPDEF: 'opdef';

ARGUMENTS: 'arguments';

RESULTS: 'results';

BODY: 'body';

EMPTY: 'null';

PARAMETERS: 'parameters';

ASSEMBLY_FORMAT: 'assemblyFormat';

// types
TYPE: 'Type';

BOOL: 'bool';

INT: 'int';

FLOAT: 'float';

STRING: 'string';

LIST: 'list';

MAP: 'map';

// stmt

IF: 'if';

ELIF: 'elif';

ELSE: 'else';

FOR: 'for';

IN: 'in';

WHILE: 'while';

RETURN: 'return';

VARIABLE: 'variable';

// marks

AND: 'and';

OR: 'or';

NOT: 'not';

IS: 'is';

Equal: '==';

NotEq: '!=';

Less: '<';

LessEq: '<=';

Greater: '>';

GreaterEq: '>=';

AT: '@';

DivDiv: '//';

Comma: ',';

Semi: ';';

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Dot: '.';

Colon: ':';

AlterOp: '|';

QuestionMark: '?';

Star: '*';

Div: '/';

Plus: '+';

Minus: '-';

Assign: '=';

StarStar: '**';

MOD: '%';

Arror: '->';

Tilde: '~';

Range: '..';

// literal

StringLiteral: SHORT_STRING_LITERAL | LONG_STRING_LITERAL;

BoolLiteral: TRUE | FALSE;

IntegerLiteral: INTEGER;

FloatPointLiteral: FLOAT_NUMBER;

// identifiers

LexerRuleName: UPPERCASE (NONDIGIT | DIGIT)*;

ParserRuleName: LOWERCASE (NONDIGIT | DIGIT)*;


Whitespace: [ \t]+ -> skip;

Newline: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~ [\r\n]* -> skip;