lexer grammar MojoLexer;

options {
	superClass = MojoLexerBase;
}

tokens {
	INDENT,
	DEDENT
}

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
	'\'\'\'' LONG_STRING_ITEM*? '\'\'\''
	; // | '"""' LONG_STRING_ITEM*? '"""'

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
DEF: 'def';

FN: 'fn';

RETURN: 'return';

PASS: 'pass';

VAR: 'var';

IF: 'if';

ELIF: 'elif';

ELSE: 'else';

WHILE: 'while';

CONTINUE: 'continue';

BREAK: 'break';

FOR: 'for';

IN: 'in';

STRUCT: 'struct';

INOUT: 'inout';

IMPORT: 'import';

FROM: 'from';

RAISES: 'raises';

RAISE: 'raise';

STRING: 'String';

AS: 'as';

ALIAS: 'alias';

AND: 'and';

NOT: 'not';

OR: 'or';

IS: 'is';

INT: 'Int';

INT8: 'Int8';

UINT8: 'UInt8';

INT16: 'Int16';

UINT16: 'UInt16';

INT32: 'Int32';

UINT32: 'UInt32';

INT64: 'Int64';

UINT64: 'UInt64';

FLOAT16: 'Float16';

FLOAT32: 'Float32';

FLOAT64: 'Float64';

SIMD: 'SIMD';

LIST: 'List';

DICT: 'Dict';

OPTIONAL: 'Optional';

ANYTYPE: 'AnyType';

ANY_TRIVIAL_REG_TYPE: 'AnyTrivialRegType';

NONE: 'None';

String_literal: SHORT_STRING_LITERAL | LONG_STRING_LITERAL;

Bool_literal: TRUE | FALSE;

Integer_literal: INTEGER;

FloatPoint_literal: FLOAT_NUMBER;

Identifier: NONDIGIT ALLCASE*;

// marks
Comma: ',';

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Dot: '.';

Colon: ':';

BITWISE_OR: '|';

BITWISE_XOR: '^';

BITWISE_AND: '&';

LEFT_SHIFT: '<<';

RIGHT_SHIFT: '>>';

Star: '*';

Div: '/';

DivDiv: '//';

Plus: '+';

Minus: '-';

Assign: '=';

ModAssign: '%=';

DivAssign: '/=';

DivDivAssign: '//=';

MinusAssign: '-=';

PlusAssign: '+=';

StarAssign: '*=';

StarStarAssign: '**=';

Equal: '==';

NotEq: '<>' | '!=';

Less: '<';

Greater: '>';

LessEq: '<=';

GreaterEq: '>=';

StarStar: '**';

MOD: '%';

AT: '@';

RightArror: '->';

Tilde: '~';

TriDot: '...';

NEWLINE: '\n';

WS: [\t\f ] -> channel(HIDDEN);

COMMENT: ('#' ~[\r\n]* | '"""' LONG_STRING_ITEM*? '"""') -> channel(HIDDEN);