parser grammar FeGenParser;

options {
    tokenVocab = FeGenLexer;
}

fegenSpec
    : fegenDecl (functionDecl | rules)* EOF
    ;

fegenDecl
    : FEGEN identifier Semi
    ;

// ======== function defination ======== //

functionDecl
    : DEF funcName LeftParen funcParams? RightParen statementBlock
    ;

funcName
    : identifier
    | builtinType
    ;

funcParams
    : identifier (Colon typeSpec)? (Comma identifier (Colon typeSpec)?)*
    ;

statementBlock
    : LeftBrace statement* RightBrace
    ;

// ======== rule defination ======== //
rules
    : ruleSpec+
    ;

ruleSpec
    : parserRuleSpec
    | lexerRuleSpec 
    ;

parserRuleSpec
    : ParserRuleName Colon ruleBlock Semi
    ;

bindElement
    : LeftBracket identifier RightBracket
    ;

ruleBlock
    : ruleAltList
    ;

ruleAltList
    : actionAlt (AlterOp actionAlt)*
    ;

actionAlt
    : alternative statementBlock?
    ;

alternative
    : element+
    ;

element
    : atomOrGroup (ebnfSuffix bindElement?)?
    ;

atomOrGroup
    : atom
    | group
    ;

atom
    : terminalDef
    | ruleref
    | notSet
    ;

terminalDef
    : LexerRuleName bindElement?
    | StringLiteral
    ;

ruleref
    : ParserRuleName bindElement?
    ;

notSet
    : Tilde setElement
    | Tilde blockSet
    ;

setElement
    : LexerRuleName
    | StringLiteral
    | characterRange
    ;

characterRange
    : StringLiteral Range StringLiteral
    ;

blockSet
    : LeftParen setElement (AlterOp setElement)* RightParen
    ;

ebnfSuffix
    : QuestionMark
    | Star
    | Plus
    ;

group
    : LeftParen altList RightParen bindElement?
    ;

altList
    : alternative (AlterOp alternative)*
    ;

// lexer rule
lexerRuleSpec
    : LexerRuleName Colon lexerRuleBlock Semi
    ;

lexerRuleBlock
    : lexerAltList
    ;

lexerAltList
    : lexerAlt (AlterOp lexerAlt)*
    ;

lexerAlt
    : lexerElements lexerCommands?
    ;

// E.g., channel(HIDDEN), skip, more, mode(INSIDE), push(INSIDE), pop
lexerCommands
    : Arror lexerCommand (Comma lexerCommand)*
    ;

lexerCommand
    : lexerCommandName
    ;

lexerCommandName
    : identifier
    ;

lexerElements
    : lexerElement+
    ;

lexerElement
    : lexerAtomOrGroup ebnfSuffix?
    ;

lexerAtomOrGroup
    : lexerAtom
    | lexerGroup
    ;

lexerAtom
    : characterRange
    | terminalDef
    | notSet
    ;

lexerGroup
    : LeftParen lexerAltList RightParen
    ;

// ======== statement ======== //

statement
    : assignStmt Semi
    | functionCall Semi
    | ifStmt
    | forStmt
    | returnStmt Semi
    ;

assignStmt
    : identifier (Colon typeSpec)? Assign expression
    ;

functionCall
    : funcName LeftParen (expression (Comma expression)*)? RightParen
    ;

ifStmt
    : ifBlock elifBlock* (elseBlock)?
    ;

ifBlock:
    IF LeftParen expression RightParen statementBlock
    ;

elifBlock:
    ELIF LeftParen expression RightParen statementBlock
    ;

elseBlock
    : ELSE statementBlock
    ;

forStmt
    : FOR LeftParen identifier Colon expression RightParen statementBlock
    ;

returnStmt
    : RETURN expression
    ;

// ======== expression ======== //


expression
    : logic_expr (IF expression ELSE expression)*
    ;

logic_expr
    : NOT? member_expr
    | member_expr ((OR | AND) member_expr)*
    ;

member_expr
    : id_expr (IN | (NOT IN) id_expr)?
    ;

id_expr
    : equ_expr ((IS | IS NOT) equ_expr)?
    ;

equ_expr
    : cmp_expr ((Equal | NotEq) cmp_expr)?
    ;

cmp_expr
    : add_expr (
    ( Less
    | LessEq
    | Greater
    | GreaterEq
    ) add_expr)?
    ;

add_expr
    : term_expr ((Plus | Minus) term_expr)*;

term_expr
    : factor_expr ((AT | Star | Div | DivDiv | MOD) factor_expr)*
    ;

factor_expr
    : (Tilde | Plus | Minus)? power_expr
    ;

power_expr
    : prim_expr (StarStar prim_expr)*
    ;

prim_expr
    : paren_surrounded_expr
    | literal
    | variable_access
    | list_expr
    | functionCall
    ;

paren_surrounded_expr
    : LeftParen expression RightParen
    ;

literal
    : BoolLiteral
    | StringLiteral+
    | IntegerLiteral
    | FloatPointLiteral
    | list_literal
    ;

list_literal
    : LeftBracket literal (Comma literal)* RightBracket
    ;

list_expr
    : LeftBracket expression (Comma expression)* RightBracket
    ;

variable_access
    : variable_access (Dot identifier)+
    | variable_access LeftBracket expression (Comma expression)* RightBracket
    | identifier
    ;

// ======== type ======== //

typeSpec
    : builtinType
    ;

builtinType
    : BOOL
    | INT
    | FLOAT
    | LIST
    | STRING
    | MAP
    ;

identifier
    : LexerRuleName
    | ParserRuleName
    ;