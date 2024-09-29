parser grammar MojoParser;
options {
    tokenVocab=MojoLexer;
}

module
    : (function_defination
    | struct_defination
    | alias_stmt
    | import_stmt
    | NEWLINE)+ EOF
    ;

function_defination
    : decorators? proto_type Colon block
    ;

decorators
    : (AT Identifier NEWLINE)+
    ;

proto_type
    : (FN | DEF) Identifier template_decl? LeftParen func_params? RightParen RAISES? (RightArror type)?
    ;

template_decl
    : LeftBracket (template_var_decl (Comma template_var_decl)* Comma?)? RightBracket
    ;

template_var_decl
    : Identifier Colon type (Assign expression)?
    ;

func_params
    : func_param (Comma func_param)* Comma?
    ;

func_param
    : common_func_param
    | init_param
    | variadic_param
    ;

init_param
    : common_func_param Assign expression
    ;

common_func_param
    : INOUT? Identifier (Colon type)?
    ;

variadic_param
    : Star Identifier (Colon type)?
    ;

block
    : NEWLINE INDENT statements DEDENT
    ;

// struct defination
struct_defination
    : STRUCT struct_name template_decl Colon struct_block
    ;

struct_name
    : Identifier
    ;

struct_block
    : NEWLINE INDENT (field_decl | function_defination | NEWLINE)* DEDENT 
    ;

field_decl
    : VAR Identifier Colon type
    ;

type
    : builtin_type
    | custom_type
    | function_type
    ;

builtin_type
    : STRING
    | INT
    | INT8
    | INT16
    | INT32
    | INT64
    | UINT8
    | UINT16
    | UINT32
    | UINT64
    | FLOAT16
    | FLOAT32
    | FLOAT64
    | SIMD LeftBracket variable_access Comma expression RightBracket
    | LIST LeftBracket type RightBracket
    | DICT LeftBracket type Comma type RightBracket
    | OPTIONAL LeftBracket type RightBracket
    | ANYTYPE
    | ANY_TRIVIAL_REG_TYPE
    | NONE
    ;

custom_type
    : Identifier template_spec?
    ;

function_type
    : FN template_decl? LeftParen (INOUT? type (Comma INOUT? type)*)? RightParen RightArror type
    | FN template_decl? LeftParen (INOUT? Identifier Colon type (Comma INOUT? Identifier Colon type)*)? RightParen RightArror type
    ;

statements
    : statement+
    ;

statement
    : function_defination
    | if_else_stmt
    | while_stmt
    | for_stmt
    | return_stmt NEWLINE
    | method_invoke NEWLINE
    | var_assign NEWLINE
    | import_stmt NEWLINE
    | alias_stmt NEWLINE
    | raise_stmt NEWLINE
    | PASS NEWLINE
    | CONTINUE NEWLINE
    | BREAK NEWLINE
    | NEWLINE
    ;

function_call
    : call_function_name template_spec? LeftParen func_call_params? RightParen
    ;

call_function_name
    : builtin_type // exactly conversion
    | Identifier
    ;

template_spec
    : LeftBracket template_param (Comma template_param)* RightBracket
    ;

template_param
    : expression
    | type
    | kwargs
    ;

func_call_params
    : expression (Comma expression)* (Comma kwargs)* Comma?
    | kwargs (Comma kwargs)* Comma?
    ;

kwargs
    : Identifier Assign expression
    | StarStar expression
    ;

return_stmt
    : RETURN expression
    ;

var_assign
    : VAR Identifier Colon type (Assign expression)?
    | (VAR)? variable_access 
    ( Assign 
    | ModAssign
    | DivAssign
    | DivDivAssign
    | MinusAssign
    | PlusAssign
    | StarAssign
    | StarStarAssign ) expression
    ;


if_else_stmt
    : decorators? if_stmt elif_stmt*  else_stmt?
    ;

if_stmt
    : IF expression Colon block
    ;

elif_stmt
    : ELIF expression Colon block
    ;

else_stmt
    : ELSE Colon block
    ;

while_stmt
    : WHILE expression Colon block else_stmt?
    ;

for_stmt
    : decorators? FOR Identifier (Comma Identifier)* IN expression Colon block
    ;

import_stmt
    : import_name
    | import_from
    ;

import_name
    : IMPORT dotted_as_names
    ;

dotted_as_names
    : dotted_as_name (',' dotted_as_name)*
    ;

dotted_as_name
    : dotted_name ('as' Identifier)?
    ;

dotted_name
    : dotted_name '.' Identifier
    | Identifier
    ;

import_from
    : FROM (Dot | TriDot)* dotted_name IMPORT import_from_targets
    | FROM (Dot | TriDot)+ IMPORT import_from_targets
    ;

import_from_targets
    : LeftParen import_from_as_names Comma? RightParen
    | import_from_as_names
    | Star
    ;

import_from_as_names
    : import_from_as_name (Comma import_from_as_name)*
    ;

import_from_as_name
    : Identifier (AS Identifier)?
    ;

alias_stmt
    : ALIAS Identifier (Colon type)? Assign expression
    ;

raise_stmt
    : RAISE expression
    ;

method_invoke
    : ((Identifier template_spec? | function_call)  Dot)* function_call
    | String_literal (Dot function_call)+
    ;

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
    : bitwise_or_expr (
    ( Less
    | LessEq
    | Greater
    | GreaterEq
    ) bitwise_or_expr)?
    ;

bitwise_or_expr
    : bitwise_xor_expr (BITWISE_OR bitwise_xor_expr)*
    ;

bitwise_xor_expr
    : bitwise_and_expr (BITWISE_XOR bitwise_and_expr)*
    ;

bitwise_and_expr
    : shift_expr (BITWISE_AND shift_expr)*
    ;

shift_expr
    : add_expr ((LEFT_SHIFT | RIGHT_SHIFT) add_expr)*
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
    | method_invoke
    ;

paren_surrounded_expr
    : LeftParen expression RightParen
    ;

literal
    : Bool_literal
    | String_literal+
    | Integer_literal
    | FloatPoint_literal
    | list_literal
    ;

list_literal
    : LeftBracket expression (Comma expression)* RightBracket
    ;

variable_access
    : variable_access (Dot Identifier)+
    | variable_access LeftBracket expression (Comma expression)* RightBracket
    | Identifier
    ;
