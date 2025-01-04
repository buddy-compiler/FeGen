#!/bin/bash

grammar_dir=./python/FeGen/grammar

grammar_name="FeGen"
lexer_name="${grammar_name}Lexer"
parser_name="${grammar_name}Parser"

cd $grammar_dir

export ANTLR4_TOOLS_ANTLR_VERSION=4.13.0
antlr4 -Dlanguage=Python3 -visitor ${parser_name}.g4 ${lexer_name}.g4
cd -