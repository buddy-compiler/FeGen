#!/bin/bash

do_clean=false

grammar_name="FeGen"
lexer_name="${grammar_name}Lexer"
parser_name="${grammar_name}Parser"

generate_files=(
    "${lexer_name}.interp"
    "${lexer_name}.tokens"
    "${parser_name}.interp"
    "${parser_name}.tokens"
    "${parser_name}.py"
    "${parser_name}Listener.py"
    "${parser_name}Visitor.py"
    )

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        clean)
            do_clean=true
            shift
            echo "clean generated files."
            ;;
        *)
            echo "generate files."
            ;;
    esac
done

if $do_clean;
then
    rm ${generate_files[@]}
else
    export ANTLR4_TOOLS_ANTLR_VERSION=4.13.0
    antlr4 -Dlanguage=Python3 -visitor ${parser_name}.g4 ${lexer_name}.g4
fi





