from typing import TextIO
from antlr4 import InputStream, Lexer, Token
from antlr4.Token import CommonToken
import sys
import re

class MojoLexerBase(Lexer):
    def __init__(self, input: InputStream, output: TextIO = sys.stdout):
        super().__init__(input, output)

        self.tokens: list[CommonToken] = []
        self.indent_len = 0
        self.in_multi_line = 0
        
    def getNext(self) -> CommonToken:
        return super().nextToken() if len(self.tokens) == 0 else self.tokens.pop(0)
    
    def nextToken(self):
        tk = self.getNext()
        tk = self.handle(tk)
        # print(tk)
        return tk
        
        
        
        if len(self.tokens) == 0:
            
            if tk.type in (self.LeftParen, self.LeftBracket): # handle LeftParen and LeftBracket, enter multi line
                self.in_multi_line += 1
                self.tokens.append(tk)
            elif tk.type in (self.RightParen, self.RightBracket): # handle RightParen and RightBracket, exit multi line
                self.in_multi_line -= 1
                self.tokens.append(tk)
            elif tk.type == self.NEWLINE: # NEWLINE
                if self.in_multi_line > 0: # handle NEWLINE in mulit line 
                    while tk.type in (self.NEWLINE, self.WS): # ignore NEWLINE and WS 
                        tk = super().nextToken()
                    self.tokens.append(tk)
                else: # indent and dedent
                    self.handle_newline(tk)
            elif tk.type == Token.EOF: # EOF
                self.handle_eof(tk)
            else:
                self.tokens.append(tk) # otherwise
        else:
            print(self.tokens[0])
            return self.tokens.pop(0)

    def handle(self, tk):
        tk = self.handle_multi_line(tk)
        tk = self.handle_indent_dedent(tk)
        tk = self.handle_eof(tk)
        return tk
    
    def handle_multi_line(self, tk: CommonToken):
        if tk.type in (self.LeftParen, self.LeftBracket): # handle LeftParen and LeftBracket, enter multi line
            self.in_multi_line += 1
            return tk
        elif tk.type in (self.RightParen, self.RightBracket): # handle RightParen and RightBracket, exit multi line
            self.in_multi_line -= 1
            return tk
        elif tk.type == self.NEWLINE and self.in_multi_line > 0: # NEWLINE
            return self.handle(self.getNext())
        else:
            return tk
    def handle_indent_dedent(self, tk):
        if tk.type == self.NEWLINE and self.in_multi_line == 0:
            # get length of tab
            tab_len = 0
            space_len = 0
            next_tk: CommonToken = self.getNext()
            while next_tk.type in (self.NEWLINE, self.WS):
                if next_tk.type == self.NEWLINE:
                    space_len = 0
                else:
                    space_len += 1
                next_tk: CommonToken = self.getNext()
                    
            tab_len = space_len // 4
            if (space_len % 4) != 0:
                self.getErrorListenerDispatch().syntaxError(self, tk, tk.line, tk.column, "wrong space.", None)
                
            # compare tab_len and self.indent_len and emit INDENT and DEDENT
            if tab_len == self.indent_len: # no indent or dedent
                self.tokens.append(next_tk)
            elif tab_len < self.indent_len: # dedent
                for _ in range(self.indent_len - tab_len):
                    dedent_token = self.create_token(tk, self.DEDENT, Token.DEFAULT_CHANNEL, tk.start - 1, "<DEDENT>")
                    self.tokens.append(dedent_token)
                self.tokens.append(next_tk)
                self.indent_len = tab_len
            elif tab_len == (self.indent_len + 1): # indent
                indent_token = self.create_token(tk, self.INDENT, Token.DEFAULT_CHANNEL, tk.start - 1, "<INDENT>")
                self.tokens.append(indent_token)
                self.tokens.append(next_tk)
                self.indent_len += 1
            else: # error indent
                self.getErrorListenerDispatch().syntaxError(self, tk, tk.line, tk.column, "wrong indent length.", None)
        return tk
                
    def handle_eof(self, tk: CommonToken):
        if tk.type == Token.EOF:
            for _ in range(self.indent_len):
                dedent_token = self.create_token(tk, self.DEDENT, Token.DEFAULT_CHANNEL, tk.start - 1, "<DEDENT>")
                self.tokens.append(dedent_token)
            self.indent_len = 0
            self.tokens.append(tk)
            return self.getNext()
        return tk
        

    def create_token(self, base_token: CommonToken, type, channel, stop, text) -> CommonToken:
        tk = base_token.clone()
        tk.type = type
        tk.channel = channel
        tk.stop = stop
        tk.text = text
        return tk