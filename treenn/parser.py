

from typing import Any
import ply.lex as lex

tokens = [
    'A',
    'B'
]

literals = ['(', ')', '+']

t_A = 'a'
t_B = 'b'


# use // or /* */ to comment
def t_COMMENT(t):
    r'(/\*(.|[\r\n])*?\*/)|(//.*)'
    for c in t.value:
        if c == '\n' or c == '\r\n':
            t.lexer.lineno += 1


# Define a rule so we can track line numbers
def t_newline(t):
    r'[\r\n]+'
    t.lexer.lineno += len(t.value)


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'


def t_error(t):
    raise ValueError(f"Illegal character '{t.value[0]}'")

lexer = lex.lex()

###############################################################

import ply.yacc as yacc
from .formallan import *

precedence = (
)

start = 'term'

def p_term_atom(p):
    '''
    term        : A
                | B
    '''
    p[0] = Leaf(p[1])

def p_term_paren(p):
    '''
    term        : '(' term ')'
    '''
    p[0] = p[2]

def p_term(p):
    '''
    term        : term '+' term
    '''
    p[0] = InfixBinTree('+', p[1], p[3])


# Build the parser
parser = yacc.yacc()

def parse(s: str) -> Any:
    return parser.parse(s, lexer = lexer)
