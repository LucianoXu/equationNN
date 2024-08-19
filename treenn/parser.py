

from typing import Any
import ply.lex as lex

reserved = {
    'a' : 'A',
    'b' : 'B',
    'True': 'TRUE'
}


tokens = [
    'ID',
    ] + list(reserved.values())


literals = ['(', ')', '+', '=']

t_A = 'a'
t_B = 'b'
t_TRUE = 'True'


def t_ID(t):
    r'[$a-zA-Z\_][a-zA-Z0-9\_]*'
    t.type = reserved.get(t.value, 'ID')
    return t

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
                | TRUE
    '''
    p[0] = Leaf(p[1])

def p_term_var(p):
    '''
    term    : ID
    '''
    p[0] = Var(p[1])

def p_term_paren(p):
    '''
    term        : '(' term ')'
    '''
    p[0] = p[2]

def p_term_plus(p):
    '''
    term        : term '+' term
    '''
    p[0] = InfixBinTree('+', p[1], p[3])

def p_term_eq(p):
    '''
    term        : term '=' term
    '''
    p[0] = InfixBinTree('=', p[1], p[3])


# Build the parser
parser = yacc.yacc()

def parse(s: str) -> Tree:
    res = parser.parse(s, lexer = lexer)
    if not res:
        raise ValueError(f"Parsing failed for the input string '{s}'")

    return res
