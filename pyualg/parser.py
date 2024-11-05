# create the parser for the given signature

from .core import Signature, Term, RewriteRule, Subst
import ply.lex as lex, ply.yacc as yacc
import re

# grammar design:
# result : expression | rewriterule | subst
# rewriterule : expression TO expression
# expression : ID | '(' expression ID expression ')' | '(' ID expression* ')'
# subst : '{' '}' | '{' ID ':' expression (',' ID ':' expression)* '}'
# ID : '[a-zA-Z_][a-zA-Z_0-9]*'
# TO : '->'

class Parser:
    def __init__(self, sig: Signature, **kwargs):
        self.sig = sig

        # build the lexer

        # map the literal strings to valid token names
        self.reserved = {}
        for i, symbol in enumerate(sig.symbol_dict.keys()):
            self.reserved[symbol] = f'TOK{i}'


        self.tokens = ['ID', 'TO'] + list(self.reserved.values())
        self.literals = ['(', ')', '{', '}', ':', ',']
        def t_ID(t):
            r'[$a-zA-Z_][a-zA-Z_0-9]*'
            t.type = self.reserved.get(t.value, 'ID')
            return t
        self.t_ID = t_ID
        self.t_TO = r'->'

        for symbol in sig.symbol_dict.keys():
            self.__dict__['t_' + self.reserved[symbol]] = re.escape(symbol)

        self.t_ignore = ' \t'

        # use // or /* */ to comment
        def t_COMMENT(t):
            r'(/\*(.|[\r\n])*?\*/)|(//.*)'
            for c in t.value:
                if c == '\n' or c == '\r\n':
                    t.lexer.lineno += 1
        self.t_COMMENT = t_COMMENT

        def t_newline(t):
            r'[\r\n]+'
            t.lexer.lineno += len(t.value)
        self.t_newline = t_newline

        def t_error(t):
            raise ValueError(f"Illegal character '{t.value[0]}'")
        self.t_error = t_error

        self.lexer = lex.lex(module=self, **kwargs)

        def p_result(p):
            '''
            result  : expression 
                    | rewriterule
                    | subst
            '''
            p[0] = p[1]
        self.p_result = p_result

        # build the parser
        def p_rewriterule(p):
            'rewriterule : expression TO expression'
            p[0] = RewriteRule(p[1], p[3])
        self.p_rewriterule = p_rewriterule

        def p_ID(p):
            'expression : ID'
            p[0] = Term(p[1])
        self.p_ID = p_ID

        def p_subst(p):
            '''
            subst : '{' '}'
                  | '{' subst_list '}'
            '''
            if len(p) == 3:
                p[0] = {}
            else:
                p[0] = Subst(p[2])
        self.p_subst = p_subst

        def p_subst_list(p):
            '''
            subst_list : ID ':' expression
                       | subst_list ',' ID ':' expression
            '''
            if len(p) == 4:
                p[0] = {p[1]: p[3]}
            else:
                p[0] = p[1]
                p[0][p[3]] = p[5]
                
        self.p_subst_list = p_subst_list

        # Helper function to capture symbol, arity, and properties
        def create_production_function(symbol, arity, is_infix):
            if arity == 0:
                def f(p):
                    p[0] = Term(symbol)
                f.__doc__ = f'''
                    expression : {self.reserved[symbol]}
                    '''
            elif is_infix:
                def f(p):
                    p[0] = Term(symbol, (p[2], p[4]))
                f.__doc__ = f'''
                    expression : '(' expression {self.reserved[symbol]} expression ')'
                    '''
            else:
                def f(p):
                    p[0] = Term(symbol, tuple(p[3:3 + arity]))
                f.__doc__ = f'''
                    expression : '(' {self.reserved[symbol]} {"expression "*arity} ')'
                    '''
            return f

        for i, symbol in enumerate(sig.symbol_dict.keys()):
            arity, properties = sig.symbol_dict[symbol]

            self.__dict__['p_' + str(i)] = create_production_function(symbol, arity, 'Infix' in properties)

        def p_error(p):
            if p:
                raise ValueError(f"Syntax error at '{p.value}'")
            else:
                raise ValueError("Syntax error at EOF")
            
        self.p_error = p_error

        self.start = 'result'
        self.parser = yacc.yacc(module=self, **kwargs)

    def parse_term(self, input_string: str) -> Term:
        res = self.parser.parse(input_string, lexer = self.lexer)
        if not isinstance(res, Term):
            raise ValueError(f"Parsing failed for the input string '{input_string}' as a term.")

        return res
    
    def parse_rewriterule(self, input_string: str) -> RewriteRule:
        res = self.parser.parse(input_string, lexer = self.lexer)
        if not isinstance(res, RewriteRule):
            raise ValueError(f"Parsing failed for the input string '{input_string}' as a rewrite rule.")

        return res
    
    def parse_subst(self, input_string: str) -> Subst:
        res = self.parser.parse(input_string, lexer = self.lexer)
        if not isinstance(res, Subst):
            raise ValueError(f"Parsing failed for the input string '{input_string}' as a substitution.")

        return res