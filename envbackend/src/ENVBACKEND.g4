grammar ENVBACKEND;

// Parser rules

// Algebra
alg : sig '[axiom]' axiom+
    ;

// Signature
sig : '[function]' func+ '[variable]' NAME+
    ;

func : funcname ':' INT ;

axiom : '(' NAME ')' expr '=' expr ;

subst : '{' '}'                                         # EmptySubst
      | '{' NAME ':' expr (',' NAME ':' expr)* '}'      # NonEmptySubst
      ;

expr :   funcname                         # Identifier
     |   funcname '(' expr+ ')'           # Application
     ;

funcname    : NAME                                 # NameFunc
            | SYMBOL                               # SymbolFunc
            ;

// Lexer rules
INT :  '0' | [1-9][0-9]* ;
NAME :  [A-Za-z_][A-Za-z0-9_]* ;
SYMBOL :  '+' | '-' | '*' | '&' | '|' | '~' ;

// Single-line comment rule
LINE_COMMENT : '//' ~[\r\n]* -> skip;

// Multi-line comment rule
BLOCK_COMMENT : '/*' .*? '*/' -> skip;

WS :   [ \t\r\n]+ -> skip ;  // Skip whitespace (spaces, tabs, newlines)
