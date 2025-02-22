grammar ENVBACKEND;

// Parser rules

////////////////////////////////////////////////
// RL langauge
proofstep : equation ':' proofaction;

// The action grammar
proofaction : NAME pos subst;

////////////////////////////////////////////////
// Algebra
alg : sig '[axiom]' axiom+
    ;

// Signature
sig : '[function]' func+ '[variable]' NAME+
    ;

func : funcname ':' INT ;

axiom : '(' NAME ')' equation ;

equation : expr '=' expr ;

subst : '{' '}'                                         # EmptySubst
      | '{' NAME ':' expr (',' NAME ':' expr)* '}'      # NonEmptySubst
      ;

expr :   funcname                         # Identifier
     |   funcname '(' expr+ ')'           # Application
     ;

funcname    : NAME                                 # NameFunc
            | SYMBOL                               # SymbolFunc
            ;

// The grammar for positions
pos : '(' INT* ')' ;

//////////////////////////////////////////////

// Lexer rules
INT :  '0' | [1-9][0-9]* ;
NAME :  [A-Za-z_][A-Za-z0-9_]* ;
SYMBOL :  '+' | '-' | '*' | '&' | '|' | '~' ;

// Single-line comment rule
LINE_COMMENT : '//' ~[\r\n]* -> skip;

// Multi-line comment rule
BLOCK_COMMENT : '/*' .*? '*/' -> skip;

WS :   [ \t\r\n]+ -> skip ;  // Skip whitespace (spaces, tabs, newlines)
