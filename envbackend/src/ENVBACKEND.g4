grammar ENVBACKEND;

// Parser rules

////////////////////////////////////////////////
// RL langauge
proofstep : proofstate proofaction;

// The state grammar
proofstate : START_STATE equation END_STATE;

// The action grammar
proofaction : START_ACTION NAME pos subst END_ACTION    # RuleAction
            | START_ACTION 'SUBST' NAME expr END_ACTION # SubstAction
            ;

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

START_STATE : '<STT>';
END_STATE : '</STT>';
START_ACTION : '<ACT>';
END_ACTION : '</ACT>';

INT :  '0' | [1-9][0-9]* ;
NAME :  [A-Za-z_][A-Za-z0-9_]* ;
SYMBOL :  '+' | '-' | '*' | '&' | '|' | '~' ;

// Single-line comment rule
LINE_COMMENT : '//' ~[\r\n]* -> skip;

// Multi-line comment rule
BLOCK_COMMENT : '/*' .*? '*/' -> skip;

WS :   [ \t\r\n]+ -> skip ;  // Skip whitespace (spaces, tabs, newlines)
