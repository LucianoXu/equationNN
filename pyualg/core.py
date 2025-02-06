from __future__ import annotations

from typing import Callable, Literal, Optional
from copy import deepcopy
import random

Property = Literal['Infix']

RESERVED_TOKENS = {'(', ')', '->', 'ID'}


class Signature:
    '''
    It defines the signature of a universal algebra.
    Here we do not use variable arity, because the constant arity is required when generating the term.

    Note that identifiers not included in the symbol_dict are reserved for variables.
    '''
    def __init__(self, symbol_dict: dict[str, tuple[int, set[Property]]]):
        '''
        symbol_dict: a dictionary that maps a symbol to a tuple, consisting of the arity and a set of properties.
        '''
        for symbol in RESERVED_TOKENS:
            if symbol in symbol_dict:
                raise ValueError(f"The symbol '{symbol}' is reserved.")
        
        self.symbol_dict : dict[str, tuple[int, set[Property]]] = symbol_dict

        # check the validity of the signature
        for symbol, (arity, properties) in symbol_dict.items():
            if arity < 0:
                raise ValueError(f"The arity of the symbol '{symbol}' is negative.")
            if 'Infix' in properties and arity != 2:
                raise ValueError(f"The symbol '{symbol}' is infix but not binary.")

    def __str__(self) -> str:
        return str(self.symbol_dict)
    
class Term:
    def __init__(self, head: str, args: tuple[Term, ...] = ()):
        self.head: str = head
        self.args: tuple[Term, ...] = args

    def __eq__(self, other):
        return isinstance(other, Term) and self.head == other.head and self.args == other.args
    
    def __hash__(self):
        return hash((self.head, self.args))
        # return 0 # fix the hash value
    
    def __str__(self):
        # for constants, return the string directly
        if not self.args:
            return self.head
        
        # return the string (data sub1 sub2 ... subn)
        return f"({self.head} {' '.join(map(str, self.args))})"
    
    def sig_str(self, sig: Signature) -> str:
        if self.head in sig.symbol_dict:
            if len(self.args) != sig.symbol_dict[self.head][0]:
                raise ValueError(f"The arity of the symbol '{self.head}' is not equal to the number of arguments.")
            
            if 'Infix' in sig.symbol_dict[self.head][1]:
                return f"({self.args[0].sig_str(sig)} {self.head} {self.args[1].sig_str(sig)})"
            else:
                return f"({self.head} {' '.join(map(lambda x: x.sig_str(sig), self.args))})"
        
        else:
            # if the symbol is not in the signature, it must be a variable
            if len(self.args) != 0:
                raise ValueError(f"The symbol '{self.head}' is not in the signature.")
            
            return self.head
    
    def __repr__(self):
        return f'Term[{self.__str__()}]'
    
    @property
    def is_atom(self) -> bool:
        return not self.args
    
    def is_var(self, sig: Signature) -> bool:
        return self.head not in sig.symbol_dict
    
    def vars(self, sig: Signature) -> set[str]:
        '''
        return the set of variables in the term
        '''
        if self.is_var(sig):
            return {self.head}
        else:
            return set().union(*[arg.vars(sig) for arg in self.args])
    
    def __getitem__(self, pos: tuple[int, ...]) -> Term:
        '''
        getitem accepts a tuple of integers as the position and returns the corresponding subterm
        '''
        if not pos:
            return self
        return self.args[pos[0]][pos[1:]]
        
    def __setitem__(self, pos: tuple[int, ...], value: Term):
        '''
        setitem accepts a tuple of integers as the position and sets the corresponding subterm
        '''
        if not pos:
            self = value
        else:
            self.args[pos[0]][pos[1:]] = value

    def all_nodes(self, pos_prefix : tuple[int, ...] = ()) -> set[tuple[tuple[int, ...], Term]]:
        '''
        Return all nodes in the term. Each node is represented by a tuple of the position and the term.
        '''
        res : set[tuple[tuple[int, ...], Term]] = set([(pos_prefix, self)])
        for i, arg in enumerate(self.args):
            res |= arg.all_nodes(pos_prefix + (i,))
        return res
    
    def get_random_node(self) -> tuple[tuple[int, ...], Term]:
        '''
        Return a random node in the term
        '''
        return random.choice(list(self.all_nodes()))
    
    @property
    def size(self) -> int:
        '''
        Return the size of the term
        '''
        return 1 + sum([arg.size for arg in self.args])
    
    def apply_at(self, opt: TermOpt, sig: Signature, pos: tuple[int, ...] = (), given_subst: Optional[Subst] = None, forbiden_heads: Optional[set[str]] = None) -> Optional[tuple[Term, Subst]]:
        '''
        Apply the function opt to the term at the specified position.
        '''
        if not pos:
            return opt(sig, self, given_subst, forbiden_heads)
        else:
            subst_tree = self.args[pos[0]].apply_at(opt, sig, pos[1:], given_subst)
            if subst_tree is None:
                return None
            return Term(self.head, self.args[:pos[0]] + (subst_tree[0],) + self.args[pos[0] + 1:]), subst_tree[1]
        
VAR_COUNT = 0
def unique() -> Term:
    '''
    Generate a unique variable.
    '''
    global VAR_COUNT
    VAR_COUNT += 1
    return Term(f'${VAR_COUNT}')
        
class Subst:
    '''
    A substitution is a mapping from variables to terms.
    '''
    def __init__(self, data: dict[str, Term]):
        self.data: dict[str, Term] = data

    def __eq__(self, other):
        return isinstance(other, Subst) and self.data == other.data

    def __str__(self):
        if not self.data:
            return "{}"
        
        return "{" + ", ".join([f"{key} : {value}" for key, value in self.data.items()]) + "}"
    
    def sig_str(self, sig: Signature) -> str:
        if not self.data:
            return "{}"
        
        return "{" + ", ".join([f"{key} : {value.sig_str(sig)}" for key, value in self.data.items()]) + "}"
    
    def __getitem__(self, key: str) -> Term:
        return self.data[key]
    
    def partial_subset(self, keys: set[str]) -> Subst:
        '''
        Return the partial substitution with the specified keys
        '''
        return Subst({ key: self.data[key] for key in keys if key in self.data })
    
    def __call__(self, term: Term) -> Term:
        '''
        return the term after applying the substitution recursively
        '''
        if term.is_atom:
            return self.data.get(term.head, term)
        else:
            return Term(term.head, tuple(map(self, term.args)))
  
class MatchingProblem:
    '''
    A matching problem.

    Based on [Term Rewriting and All That] Sec.4.7
    '''
    def __init__(self, sig: Signature, ineqs : list[tuple[Term, Term]]):
        self.ineqs = ineqs
        self.sig = sig

    def __str__(self) -> str:
        
        return "{" + ", ".join([f"{lhs} ≲? {rhs}" for lhs, rhs in self.ineqs]) + "}"

    def solve(self, given_subst: Optional[Subst] = None, forbiden_heads: Optional[set[str]] = None) -> Optional[Subst]:
        return MatchingProblem.solve_matching(self, given_subst, forbiden_heads)

    @staticmethod
    def solve_matching(mp: MatchingProblem, given_subst: Optional[Subst] = None, forbiden_heads: Optional[set[str]] = None) -> Subst | None:
        
        # the set of heads that are not allowed to be assigned to variables
        forbiden_heads = set() if forbiden_heads is None else forbiden_heads

        ineqs = mp.ineqs
        sig = mp.sig

        if given_subst is not None:
            subst = deepcopy(given_subst.data)
        else:
            subst : dict[str, Term] = {}

        while len(ineqs) > 0:
            lhs, rhs = ineqs[0]

            if lhs.is_var(sig):
                if lhs.head in subst:
                    if Subst(subst)(lhs) == rhs:
                        ineqs = ineqs[1:]
                        continue
                    else:
                        return None
                else:
                    if rhs.head in forbiden_heads:
                        return None
                    
                    subst[lhs.head] = rhs
                    ineqs = ineqs[1:]
                    continue

            # being function constructions
            elif rhs.is_var(sig):
                return None
            
            else:
                if lhs.head == rhs.head:
                    ineqs = ineqs[1:]
                    for i in range(len(lhs.args)):
                        ineqs.append((lhs.args[i], rhs.args[i]))
                    continue

                else:
                    return None
                    
        return Subst(subst)


    @staticmethod
    def single_match(sig: Signature, lhs : Term, rhs : Term, given_subst: Optional[Subst] = None, forbiden_heads: Optional[set[str]] = None) -> Subst | None:
        return MatchingProblem(sig, [(lhs, rhs)]).solve(given_subst, forbiden_heads)
    

class RewriteRule:
    def __init__(self, lhs: Term, rhs: Term):
        self.lhs: Term = lhs
        self.rhs: Term = rhs

    def vars(self, sig: Signature) -> set[str]:
        return self.lhs.vars(sig) | self.rhs.vars(sig)
    
    def inst_vars(self, sig: Signature) -> set[str]:
        '''
        return the set of variables that need to be instantiated
        '''
        return self.rhs.vars(sig) - self.lhs.vars(sig)


    def __str__(self):
        return f"{self.lhs} -> {self.rhs}"
    
    def __call__(self, sig: Signature, term: Term, given_subst: Optional[Subst] = None, forbiden_heads: Optional[set[str]] = None) -> Optional[tuple[Term, Subst]]:
        '''
        Apply the rule to the term.
        Result: the rewriten result, or None if the rule does not match the term.
        '''
        matcher = MatchingProblem.single_match(sig, self.lhs, term, given_subst, forbiden_heads)

        if matcher is None:
            return None
    
        inst_vars = self.inst_vars(sig)
        # check whether the inst_vars are provided in given_subst
        if inst_vars and (given_subst is None or not inst_vars.issubset(given_subst.data.keys())):
            return None
        
        return matcher(self.rhs), matcher
        
    def subst(self, subst: Subst) -> RewriteRule:
        return RewriteRule(subst(self.lhs), subst(self.rhs))
    
class TRS:
    '''
    A term rewriting system.
    '''
    def __init__(self, sig: Signature, rules: list[RewriteRule]):
        '''
        A term rewriting system consists of a signature and a list of rewrite rules.
        '''
        self.sig = sig
        self.rules = rules

    def __call__(self, term: Term, alg: Literal["inner_most", "outer_most"] = "inner_most") -> Term:

        # check the variable conincidence
        overlap = term.vars(self.sig).union(*[rule.vars(self.sig) for rule in self.rules])

        if len(overlap) > 0:
            # rename the variables
            subst = Subst({ var: unique() for var in overlap })
            
            compiled_rules = [rule.subst(subst) for rule in self.rules]
        else:
            compiled_rules = self.rules

        current_term = term

        # choose the algorithm
        if alg == "outer_most":
            rewrite = TRS.rewrite_outer_most  
        elif alg == "inner_most":
            rewrite = TRS.rewrite_inner_most

        while True:

            # check whether rewrite rules are applicable
            new_term = rewrite(self.sig, compiled_rules, current_term)

            if new_term is None:
                return current_term
            
            current_term = new_term
            
    

    @staticmethod
    def rewrite_outer_most(sig: Signature, rules: list[RewriteRule], term : Term) -> Term | None:
        '''
        rewrite the term using the rules. Return the result.
        return None when no rewriting is applicable

        algorithm: outer most
        '''

        # try to rewrite the term using the rules
        for rule in rules:
            new_term = rule(sig, term)
            if new_term is not None:
                return new_term[0]
                        
        if not term.is_atom:
            # try to rewrite the subterms
            for i in range(len(term.args)):
                new_subterm = TRS.rewrite_outer_most(sig, rules, term.args[i])
                if new_subterm is not None:
                    return Term(term.head, term.args[:i] +  (new_subterm,) + term.args[i+1:])
                
        return None
    

    @staticmethod
    def rewrite_inner_most(sig: Signature, rules: list[RewriteRule], term : Term) -> Term | None:
        '''
        rewrite the term using the rules. Return the result.
        return None when no rewriting is applicable

        algorithm: outer most
        '''

        if not term.is_atom:
            # try to rewrite the subterms
            for i in range(len(term.args)):
                new_subterm = TRS.rewrite_inner_most(sig, rules, term.args[i])
                if new_subterm is not None:
                    return Term(term.head, term.args[:i] + (new_subterm,) + term.args[i+1:])
                

        # try to rewrite the term using the rules
        for rule in rules:
            new_term = rule(sig, term)
            if new_term is not None:
                return new_term[0]
        
        return None


# signature, term, given_subst, forbiden_heads
TermOpt = Callable[[Signature, Term, Optional[Subst], Optional[set[str]]], Optional[tuple[Term, Subst]]]

__all__ = ['RESERVED_TOKENS', 'Property', 'Signature', 'Term', 'unique', 'Subst', 'MatchingProblem', 'RewriteRule',
           'TRS', 'TermOpt']