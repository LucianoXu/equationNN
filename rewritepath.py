from __future__ import annotations
from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst

class RewritePath:
    '''
    The class to represent a path of rewriting operations
    A path is a sequence of tuples (term, operation, position, substitution, required_instantiation)
    '''
    def __init__(self, sig: Signature, path: tuple[tuple[Term, TermOpt, tuple[int, ...], Subst, Subst], ...], current : Term):
        self.sig = sig
        self.path = list(path)
        self.current: Term = current

    @property
    def start(self) -> Term:
        '''
        return the current tree
        '''
        if not self.path:
            return self.current
        else:
            return self.path[0][0]


    def get_inverse(self, inverse_table: dict[TermOpt, TermOpt], inst_vars: dict[TermOpt, set[str]]) -> RewritePath:
        '''
        return the inverse path of the current path
        '''
        if not self.path:
            return RewritePath(self.sig, (), self.start)
        
        new_current = self.path[0][0]
        new_path = []

        _, opt, pos, subst, _ = self.path[-1]
        inv_opt = inverse_table[opt]
        required_instantiation = subst.partial_subset(inst_vars[inv_opt])
        new_path.append((self.current, inv_opt, pos, subst, required_instantiation))

        for i in range(len(self.path) - 2, -1, -1):
            _, opt, pos, subst, _ = self.path[i]
            inv_opt = inverse_table[opt]
            required_instantiation = subst.partial_subset(inst_vars[inv_opt])
            new_path.append((self.path[i+1][0], inv_opt, pos, subst, required_instantiation))
        
        return RewritePath(self.sig, tuple(new_path), new_current)

    
    def apply(self, opt: TermOpt, pos: tuple[int, ...], subst: Subst, inst_vars : dict[TermOpt, set[str]], forbiden_heads: Optional[set[str]] = None) -> bool:
        '''
        apply the operation at the position, and update the path
        return: whether the application is successful
        '''
        apply_res = self.current.apply_at(opt, self.sig, pos, subst, forbiden_heads)
        if apply_res is None:
            return False
        
        new_current, subst_res = apply_res
        required_instantiation = subst.partial_subset(inst_vars[opt])
        
        self.path.append((self.current, opt, pos, subst_res, required_instantiation))
        self.current = new_current
        return True

    def verify(self, verify_opt: Optional[set[TermOpt]], forbiden_heads: Optional[set[str]] = None):
        '''
        Verify the specified options in the path.
        '''
        if not self.path:
            return
        

        for i in range(len(self.path)):
            pre, opt, pos, subst, _ = self.path[i]
            post = self.path[i+1][0] if i < len(self.path) - 1 else self.current

            if verify_opt is None or opt in verify_opt:
                apply_res = pre.apply_at(opt, self.sig, pos, subst, forbiden_heads)
                if apply_res is None or apply_res[0] != post:
                    raise ValueError(f"Verification failed at position {i}: {pre} -- {opt} at {pos} --> {post}")
            
            pre = post

    def __str__(self):
        '''
        Print out the terms and the operations line by line
        '''
        res = []
        for term, opt, pos, _, subst in self.path:
            res.append(term.sig_str(self.sig))
            res.append(f" -- {opt} at {pos} with {subst} -->")
        res.append(self.current.sig_str(self.sig))
        return '\n'.join(res)
    
    