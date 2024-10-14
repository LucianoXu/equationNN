from __future__ import annotations
from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt

class RewritePath:
    '''
    The class to represent a path of rewriting operations
    '''
    def __init__(self, sig: Signature, path: tuple[tuple[Term, TermOpt, tuple[int, ...]], ...], current : Term):
        self.sig = sig
        self.path = list(path)
        self.current = current

    @property
    def start(self) -> Term:
        '''
        return the current tree
        '''
        if not self.path:
            return self.current
        else:
            return self.path[0][0]


    def get_inverse(self, inverse_table: dict[TermOpt, TermOpt]) -> RewritePath:
        '''
        return the inverse path of the current path
        '''
        if not self.path:
            return RewritePath(self.sig, (), self.start)
        
        new_current = self.path[0][0]
        new_path = []

        _, opt, pos = self.path[-1]
        new_path.append((self.current, inverse_table[opt], pos))

        for i in range(len(self.path) - 2, -1, -1):
            _, opt, pos = self.path[i]
            new_path.append((self.path[i+1][0], inverse_table[opt], pos))
        
        return RewritePath(self.sig, tuple(new_path), new_current)

    
    def apply(self, opt: TermOpt, pos: tuple[int, ...]) -> bool:
        '''
        apply the operation at the position, and update the path
        return: whether the application is successful
        '''
        new_current = self.current.apply_at(opt, self.sig, pos)
        if new_current is None:
            return False
        
        self.path.append((self.current, opt, pos))
        self.current = new_current
        return True

    def verify(self, verify_opt: Optional[set[TermOpt]]):
        '''
        Verify the specified options in the path.
        '''
        if not self.path:
            return
        

        for i in range(len(self.path)):
            pre, opt, pos = self.path[i]
            post = self.path[i+1][0] if i < len(self.path) - 1 else self.current

            if verify_opt is None or opt in verify_opt:
                if pre.apply_at(opt, self.sig, pos) != post:
                    raise ValueError(f"Verification failed at position {i}: {pre} -- {opt} at {pos} --> {post}")
            
            pre = post

    def __str__(self):
        '''
        Print out the terms and the operations line by line
        '''
        res = []
        for term, opt, pos in self.path:
            res.append(term.sig_str(self.sig))
            res.append(f" -- {opt} at {pos} -->")
        res.append(self.current.sig_str(self.sig))
        return '\n'.join(res)
    
    