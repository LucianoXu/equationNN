from __future__ import annotations
from .formallan import *



class RewritePath:
    '''
    The class to represent a path of rewriting operations
    '''
    def __init__(self, path: tuple[tuple[Tree, TreeOpt, tuple[int, ...]], ...], current : Tree):
        self.path = list(path)
        self.current = current

    @property
    def start(self) -> Tree:
        '''
        return the current tree
        '''
        if self.path == ():
            return self.current
        else:
            return self.path[0][0]

    def get_inverse(self, inverse_table: dict[TreeOpt, TreeOpt]) -> RewritePath:
        '''
        return the inverse path of the current path
        '''
        if not self.path:
            return RewritePath((), self.start)
        
        new_current = self.path[0][0]
        new_path = []

        _, opt, pos = self.path[-1]
        new_path.append((self.current, inverse_table[opt], pos))

        for i in range(len(self.path) - 2, -1, -1):
            _, opt, pos = self.path[i]
            new_path.append((self.path[i+1][0], inverse_table[opt], pos))
        
        return RewritePath(tuple(new_path), new_current)
    
    def apply(self, opt: TreeOpt, pos: tuple[int, ...]) -> bool:
        '''
        apply the operation at the position, and update the path
        return: whether the application is successful
        '''
        new_current = self.current.apply(opt, pos)
        if new_current is None:
            return False
        
        self.path.append((self.current, opt, pos))
        self.current = new_current
        return True

    def verify(self, verify_opt: Optional[set[TreeOpt]]):
        '''
        Verify the specified options in the path.
        '''
        if not self.path:
            return
        

        for i in range(len(self.path)):
            pre, opt, pos = self.path[i]
            post = self.path[i+1][0] if i < len(self.path) - 1 else self.current

            if verify_opt is None or opt in verify_opt:
                if pre.apply(opt, pos) != post:
                    raise ValueError(f"Verification failed at position {i}: {pre} -- {opt} at {pos} --> {post}")
            
            pre = post

    def __str__(self):
        '''
        Print out the terms and the operations line by line
        '''
        res = []
        for term, opt, pos in self.path:
            res.append(str(term))
            res.append(f" -- {opt} at {pos} -->")
        res.append(str(self.current))
        return '\n'.join(res)
    
    




def rule_comm(term: Tree) -> Optional[Tree]:
    '''
    x + y -> y + x
    '''
    if term.is_atom:
        return None
    if term.data == '+':
        return InfixBinTree('+', term.sub[1], term.sub[0])
    return None

def rule_assoc1(term: Tree) -> Optional[Tree]:
    '''
    (x + y) + z -> x + (y + z)
    '''
    if term.is_atom:
        return None
    if term.data == '+':
        if term.sub[0].data == '+':
            return InfixBinTree('+', term.sub[0].sub[0], InfixBinTree('+', term.sub[0].sub[1], term.sub[1]))
    return None

def rule_assoc2(term: Tree) -> Optional[Tree]:
    '''
    x + (y + z) -> (x + y) + z
    '''
    if term.is_atom:
        return None
    if term.data == '+':
        if term.sub[1].data == '+':
            return InfixBinTree('+', InfixBinTree('+', term.sub[0], term.sub[1].sub[0]), term.sub[1].sub[1])
    return None

def rule_eq_reduce(term: Tree) -> Optional[Tree]:
    '''
    x = x -> True
    '''
    if term.is_atom:
        return None
    
    if term.data == '=':
        if term.sub[0] == term.sub[1]:
            return Leaf('True')
    
    return None

def rule_eq_expand(term: Tree) -> Optional[Tree]:
    '''
    True -> x = x
    '''
    if term == Leaf('True'):
        newvar = Var.unique()
        return InfixBinTree('=', newvar, newvar)
    
    return None

inverse_table = {
    rule_comm : rule_comm,
    rule_assoc1 : rule_assoc2,
    rule_assoc2 : rule_assoc1,
    rule_eq_expand : rule_eq_reduce
}

opt_tokenizer = {
    rule_comm: 0,
    rule_assoc1: 1,
    rule_assoc2: 2,
    rule_eq_reduce: 3
}