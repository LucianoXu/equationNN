from __future__ import annotations
from .formallan import *



class RewritePath:
    '''
    The class to represent a path of rewriting operations
    '''
    def __init__(self, start : Tree, path: tuple[tuple[TreeOpt, tuple[int, ...], Tree], ...] = ()):
        self.start = start
        self.path = list(path)

    @property
    def current(self) -> Tree:
        '''
        return the current tree
        '''
        if self.path == ():
            return self.start
        else:
            return self.path[-1][2]

    def get_inverse(self, inverse_table: dict[TreeOpt, TreeOpt]) -> RewritePath:
        '''
        return the inverse path of the current path
        '''
        if not self.path:
            return RewritePath(self.start, ())
        
        new_start = self.path[-1][2]
        new_path = []
        for i in range(len(self.path) - 1, 0, -1):
            opt, pos, _ = self.path[i]
            new_path.append((inverse_table[opt], pos, self.path[i - 1][2]))

        opt, pos, _ = self.path[0]
        new_path.append((inverse_table[opt], pos, self.start))
        
        return RewritePath(new_start, tuple(new_path))
    
    def apply(self, opt: TreeOpt, pos: tuple[int, ...]) -> bool:
        '''
        apply the operation at the position, and update the path
        return: whether the application is successful
        '''
        new_tree = self.current.apply(opt, pos)
        if new_tree is None:
            return False
        
        self.path.append((opt, pos, new_tree))
        return True

    def verify(self, verify_opt: Optional[set[TreeOpt]]):
        '''
        Verify the specified options in the path.
        '''
        if not self.path:
            return
        
        pre = self.start

        for i in range(len(self.path) - 1):
            opt, pos, post = self.path[i]

            if verify_opt is None or opt in verify_opt:
                if pre.apply(opt, pos) != post:
                    raise ValueError(f"Verification failed at position {i}: {pre} -- {opt} at {pos} --> {post}")
            
            pre = post

    def __str__(self):
        '''
        Print out the terms and the operations line by line
        '''
        res = []
        res.append(str(self.start))
        for opt, pos, term in self.path:
            res.append(f" -- {opt} at {pos} -->")
            res.append(str(term))
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