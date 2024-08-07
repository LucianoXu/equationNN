
from __future__ import annotations
import random
from typing import Callable, Optional

class Tree:
    def __init__(self, data: str, sub: tuple[Tree, ...]):
        self.data = data
        self.sub = sub

    def __eq__(self, other):
        return isinstance(other, Tree) and self.data == other.data and self.sub == other.sub
    
    def __hash__(self):
        return hash((self.data, self.sub))
        # return 0 # fix the hash value
    
    def __str__(self):
        # return the string (data sub1 sub2 ... subn)
        return f"({self.data} {' '.join(map(str, self.sub))})"
    
    @property
    def is_atom(self) -> bool:
        return not self.sub
    
    # getitem accepts a tuple of integers as the position and returns the corresponding sub-tree
    def __getitem__(self, pos: int|tuple[int, ...]) -> Tree:
        if isinstance(pos, int):
            return self.sub[pos]
        else:
            if not pos:
                return self
            return self.sub[pos[0]][pos[1:]]
        
    def apply(self, opt: TreeOpt, pos: tuple[int, ...] = ()) -> Optional[Tree]:
        '''
        try to apply the operation at the position, and return the substituted new tree
        If the operation fails, return None
        '''
        if not pos:
            return opt(self)
        
        subst_tree = self.sub[pos[0]].apply(opt, pos[1:])
        if subst_tree is None:
            return None
        
        new_sub = self.sub[:pos[0]] + (subst_tree,) + self.sub[pos[0] + 1:]
        return Tree(self.data, new_sub)
    
    def all_nodes(self, pos_prefix : tuple[int, ...] = ()) -> set[tuple[tuple[int, ...], Tree]]:
        '''
        return all nodes in the tree
        '''
        res : set[tuple[tuple[int, ...], Tree]] = set([(pos_prefix, self)])
        for i, sub in enumerate(self.sub):
            res |= sub.all_nodes(pos_prefix + (i,))
        return res
    
    def get_random_node(self) -> tuple[tuple[int, ...], Tree]:
        '''
        return a random node in the tree
        '''
        return random.choice(list(self.all_nodes()))
        
# a tree operation accepts a tree as input and return the replacement result
# If the operation fails, return None.
TreeOpt = Callable[[Tree], Optional[Tree]]

class Leaf(Tree):
    def __init__(self, data: str):
        super().__init__(data, ())

    def __str__(self):
        return self.data
    
class Var(Leaf):
    '''
    The tree node representing variables
    '''
    count = 0

    def __init__(self, data: str):
        super().__init__(data)

    @staticmethod
    def unique() -> Var:
        '''
        return a unique variable
        '''
        Var.count += 1
        return Var(f"${Var.count}")
        

class InfixBinTree(Tree):
    def __init__(self, data: str, left: Tree, right: Tree):
        super().__init__(data, (left, right))

    def __str__(self):
        return f"({self.sub[0]} {self.data} {self.sub[1]})"
    
    def apply(self, opt: TreeOpt, pos: tuple[int, ...] = ()) -> Tree | None:
        if not pos:
            return opt(self)
        
        subst_tree = self.sub[pos[0]].apply(opt, pos[1:])
        if subst_tree is None:
            return None
        
        new_sub = self.sub[:pos[0]] + (subst_tree,) + self.sub[pos[0] + 1:]
        return InfixBinTree(self.data, *new_sub)