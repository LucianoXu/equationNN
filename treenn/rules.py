from .formallan import *

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