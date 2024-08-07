

class Tree:
    ...

class Leaf(Tree):
    def __init__(self, data: str):
        self.data = data

    def __str__(self):
        return self.data
    
    def __eq__(self, other):
        return isinstance(other, Leaf) and self.data == other.data

class PrefixBinTree(Tree):
    def __init__(self, data: str, left: Tree, right: Tree):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        return f"({self.data} {self.left} {self.right})"
    
    def __eq__(self, other):
        return isinstance(other, PrefixBinTree) and self.data == other.data and self.left == other.left and self.right == other.right
    
class InfixBinTree(Tree):
    def __init__(self, data: str, left: Tree, right: Tree):
        self.left = left
        self.data = data
        self.right = right

    def __str__(self):
        return f"({self.left} {self.data} {self.right})"
    
    def __eq__(self, other):
        return isinstance(other, InfixBinTree) and self.data == other.data and self.left == other.left and self.right == other.right