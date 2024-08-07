
from treenn import *

def test_tree_str():
    leaf = Leaf("leaf")
    assert str(leaf) == "leaf"

    left = Leaf("left")
    right = Leaf("right")
    tree = PrefixBinTree("root", left, right)
    assert str(tree) == "(root left right)"

    left = Leaf("left")
    right = Leaf("right")
    tree = InfixBinTree("root", left, right)
    assert str(tree) == "(left root right)"

def test_parser():
    assert parse("a") == Leaf("a")
    assert parse("b") == Leaf("b")
    assert parse("(a+b)") == InfixBinTree("+", Leaf("a"), Leaf("b"))
    assert parse("(a+(a+b))") == InfixBinTree("+", Leaf("a"), InfixBinTree("+", Leaf("a"), Leaf("b")))
    assert parse("((a+b)+a)") == InfixBinTree("+", InfixBinTree("+", Leaf("a"), Leaf("b")), Leaf("a"))