
from treenn import *

def test_tree_str():
    leaf = Leaf("leaf")
    assert str(leaf) == "leaf"

    left = Leaf("left")
    right = Leaf("right")
    tree = Tree("root", (left, right, right))
    assert str(tree) == "(root left right right)"

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

def test_tree_getitem():
    term = parse("((a+b)+a)")
    assert term[()] == term
    assert term[1] == parse('a')
    assert term[0, 1] == parse('b')

def test_tree_apply():
    term = parse("(a+b)+(b+a)")
    assert term.apply(rule_comm) == parse("(b+a)+(a+b)")
    assert term.apply(rule_comm, (0,)) == parse("(b+a)+(b+a)")
    assert term.apply(rule_assoc1, (0,)) == None
    assert term.apply(rule_assoc1) == parse("a+(b+(b+a))")
