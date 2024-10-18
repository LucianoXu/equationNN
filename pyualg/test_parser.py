from .core import *
from .parser import *

def test_parser():
    sig = Signature(
        {
            'c': (0, set()),
            'f': (2, set()),
            'g': (1, set()),
            '+' : (2, {'Infix'}),
        }
    )

    parser = Parser(sig)
    assert str(parser.parse_term('c')) == 'c'
    assert str(parser.parse_term('((f (g x) y) + z)')) == '(+ (f (g x) y) z)'
    assert parser.parse_term('((f (g x) y) + z)').sig_str(sig) == '((f (g x) y) + z)'

def test_variable():
    sig = Signature(
        {
            'c': (0, set()),
            'f': (2, set()),
            'g': (1, set()),
            '+' : (2, {'Infix'}),
        }
    )

    parser = Parser(sig)
    assert parser.parse_term('c').vars(sig) == set()
    assert parser.parse_term('((f (g x) y) + z)').vars(sig) == {'x', 'y', 'z'}


def test_matcher():
    sig = Signature(
        {
            'c': (0, set()),
            'f': (2, set()),
            'g': (1, set()),
            '+' : (2, {'Infix'}),
        }
    )
    parser = Parser(sig)

    term = parser.parse_term('((f (g x) y) + y)')
    pattern = parser.parse_term('((f x m) + m)')
    matcher = MatchingProblem.single_match(sig, pattern, term)
    assert matcher == Subst({'m': Term('y'), 'x': Term('g', (Term('x'),))})
    
    term = parser.parse_term('(g (x + y))')
    pattern = parser.parse_term('(g ($1 + $1))')
    matcher = MatchingProblem.single_match(sig, pattern, term)
    assert matcher == None

def test_rewrite_rule():
    sig = Signature(
        {
            'c': (0, set()),
            'f': (2, set()),
            'g': (1, set()),
            '+' : (2, {'Infix'}),
        }
    )
    parser = Parser(sig)
    term = parser.parse_term('((f (g x) y) + y)')
    rule = parser.parse_rewriterule('((f x m) + m) -> ((f m x) + x)')
    result = rule(sig, term)
    assert result == parser.parse_term('((f y (g x)) + (g x))')

def test_trs():
    sig = Signature(
        {
            '&': (2, {'Infix'}),
            '|': (2, {'Infix'}),
            '~': (1, set()),
        }
    )
    parser = Parser(sig)
    trs = TRS(sig,[
        parser.parse_rewriterule('(~ (~ a)) -> a'),
        parser.parse_rewriterule('(a | b) -> (~ ((~ a) & (~ b)))'),
        parser.parse_rewriterule('(a & a) -> a'),
        parser.parse_rewriterule('(a | a) -> a'),
    ])
    term = parser.parse_term('(~ (~ b))')
    result = trs(term)
    assert result == parser.parse_term('b')

    term = parser.parse_term('(~ (a | b))')
    result = trs(term)
    assert result == parser.parse_term('((~ a) & (~ b))')