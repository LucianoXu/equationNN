from typing import Optional
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst, Parser
from rewritepath import RewritePath
import random

from scenario import *

def test_example_gen():
    for _ in range(100):
        A = gen_example(max_step=10, max_height=3)
        A.verify({r_L2R, r_R2L}, forbidden_heads)