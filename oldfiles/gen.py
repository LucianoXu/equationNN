from typing import Optional

from tqdm import tqdm
from pyualg import Signature, Term, RewriteRule, TermOpt, Subst, Parser
from rewritepath import RewritePath
from scenario import *
import random

def gen_expression(max_height: int) -> Term:
    '''
    Generate an expression tree with height at most max_height
    '''
    if max_height == 1:
        return Term(random.choice(['x', 'y', 'z', 'w', 'u', 'v']))
    
    choice = random.choices(['|', '&', '~', 'x', 'y', 'z', 'w', 'u', 'v'], [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], k=1)[0]

    if choice == '|' or choice == '&':
        left = gen_expression(max_height - 1)
        right = gen_expression(max_height - 1)
        return Term(choice, (left, right))
    elif choice == '~':
        return Term('~', (gen_expression(max_height - 1),))
    else:
        return Term(choice)

def gen_example(max_step: int = 10, max_height: int = 3, rule_prob : Optional[dict[TermOpt, float]] = None, ban_reverse: bool = True) -> RewritePath:
    '''
    Generate a path of rewriting operations. The path starts from a random expression tree with height at most max_height.
    The path has at most max_step steps.
    The algorithm will check all possible choices of rules and positions at each step. If no choice is possible, the path will be returned directly.
    '''

    # use the equation x = x as the initial term
    # term = gen_expression(max_height)
    term = Term("x")

    path = RewritePath(signature, (), Term('=', (term, term)))


    for _ in range(max_step):
        all_nodes = list(path.current.all_nodes())

        if rule_prob is None:
            rules = rule_ls.copy()
        else:
            rules = list(rule_prob.keys())

        while True:

            # create the probs
            if rule_prob is None:
                probs = [1.0/len(rules) for _ in range(len(rules))]
            else:
                probs = [rule_prob[rule] for rule in rules]

            # select the rule and position
            gen_rule = random.choices(rules, probs, k=1)[0]

            # search through all positions
            remaining_nodes = all_nodes.copy()
            while len(remaining_nodes) > 0:
                node = random.choice(remaining_nodes)
                pos = node[0]
                
                if ban_reverse:
                    # remove the direct inverse operation
                    if len(path.path)>0 and INV_GEN_RULES[gen_rule] == path.path[-1][1] and pos == path.path[-1][2]:
                        remaining_nodes.remove(node)
                        continue

                # generate the given substitution
                given_subst = {}
                for var in INST_VARS[gen_rule]:
                    given_subst[var] = gen_expression(max_height)

                if path.apply(gen_rule, pos, Subst(given_subst), inst_vars=INST_VARS, forbiden_heads=forbidden_heads):
                    break

                remaining_nodes.remove(node)

            # if successfully applied a rule, remaining_nodes should not be empty
            if len(remaining_nodes) > 0:
                break
            
            else:
                rules.remove(gen_rule)
                if len(rules) == 0:
                    return path

    return path

def get_examples(max_step: int = 10, max_height: int = 3, rule_prob : Optional[dict[TermOpt, float]] = None, count: int = 1000, ban_reverse: bool = True) -> list[RewritePath]:
    '''
    Generate a list of examples.
    '''
    path_ls = []
    for _ in tqdm(range(count), desc="Generating examples"):
        path = gen_example(max_step, max_height, rule_prob, ban_reverse)
        path_ls.append(path)

    return path_ls


def get_examples_balenced(max_step: int = 10, max_height: int = 3, count: int = 1000, progress_bar: bool = True, ban_reverse: bool = True) -> list[RewritePath]:
    '''
    Generate a list of examples (in a balenced way).
    '''
    path_ls = []
    rule_count = {rule:1 for rule in rule_ls}

    if progress_bar:
        iterable = tqdm(range(count), desc="Generating examples")
    else:
        iterable = range(count)

    for _ in iterable:

        # calculate the prob for balancing
        rule_probs = {rule:1.0/rule_count[rule]**3 for rule in rule_count}

        path = gen_example(max_step, max_height, rule_probs, ban_reverse)
        path_ls.append(path)

        for _, rule, _, _, _ in path.path:
            rule_count[rule] += 1

    return path_ls

def count_gen_rule(path_ls : list[RewritePath]) -> dict[str, int]:
    '''
    Count the number of times each rule is used in the path list.
    '''
    count = {rule:0 for rule in rule_ls}
    for path in path_ls:
        for _, rule, _, _, _ in path.path:
            count[rule] += 1
    return {RULE_NAMES[rule] : count[rule] for rule in count}

        

if __name__ == "__main__":
    from rewritepath import path_to_examples
    # path = gen_example(10, 3)
    # print(path)

    # examples = path_to_examples(path, signature)
    # for example in examples:
    #     print(example)

    ls = get_examples_balenced(10, 3, count=1000)
    print(count_gen_rule(ls))


    