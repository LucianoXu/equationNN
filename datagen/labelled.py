import numpy as np
import torch
from randomgen import *

def get_term_opt_pairs(path: RewritePath) -> list[tuple[Tree, TreeOpt, tuple[int, ...]]]:
    '''
    return the list of (term, opt, pos) pairs
    '''
    if not path.path:
        return []
    
    res = []
    pre = path.start
    for opt, pos, post in path.path:
        res.append((pre, opt, pos))
        pre = post
        
    return res

def get_single_example_data(
        term: Tree, opt: TreeOpt, pos: tuple[int, ...],
        term_tokenizer: dict[str, int], opt_tokenizer: dict[TreeOpt, int]) -> tuple[list[int], list[tuple[int, ...]], list[int]]:
    '''
    accept: the term, the operation, and the position, as well as two tokenizers
    Note: '0' in opt tokenizer is reserved for no operation
    return:
    - the tokenized term data
    - the position data
    - the target data
    '''
    
    node_list, pos_list, idx_dict = term.flatten()
    term_data = [term_tokenizer[term.data] for term in node_list]
    target_data = [0] * len(node_list)
    target_data[idx_dict[pos]] = opt_tokenizer[opt]

    return term_data, pos_list, target_data
