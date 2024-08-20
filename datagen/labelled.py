import numpy as np
import torch
from tqdm import tqdm
from randomgen import *
from numba import njit, jit

def get_term_opt_pairs(path: RewritePath) -> list[tuple[Tree, TreeOpt, tuple[int, ...]]]:
    '''
    return the list of (term, opt, pos) pairs
    '''

    return path.path.copy()

PosInst = tuple[list[list[int]], list[list[list[int]]]]

# @njit
def pos_list2encoding_instruction(pos_list: list[tuple[int, ...]], 
                                  height: int,
                                  width: int) -> PosInst:
    '''
    convert the position list to the encoding instruction list

    input:
    - height: the height of the tree
    - width: the maximum branch number of the tree

    return
    - layer_inst: the instruction lists for each layer, (height-1)*? shape, with the index i list containing the indices that needs to apply the rotary embedding at depth i+1 (depth 0 does not need and is not considered)
    - branch_inst: the instruction lists for each branch, (height-1)*width*? shape, with the index i,j list containing the indices that needs to apply the rotary embedding at depth i+1 and branch j 
    '''
    layer_inst : list[list[int]] = [[] for _ in range(height-1)]
    branch_inst : list[list[list[int]]] = [[[] for _ in range(width)] for _ in range(height-1)]

    for i in range(len(pos_list)):
        pos = pos_list[i]
        for j in range(len(pos)):
            # register in the layer inst
            layer_inst[j].append(i)
            # register in the branch layer
            branch_inst[j][pos[j]].append(i)

    return layer_inst, branch_inst


def get_model_input_from_term(
        term: Tree, max_height: int, width: int,
        term_tokenizer: dict[str, int]) -> tuple[list[int], list[tuple[int, ...]], PosInst]:
    
    node_list, pos_list, _ = term.flatten()

    term_data = [term_tokenizer[term.data] for term in node_list]

    pos_instruct = pos_list2encoding_instruction(
        pos_list, height=max_height, width=width)
    
    return term_data, pos_list, pos_instruct

def model_input_add_padding(
        term_data: list[int], pos_list: list[tuple[int, ...]], pos_instruct: PosInst,
        max_length: Optional[int] = None, device = 'cpu') -> tuple[torch.Tensor, list[tuple[int, ...]], list[PosInst], torch.Tensor]:
    '''
    add padding (0) to the model input, add mask tensor to the output
    transform the output to suit the input format of transformer
    return: input encoding, position list, position encoding instruction, mask
    '''
    if max_length is None:
        max_length = len(term_data)

    term_data = term_data + [0] * (max_length - len(term_data))
    input = torch.tensor(term_data, dtype=torch.int64).to(device).unsqueeze(0)

    mask = [1] * len(term_data) + [0] * (max_length - len(term_data))
    mask = torch.tensor(mask, dtype=torch.int64).to(device).view((1,1,1,-1))

    return input, pos_list, [pos_instruct], mask
    


def get_single_example_data(
        term: Tree, max_height: int, width: int,
        opt: TreeOpt, pos: tuple[int, ...],
        term_tokenizer: dict[str, int], opt_tokenizer: dict[TreeOpt, int]) -> tuple[list[int], list[tuple[int, ...]], PosInst, list[int]]:
    '''
    Transform one operation into the single supervised learning data example.

    accept: the term, the operation, and the position, as well as two tokenizers
    Note: '0' in opt tokenizer is reserved for no operation
    return:
    - the tokenized term data
    - the position instruction
    - the target data
    '''
    node_list, pos_list, idx_dict = term.flatten()

    term_data = [term_tokenizer[term.data] for term in node_list]

    target_data = [0] * len(node_list)
    target_data[idx_dict[pos]] = opt_tokenizer[opt]

    pos_instruct = pos_list2encoding_instruction(pos_list, height=max_height, width=width)

    return term_data, pos_list, pos_instruct, target_data


def synthesize_example_thread(height: int, max_height: int, path_length: int, n: int,
                              max_length: Optional[int] = None) -> list[tuple[list[int], list[tuple[int, ...]], PosInst, list[int]]]:
    '''
    synthesize n examples for the thread with the maximum height
    path_length: the maximum length of every random rewriting path
    max_length: if not None, will filter out the examples with the length greater than max_length
    '''
    examples: list[tuple[list[int], list[tuple[int, ...]], PosInst, list[int]]] = []

    random_rule = [rule_comm, rule_assoc1, rule_assoc2]

    # use tqdm to visualize the progress
    progress_bar = tqdm(total=n)

    while len(examples) < n:
        path = get_head(height)
        
        # construct the random rewriting path
        random_apply_n(path, random_rule, path_length)

        # get the supervised learning data
        invpath = path.get_inverse(inverse_table)

        for step in invpath.path:
            term, opt, pos = step
            single_data = get_single_example_data(
                term, max_height, 2,
                opt, pos, 
                term_tokenizer, opt_tokenizer
            )

            # skip the example if the length is too long
            if max_length is not None and len(single_data[0]) > max_length:
                continue

            examples.append(single_data)

        progress_bar.update(len(invpath.path))

    return examples    
            

