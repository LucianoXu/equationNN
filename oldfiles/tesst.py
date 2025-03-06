from model import SmallArgs, Llama3, batch_generation, tok_decode
import json
from elab import ELab
from model.tokenizer import tok_encode
from pyualg.core import Term
from rl import GenEnv, gen_example_group, test_env
from evaluation import test_intere, calc_avg_intere
from gen import get_examples_balenced
from tqdm import tqdm

device = json.load(open('config.json'))['backend']

def eval_fuzzer_avg_intere(max_step: int, max_height: int, count: int = 1000, timeout: float = 10):
    '''
    Evaluate the average interestingness of the fuzzer.
    '''
    # generate examples

    path_ls = get_examples_balenced(max_step, max_height, count = count)

    example_ls = []
    for path in path_ls:
        example_ls.append(path.current)

    # evaluate the interestingness
    return calc_avg_intere(example_ls, timeout)

def eval_model_avg_intere(model: Llama3, max_step: int, context_length: int, T: float, count: int = 1000, timeout: float = 10, batch_size = 10):
    example_ls = []
    for _ in tqdm(range(count//batch_size), desc="Generating examples"):

        envs= [GenEnv(Term("=", (Term("x"), Term("x")))) for _ in range(batch_size)]

        for _ in range(max_step):
            batch = [env.state for env in envs]
            actions, _ = batch_generation(model, batch, context_length, T)

            for i, env in enumerate(envs):
                env.step(actions[i])


        example_ls.extend([env.problem for env in envs])
    
    # evaluate the interestingness
    return calc_avg_intere(example_ls, timeout)


if __name__ == '__main__':
    args = SmallArgs()
    model = Llama3(args, device=device)
    ELab('ckpt/OMLgenBal', version_name='1565', model=model)

    # test_env(model, 10, 256, 0.6)

    # print(eval_fuzzer_avg_intere(10, 3, 300, 10))

    print(eval_model_avg_intere(model, 10, 160, 0.6, 100, 10, 10))