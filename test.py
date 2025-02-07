from model import SmallArgs, Llama3, batch_generation, tok_decode
import json
from elab import ELab
from rl import gen_example_group

device = json.load(open('config.json'))['backend']

if __name__ == '__main__':
    args = SmallArgs()
    model = Llama3(args, device=device)
    ELab('ckpt/OMLgen', version_name='latest', model=model)

    # beams = [" (x = x) :"] * 10

    # res, _ = batch_generation(model, beams, 256, 0.5)

    # for i in range(len(res)):
    #     print(f"Beam {i}: {res[i]}")

    traces = gen_example_group(model, 1, 15, 256, 0.6)

    for trace in traces[0]:
        print(trace[0])