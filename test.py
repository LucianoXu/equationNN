from model import SmallArgs, Llama3, batch_generation
import json
from elab import ELab

device = json.load(open('config.json'))['backend']

if __name__ == '__main__':
    args = SmallArgs()
    model = Llama3(args, device=device)
    ELab('ckpt/OMLgen', version_name='latest', model=model)

    beams = [" (x = x) :"] * 10

    res, _ = batch_generation(model, beams, 256, 0.5)

    for i in range(len(res)):
        print(f"Beam {i}: {res[i]}")