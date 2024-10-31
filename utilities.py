from model import *
import json

def load_model(
        model_checkpint: str, 
        model_args: ModelArgs,
        device: str = 'cpu'
        ) -> Transformer:
    '''
    Load the model from the given checkpoint, with the specified arguments
    '''

    model = Transformer(model_args, device)
    obj = torch.load(model_checkpint)
    model.load_state_dict(obj['model_dict'])
    return model