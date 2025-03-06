from .backend_interface import env

class Scenario:
    '''
    The class that collects the environment backend components for a specific algebra.
    '''
    def __init__(self, alg: env.Algebra|str):

        if isinstance(alg, str):
            parse_res = env.parse_alg(alg)
            if parse_res is None:
                raise ValueError("Failed to parse the algebra description:\n\n" + alg)
            assert parse_res is not None
            self.alg = parse_res
        else:
            self.alg = alg

        self.sig = self.alg.signature
        self.kernel = env.SymbolKernel(self.alg)
        self.tokenizer = env.Tokenizer(self.alg)

        self.SOS_ID = self.tokenizer.get_encoding("<SOS>")
        self.EOS_ID = self.tokenizer.get_encoding("<EOS>")
        self.PAD_ID = self.tokenizer.get_encoding("<PAD>")