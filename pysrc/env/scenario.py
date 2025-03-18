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

        self.PAD = self.tokenizer.get_encoding("<PAD>")

        self.START_STT = self.tokenizer.get_encoding("<STT>")
        self.END_STT = self.tokenizer.get_encoding("</STT>")
        self.START_ACT = self.tokenizer.get_encoding("<ACT>")
        self.END_ACT = self.tokenizer.get_encoding("</ACT>")