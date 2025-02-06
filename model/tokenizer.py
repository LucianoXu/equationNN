

TOKENS = [
    # tokesn for the term
    '(', ')', '&', '|', '~', 'x', 'y', 'z', 'w', 'u', 'v', '=',

    # tokens to specify the rewriting
    'subst', 'commM', 'commJ', 'assocM1', 'assocM2', 'assocJ1', 'assocJ2',
    'absorpM1', 'absorpM2', 'absorpJ1', 'absorpJ2', 'doubleNeg1', 'doubleNeg2',
    'deMorganM1', 'deMorganM2', 'deMorganJ1', 'deMorganJ2',
    'OML1', 'OML2',
    
    '0', '1',
    '{', '}', ',', 'X', 'Y',
    
    # token to separate the term and the rewriting
    ':', 

    # special tokens
    '<PAD>', '<EOS>', '<SOS>',
]

token2id = {token: i for i, token in enumerate(TOKENS)}
id2token = {i: token for i, token in enumerate(TOKENS)}


def tok_encode(txt: str) -> list[int]:
    """
    Return the encoding of the example. If encoding is not possible, raise an error.
    """
    encoding: list[int] = []
    while txt:
        while txt[0] == ' ' or txt[0] == '\n' or txt[0] == '\t':
            txt = txt[1:]
            if not txt:
                return encoding

        for token in TOKENS:
            if txt.startswith(token):
                encoding.append(token2id[token])
                txt = txt[len(token):]
                break
        else:
            raise ValueError(f"Cannot encode: {txt}")
    
    return encoding

def tok_decode(encoding: list[int]):
    """
    return the decoding of the example
    """
    txt = ""
    for id in encoding:
        txt += id2token[id] + " "
    return txt
