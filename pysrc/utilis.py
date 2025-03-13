import sys
import shlex

from .env import env, Scenario

def get_command():
    '''
    Get the full command that initiated the current process.
    '''
    return f"{sys.executable} " + " ".join(map(shlex.quote, sys.argv))

def parse_examples(scenario: Scenario, file: str):
    with open(file) as f:
        lines = f.readlines()

    # remove those lines that start with %
    lines = [line for line in lines if not line.startswith("%")]

    # remove empty lines
    lines = [line for line in lines if line.strip()]

    examples = []
    for line in lines:
        eq = env.parse_equation(line)
        if eq is None:
            raise ValueError(f"Cannot parse the equation: {line}")

        # check whether the equation is valid
        if not (scenario.sig.term_valid(eq.lhs) and scenario.sig.term_valid(eq.rhs)):
            raise ValueError(f"The equation {eq} is not valid in the signature.")

        examples.append(eq)
        
    return examples


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu_nltk_intlists(references: list[list[int]], hypothesis: list[list[int]], weights=(0.25, 0.25, 0.25, 0.25)) -> list[float]:
    """
    Calculate BLEU using NLTK's BLEU implementation for integer-token samples.
    
    :param references: A list of reference samples, 
                       where each sample is a list of integer tokens (e.g., [101, 42, 37, ...]).
    :param hypothesis: A list of generated samples, 
                       where each sample is a list of integer tokens (e.g., [101, 42, 37, ...]).
    :param weights: Weights for n-gram precision. 
                    Default is (0.25, 0.25, 0.25, 0.25) corresponding to 4-gram BLEU.
    :return: The individual BLEU scores.
    """
    scores = []
    smoothing_fn = SmoothingFunction().method1  # A simple smoothing function for BLEU

    # NLTK's sentence_bleu expects a list of token-lists as references
    for h in hypothesis:
        bleu = sentence_bleu(
            references,           # list of references
            h,           # single hypothesis (list of tokens)
            weights=weights,
            smoothing_function=smoothing_fn
        )
        scores.append(bleu)

    return scores

def self_bleu_nltk_intlists(samples: list[list[int]], weights=(0.25, 0.25, 0.25, 0.25)) -> list[float]:
    """
    Calculate Self-BLEU using NLTK's BLEU implementation for integer-token samples.
    
    :param samples: A list of generated samples, 
                    where each sample is a list of integer tokens (e.g., [101, 42, 37, ...]).
    :param weights: Weights for n-gram precision. 
                    Default is (0.25, 0.25, 0.25, 0.25) corresponding to 4-gram BLEU.
    :return: The individual BLEU scores (the lower the score, the more diverse the set of samples).
    """
    scores = []
    smoothing_fn = SmoothingFunction().method1  # A simple smoothing function for BLEU

    for i, sample in enumerate(samples):
        # Current sample is the "hypothesis"
        hypothesis = sample
        # All other samples are the references
        references = [ref for j, ref in enumerate(samples) if j != i]

        # NLTK's sentence_bleu expects a list of token-lists as references
        bleu = sentence_bleu(
            references,           # list of references
            hypothesis,           # single hypothesis (list of tokens)
            weights=weights,
            smoothing_function=smoothing_fn
        )
        scores.append(bleu)

    # Return BLEU scores
    return scores