
def test_self_bleu_nltk_intlists():
    from ..utilis import self_bleu_nltk_intlists

    # Example tokenized integer lists (just placeholders for demonstration)
    generated_samples = [
        [101, 32, 45, 999],  # e.g., tokenized "x^2 + y^2 = z^2"
        [101, 54, 78, 456],  # e.g., tokenized "a^2 + b^2 = c^2"
        [555, 666, 777],
        [12, 34, 56, 78],
        [999, 123, 777, 555],
        [999, 123, 777, 555],
        [999, 123, 777, 2, 2]
    ]
    
    score = self_bleu_nltk_intlists(generated_samples)
    print(f"Self-BLEU (NLTK) Score: {score:.4f}")
