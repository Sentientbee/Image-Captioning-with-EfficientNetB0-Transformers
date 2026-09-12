import pytest
from src.metrics.evaluate import compute_corpus_bleu, compute_rouge_l


def test_compute_corpus_bleu_perfect_match():
    references = [
        ["a brown dog runs through green grass", "dog is running outdoors"],
        ["two young boys are playing soccer", "boys play with a ball"],
    ]
    hypotheses = [
        "a brown dog runs through green grass",
        "two young boys are playing soccer",
    ]

    scores = compute_corpus_bleu(references, hypotheses)
    assert scores["BLEU-1"] == pytest.approx(100.0, rel=1e-2)
    assert scores["BLEU-2"] == pytest.approx(100.0, rel=1e-2)
    assert scores["BLEU-4"] == pytest.approx(100.0, rel=1e-2)


def test_compute_corpus_bleu_word_level_discrimination():
    # If this was character-level, "a dog" and "a dig" would have ~80% BLEU
    # At word level, n-gram precision for higher n-grams is strictly checked
    references = [["the cat sat on the mat"]]
    hypotheses = ["the cat sat on a rug"]

    scores = compute_corpus_bleu(references, hypotheses)
    assert scores["BLEU-1"] > 50.0
    assert scores["BLEU-4"] < 70.0


def test_compute_rouge_l():
    references = [
        ["the quick brown fox jumps over the lazy dog"],
    ]
    hypotheses = [
        "the quick brown fox jumps",
    ]
    score = compute_rouge_l(references, hypotheses)
    assert score > 50.0
