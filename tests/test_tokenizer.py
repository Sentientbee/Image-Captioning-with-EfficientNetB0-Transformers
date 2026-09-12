import pytest
import tensorflow as tf
from src.data.tokenizer import CaptionTokenizer, custom_standardization


def test_custom_standardization():
    sample = tf.constant("<start> A brown dog jumps over a fence! <end>")
    cleaned = custom_standardization(sample).numpy().decode("utf-8")
    assert "<start>" in cleaned
    assert "<end>" in cleaned
    assert "!" not in cleaned
    assert "brown dog jumps" in cleaned


def test_tokenizer_from_vocab_file():
    tokenizer = CaptionTokenizer.from_vocab_file(
        "data/vocab.json",
        max_tokens=10000,
        seq_length=25,
    )
    assert len(tokenizer) == 10000
    assert tokenizer.START_TOKEN == "<start>"
    assert tokenizer.END_TOKEN == "<end>"
    assert tokenizer.index_to_word[3] == "<start>"
    assert tokenizer.index_to_word[4] == "<end>"


def test_tokenizer_encode_decode():
    tokenizer = CaptionTokenizer.from_vocab_file("data/vocab.json")
    caption = "<start> a dog is running <end>"
    seq = tokenizer.text_to_sequence(caption)

    assert seq.shape == (1, 25)
    decoded = tokenizer.sequence_to_text(seq[0])
    assert "a dog is running" in decoded


def test_caption_formatting():
    tokenizer = CaptionTokenizer.from_vocab_file("data/vocab.json")
    raw = "two children playing"
    formatted = tokenizer.format_caption(raw)
    assert formatted == "<start> two children playing <end>"
    assert tokenizer.clean_caption(formatted) == "two children playing"
