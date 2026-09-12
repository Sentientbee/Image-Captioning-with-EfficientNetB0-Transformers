import numpy as np
import pytest
import tensorflow as tf

from src.config import load_config
from src.data.tokenizer import CaptionTokenizer
from src.inference.attention_map import AttentionVisualizer
from src.inference.beam_search import BeamSearchGenerator
from src.models.captioner import build_caption_model


@pytest.fixture(scope="module")
def model_and_tokenizer():
    cfg = load_config()
    tokenizer = CaptionTokenizer.from_vocab_file(
        cfg.paths.vocab_path,
        max_tokens=cfg.model.vocab_size,
        seq_length=cfg.model.seq_length,
    )
    model = build_caption_model(
        image_size=(299, 299),
        channels=3,
        seq_length=25,
        vocab_size=10000,
        embed_dim=64,
        ff_dim=64,
        num_heads_encoder=2,
        num_heads_decoder=3,
        cnn_trainable=False,
    )
    return model, tokenizer


def test_attention_visualizer_4d_input_support(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    vis = AttentionVisualizer(model=model, tokenizer=tokenizer, max_length=10)

    # 4D numpy array with batch size 1
    arr_4d = np.random.randint(0, 256, (1, 299, 299, 3), dtype=np.uint8)
    pil_img, tensor = vis._prepare_inputs(arr_4d)
    assert pil_img.size == (299, 299)
    assert tensor.shape == (1, 299, 299, 3)

    # 4D tensorflow tensor with batch size 1
    tensor_4d = tf.random.uniform((1, 299, 299, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    pil_img2, tensor2 = vis._prepare_inputs(tensor_4d)
    assert pil_img2.size == (299, 299)
    assert tensor2.shape == (1, 299, 299, 3)


def test_extract_attention_for_words_synchronization(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    vis = AttentionVisualizer(model=model, tokenizer=tokenizer, max_length=10)

    dummy_img = np.random.randint(0, 256, (299, 299, 3), dtype=np.uint8)
    words = ["a", "black", "dog", "playing"]

    extracted_words, attns, pil_img = vis.extract_attention_for_words(dummy_img, words)
    assert extracted_words == words
    assert len(attns) == len(words)
    for attn_grid in attns:
        assert attn_grid.shape == (10, 10)
        assert 0.0 <= np.min(attn_grid)
        assert np.max(attn_grid) <= 1.0


def test_beam_search_temperature_parameter_isolation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    beam = BeamSearchGenerator(model=model, tokenizer=tokenizer, beam_width=2, temperature=1.0)

    dummy_img = np.random.randint(0, 256, (299, 299, 3), dtype=np.uint8)
    # Pass custom temperature without mutating instance temperature
    _, candidates = beam.generate(dummy_img, beam_width=2, temperature=0.5)
    assert beam.temperature == 1.0
    assert len(candidates) > 0
