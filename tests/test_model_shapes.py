import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.decoder import TransformerDecoderBlock
from src.models.encoder import TransformerEncoderBlock
from src.models.positional_embedding import PositionalEmbedding


def test_positional_embedding_shape():
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    vocab_size = 100

    pos_emb = PositionalEmbedding(sequence_length=seq_len, vocab_size=vocab_size, embed_dim=embed_dim)
    dummy_input = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)

    output = pos_emb(dummy_input)
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_transformer_encoder_shape():
    batch_size = 2
    num_patches = 16
    cnn_channels = 128
    embed_dim = 64

    enc = TransformerEncoderBlock(embed_dim=embed_dim, dense_dim=64, num_heads=2)
    dummy_features = tf.random.normal((batch_size, num_patches, cnn_channels))

    output = enc(dummy_features, training=False)
    assert output.shape == (batch_size, num_patches, embed_dim)


def test_transformer_decoder_shape_and_attention_scores():
    batch_size = 2
    seq_len = 8
    num_patches = 16
    embed_dim = 64
    vocab_size = 100
    num_heads = 4

    dec = TransformerDecoderBlock(
        embed_dim=embed_dim,
        ff_dim=64,
        num_heads=num_heads,
        vocab_size=vocab_size,
        seq_length=seq_len,
    )

    dummy_tokens = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)
    dummy_img_features = tf.random.normal((batch_size, num_patches, embed_dim))

    # Test standard call
    preds = dec(dummy_tokens, dummy_img_features, training=False)
    assert preds.shape == (batch_size, seq_len, vocab_size)

    # Test call with return_attention_scores=True
    preds, attn_scores = dec(
        dummy_tokens,
        dummy_img_features,
        training=False,
        return_attention_scores=True,
    )
    assert preds.shape == (batch_size, seq_len, vocab_size)
    assert attn_scores is not None
    # Cross attention scores shape: (batch, num_heads, query_seq_len, key_num_patches)
    assert attn_scores.shape == (batch_size, num_heads, seq_len, num_patches)
