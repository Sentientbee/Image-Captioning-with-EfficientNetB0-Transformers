from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.positional_embedding import PositionalEmbedding


class TransformerDecoderBlock(layers.Layer):
    """Transformer Decoder with Causal Self-Attention and Image Cross-Attention.

    Fixes:
        - Resolves cross-attention key mask bug (does not pass text mask to visual keys).
        - Handles mask=None without raising UnboundLocalError.
        - Supports return_attention_scores=True to extract cross-attention maps for visualization.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 512,
        num_heads: int = 3,
        vocab_size: int = 10000,
        seq_length: int = 25,
        dropout_rate_attn: float = 0.1,
        dropout_rate_dense: float = 0.3,
        dropout_rate_out: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dropout_rate_attn = dropout_rate_attn
        self.dropout_rate_dense = dropout_rate_dense
        self.dropout_rate_out = dropout_rate_out

        self.embedding = PositionalEmbedding(
            sequence_length=seq_length,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            name="dec_pos_embedding",
        )

        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate_attn,
            name="dec_self_attn",
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate_attn,
            name="dec_cross_attn",
        )

        self.layernorm_1 = layers.LayerNormalization(name="dec_ln_1")
        self.layernorm_2 = layers.LayerNormalization(name="dec_ln_2")
        self.layernorm_3 = layers.LayerNormalization(name="dec_ln_3")

        self.ffn_1 = layers.Dense(ff_dim, activation="relu", name="dec_ffn_1")
        self.dropout_1 = layers.Dropout(dropout_rate_dense, name="dec_dropout_1")
        self.ffn_2 = layers.Dense(embed_dim, name="dec_ffn_2")
        self.dropout_2 = layers.Dropout(dropout_rate_out, name="dec_dropout_2")

        self.out_dense = layers.Dense(vocab_size, activation="softmax", name="dec_vocab_projection")
        self.supports_masking = True

    def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        """Generates lower-triangular causal attention mask (batch, seq_len, seq_len)."""
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(
        self,
        inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        training: bool = False,
        mask: Optional[tf.Tensor] = None,
        return_attention_scores: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass through the decoder block.

        Args:
            inputs: Token indices tensor (batch, seq_len).
            encoder_outputs: Visual feature tokens from encoder (batch, num_patches, embed_dim).
            training: Whether running in training mode.
            mask: Optional text padding mask (batch, seq_len).
            return_attention_scores: If True, also returns cross-attention weights.

        Returns:
            Word predictions (batch, seq_len, vocab_size), or tuple with attention scores.
        """
        # 1. Embedding and Causal Mask
        embedded_inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(embedded_inputs)

        # 2. Text Padding Mask combined with Causal Mask for Self-Attention
        if mask is not None:
            padding_mask_self = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_self_mask = tf.minimum(padding_mask_self, causal_mask)
        else:
            combined_self_mask = causal_mask

        # 3. Causal Multi-Head Self-Attention
        attn_out_1 = self.self_attention(
            query=embedded_inputs,
            value=embedded_inputs,
            key=embedded_inputs,
            attention_mask=combined_self_mask,
            training=training,
        )
        normed_1 = self.layernorm_1(embedded_inputs + attn_out_1)

        # 4. Multi-Head Cross-Attention to Image Patch Tokens
        # Query: text representations. Keys/Values: image features.
        # Note: Visual features have no padding tokens, so attention_mask is None.
        if return_attention_scores:
            attn_out_2, cross_attn_scores = self.cross_attention(
                query=normed_1,
                value=encoder_outputs,
                key=encoder_outputs,
                attention_mask=None,
                training=training,
                return_attention_scores=True,
            )
        else:
            attn_out_2 = self.cross_attention(
                query=normed_1,
                value=encoder_outputs,
                key=encoder_outputs,
                attention_mask=None,
                training=training,
            )
            cross_attn_scores = None

        normed_2 = self.layernorm_2(normed_1 + attn_out_2)

        # 5. Position-wise Feed-Forward Network
        ffn_out = self.ffn_1(normed_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_2(ffn_out)
        normed_3 = self.layernorm_3(normed_2 + ffn_out)
        normed_3 = self.dropout_2(normed_3, training=training)

        # 6. Vocab Projection
        predictions = self.out_dense(normed_3)

        if return_attention_scores:
            return predictions, cross_attn_scores
        return predictions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "ff_dim": self.ff_dim,
                "num_heads": self.num_heads,
                "vocab_size": self.vocab_size,
                "seq_length": self.seq_length,
                "dropout_rate_attn": self.dropout_rate_attn,
                "dropout_rate_dense": self.dropout_rate_dense,
                "dropout_rate_out": self.dropout_rate_out,
            }
        )
        return config
