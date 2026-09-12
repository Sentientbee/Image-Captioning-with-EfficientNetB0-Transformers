import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    """Adds learnable positional encodings to token embeddings."""

    def __init__(
        self,
        sequence_length: int = 25,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="token_embedding",
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim,
            name="position_embedding",
        )
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs) * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
