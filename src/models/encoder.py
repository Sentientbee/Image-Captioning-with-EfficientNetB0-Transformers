from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet


def build_cnn_encoder(
    image_size: Tuple[int, int] = (299, 299),
    channels: int = 3,
    trainable: bool = False,
    fine_tune_from_layer: Optional[int] = None,
) -> keras.Model:
    """Builds an EfficientNetB0 CNN backbone feature extractor.

    Args:
        image_size: Input spatial dimensions (height, width).
        channels: Number of image color channels (3 for RGB).
        trainable: Whether the CNN backbone is trainable.
        fine_tune_from_layer: Optional layer index to unfreeze from.

    Returns:
        Keras Model outputting reshaped spatial feature tokens (batch, num_patches, 1280).
    """
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*image_size, channels),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = trainable
    if trainable and fine_tune_from_layer is not None:
        for layer in base_model.layers[:fine_tune_from_layer]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_from_layer:]:
            layer.trainable = True

    base_model_out = base_model.output
    # Reshape (batch, H, W, C) -> (batch, H*W, C), for 299x299 H*W is 10*10 = 100
    reshaped_out = layers.Reshape((-1, base_model_out.shape[-1]), name="spatial_flatten")(base_model_out)
    return keras.models.Model(base_model.input, reshaped_out, name="cnn_encoder")


class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder layer that applies self-attention across image patch tokens."""

    def __init__(
        self,
        embed_dim: int = 512,
        dense_dim: int = 512,
        num_heads: int = 2,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.dense_in = layers.Dense(embed_dim, activation="relu", name="patch_projection")
        self.layernorm_1 = layers.LayerNormalization(name="enc_ln_1")
        self.layernorm_2 = layers.LayerNormalization(name="enc_ln_2")
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate,
            name="enc_self_attn",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Project raw CNN channels to model embed_dim
        projected = self.dense_in(inputs)
        normed_inputs = self.layernorm_1(projected)
        attn_out = self.attention(
            query=normed_inputs,
            value=normed_inputs,
            key=normed_inputs,
            training=training,
        )
        out = self.layernorm_2(projected + attn_out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
