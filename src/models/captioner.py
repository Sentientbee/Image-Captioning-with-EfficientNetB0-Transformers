from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.decoder import TransformerDecoderBlock
from src.models.encoder import TransformerEncoderBlock, build_cnn_encoder


class ImageCaptioningModel(keras.Model):
    """End-to-end Image Captioning Model combining CNN encoder and Transformer decoder.

    Optimizations & Fixes:
        - Vectorized multi-caption loss computation for significant training speedup.
        - CNN backbone gradients properly tracked inside tf.GradientTape if unfreezed.
        - Numerically stable masked SparseCategoricalCrossentropy loss and token accuracy.
    """

    def __init__(
        self,
        cnn_model: keras.Model,
        encoder: TransformerEncoderBlock,
        decoder: TransformerDecoderBlock,
        num_captions_per_image: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")

    def build(self, input_shape=None):
        super().build(input_shape)
        self.built = True

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
        return_attention_scores: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Functional call for model execution.

        Args:
            inputs: Tuple of (image_tensor, caption_input_tokens)
            training: Boolean flag for training mode
            return_attention_scores: If True, returns (predictions, attention_scores)
        """
        images, captions = inputs
        img_features = self.cnn_model(images, training=training)
        encoded_img = self.encoder(img_features, training=training)
        mask = tf.math.not_equal(captions, 0)

        return self.decoder(
            captions,
            encoded_img,
            training=training,
            mask=mask,
            return_attention_scores=return_attention_scores,
        )

    def calculate_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Computes masked sparse categorical cross-entropy loss."""
        loss = self.loss(y_true, y_pred)
        mask_float = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask_float
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask_float) + 1e-8)

    def calculate_accuracy(self, y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Computes masked next-token prediction accuracy."""
        pred_tokens = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        matches = tf.equal(y_true, pred_tokens)
        matches_masked = tf.math.logical_and(mask, matches)
        matches_float = tf.cast(matches_masked, dtype=tf.float32)
        mask_float = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(matches_float) / (tf.reduce_sum(mask_float) + 1e-8)

    def _compute_batch_loss_and_acc(
        self,
        batch_img: tf.Tensor,
        batch_seq: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Vectorized computation across all captions per image."""
        batch_size = tf.shape(batch_img)[0]
        num_captions = tf.shape(batch_seq)[1]

        # 1. Visual Feature Extraction inside tape
        img_features = self.cnn_model(batch_img, training=training)
        encoded_img = self.encoder(img_features, training=training)

        # 2. Vectorize: Repeat visual features across all captions
        encoded_img_repeated = tf.repeat(encoded_img, repeats=num_captions, axis=0)

        # 3. Reshape captions from (B, 5, seq_len) to (B*5, seq_len)
        flat_seq = tf.reshape(batch_seq, (batch_size * num_captions, -1))
        flat_inp = flat_seq[:, :-1]
        flat_true = flat_seq[:, 1:]

        mask = tf.math.not_equal(flat_true, 0)
        flat_pred = self.decoder(flat_inp, encoded_img_repeated, training=training, mask=mask)

        loss = self.calculate_loss(flat_true, flat_pred, mask)
        acc = self.calculate_accuracy(flat_true, flat_pred, mask)
        return loss, acc

    def train_step(self, batch_data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_img, batch_seq = batch_data

        with tf.GradientTape() as tape:
            loss, acc = self._compute_batch_loss_and_acc(batch_img, batch_seq, training=True)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_img, batch_seq = batch_data
        loss, acc = self._compute_batch_loss_and_acc(batch_img, batch_seq, training=False)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


def build_caption_model(
    image_size: Tuple[int, int] = (299, 299),
    channels: int = 3,
    seq_length: int = 25,
    vocab_size: int = 10000,
    embed_dim: int = 512,
    ff_dim: int = 512,
    num_heads_encoder: int = 2,
    num_heads_decoder: int = 3,
    cnn_trainable: bool = False,
    dropout_rate_attn: float = 0.1,
    dropout_rate_dense: float = 0.3,
    dropout_rate_out: float = 0.5,
) -> ImageCaptioningModel:
    """Factory helper to build and initialize ImageCaptioningModel."""
    cnn = build_cnn_encoder(image_size=image_size, channels=channels, trainable=cnn_trainable)
    encoder = TransformerEncoderBlock(embed_dim=embed_dim, dense_dim=ff_dim, num_heads=num_heads_encoder)
    decoder = TransformerDecoderBlock(
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        num_heads=num_heads_decoder,
        vocab_size=vocab_size,
        seq_length=seq_length,
        dropout_rate_attn=dropout_rate_attn,
        dropout_rate_dense=dropout_rate_dense,
        dropout_rate_out=dropout_rate_out,
    )

    model = ImageCaptioningModel(cnn_model=cnn, encoder=encoder, decoder=decoder)

    # Materialize layers and ensure model.built is True for Keras 3 ModelCheckpoint
    dummy_img = tf.zeros((1, *image_size, channels))
    dummy_seq = tf.zeros((1, seq_length - 1), dtype=tf.int32)
    _ = model((dummy_img, dummy_seq), training=False)
    model.built = True

    return model
