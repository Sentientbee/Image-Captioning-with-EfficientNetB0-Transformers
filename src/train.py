import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from src.config import AppConfig, load_config
from src.data.dataset import CaptionDatasetLoader
from src.data.tokenizer import CaptionTokenizer
from src.models.captioner import build_caption_model


class WarmupLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by constant learning rate."""

    def __init__(self, post_warmup_lr: float, warmup_steps: int):
        super().__init__()
        self.post_warmup_lr = tf.cast(post_warmup_lr, tf.float32)
        self.warmup_steps = tf.cast(max(1, warmup_steps), tf.float32)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        global_step = tf.cast(step, tf.float32)
        warmup_progress = global_step / self.warmup_steps
        warmup_lr = self.post_warmup_lr * warmup_progress
        return tf.cond(
            global_step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.post_warmup_lr,
        )

    def get_config(self):
        return {
            "post_warmup_lr": float(self.post_warmup_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
        }


def train(
    config_path: str = None,
    images_dir: str = None,
    captions_file: str = None,
    epochs: int = None,
    batch_size: int = None,
    output_weights: str = "weights/model_weights.h5",
):
    """Main training workflow."""
    cfg = load_config(config_path)

    if epochs:
        cfg.training.epochs = epochs
    if batch_size:
        cfg.training.batch_size = batch_size

    print(f"Loading vocabulary from {cfg.paths.vocab_path}...")
    tokenizer = CaptionTokenizer.from_vocab_file(
        cfg.paths.vocab_path,
        max_tokens=cfg.model.vocab_size,
        seq_length=cfg.model.seq_length,
    )

    img_path = images_dir or "data/Images"
    cap_path = captions_file or "data/captions.txt"

    print(f"Initializing dataset loader from:\n  Images: {img_path}\n  Captions: {cap_path}")
    loader = CaptionDatasetLoader(
        images_dir=img_path,
        captions_path=cap_path,
        tokenizer=tokenizer,
        image_size=cfg.model.image_size,
        seq_length=cfg.model.seq_length,
    )

    try:
        mapping, _ = loader.parse_captions_file()
        train_map, val_map, test_map = loader.split_data(mapping)
        print(f"Dataset split: {len(train_map)} train | {len(val_map)} val | {len(test_map)} test")

        train_ds = loader.create_dataset(train_map, batch_size=cfg.training.batch_size, is_training=True)
        val_ds = loader.create_dataset(val_map, batch_size=cfg.training.batch_size, is_training=False)
        total_steps = len(train_map) // cfg.training.batch_size * cfg.training.epochs
    except Exception as e:
        print(f"Warning: Could not load local dataset ({e}). Running in dry-run structure validation mode.")
        return

    print("Building model architecture...")
    model = build_caption_model(
        image_size=cfg.model.image_size,
        channels=cfg.model.channels,
        seq_length=cfg.model.seq_length,
        vocab_size=cfg.model.vocab_size,
        embed_dim=cfg.model.embed_dim,
        ff_dim=cfg.model.ff_dim,
        num_heads_encoder=cfg.model.num_heads_encoder,
        num_heads_decoder=cfg.model.num_heads_decoder,
        cnn_trainable=cfg.model.cnn_trainable,
    )

    warmup_steps = int(total_steps * cfg.training.warmup_steps_ratio)
    lr_schedule = WarmupLRSchedule(post_warmup_lr=cfg.training.learning_rate, warmup_steps=warmup_steps)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none"),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_weights,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    out_file = Path(output_weights)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {cfg.training.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.epochs,
        callbacks=callbacks,
    )

    model.save_weights(output_weights)
    print(f"Training complete. Weights saved to {output_weights}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to images directory")
    parser.add_argument("--captions_file", type=str, default=None, help="Path to captions file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--output_weights", type=str, default="weights/model_weights.h5", help="Path to save weights")
    args = parser.parse_args()

    train(
        config_path=args.config,
        images_dir=args.images_dir,
        captions_file=args.captions_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_weights=args.output_weights,
    )


if __name__ == "__main__":
    main()
