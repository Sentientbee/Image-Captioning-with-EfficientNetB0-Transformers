import argparse
from pathlib import Path
from typing import Optional, Union
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import AppConfig, load_config, resolve_weights_path
from src.data.dataset import decode_and_resize
from src.data.tokenizer import CaptionTokenizer
from src.models.captioner import ImageCaptioningModel, build_caption_model


class GreedyGenerator:
    """Generates captions using greedy (argmax) token decoding."""

    def __init__(
        self,
        model: ImageCaptioningModel,
        tokenizer: CaptionTokenizer,
        max_length: int = 24,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, image_input: Union[str, Path, np.ndarray, tf.Tensor]) -> str:
        """Generates a caption for an input image path or tensor."""
        if isinstance(image_input, (str, Path)):
            img_tensor = decode_and_resize(str(image_input), self.model.cnn_model.input_shape[1:3])
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            img_tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        else:
            arr = image_input
            if len(arr.shape) == 4 and arr.shape[0] == 1:
                arr = arr[0]
            img_tensor = arr

        if tf.reduce_max(img_tensor) <= 1.0:
            img_tensor = img_tensor * 255.0

        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, 0)

        img_features = self.model.cnn_model(img_tensor, training=False)
        encoded_img = self.model.encoder(img_features, training=False)

        decoded_tokens = [self.tokenizer.START_TOKEN]
        for i in range(self.max_length):
            caption_str = " ".join(decoded_tokens)
            token_ids = self.tokenizer.text_to_sequence([caption_str])[:, :-1]
            mask = tf.math.not_equal(token_ids, 0)

            predictions = self.model.decoder(token_ids, encoded_img, training=False, mask=mask)
            sampled_idx = int(np.argmax(predictions[0, i, :].numpy()))
            sampled_word = self.tokenizer.index_to_word.get(sampled_idx, self.tokenizer.UNK_TOKEN)

            if sampled_word == self.tokenizer.END_TOKEN:
                break
            decoded_tokens.append(sampled_word)

        return self.tokenizer.clean_caption(" ".join(decoded_tokens))


def main():
    parser = argparse.ArgumentParser(description="Generate image caption using Greedy Search")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file (.h5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = CaptionTokenizer.from_vocab_file(
        cfg.paths.vocab_path,
        max_tokens=cfg.model.vocab_size,
        seq_length=cfg.model.seq_length,
    )

    model = build_caption_model(
        image_size=cfg.model.image_size,
        channels=cfg.model.channels,
        seq_length=cfg.model.seq_length,
        vocab_size=cfg.model.vocab_size,
        embed_dim=cfg.model.embed_dim,
        ff_dim=cfg.model.ff_dim,
    )

    weights_path = resolve_weights_path(args.weights or cfg.paths.weights_path)
    if weights_path and weights_path.exists():
        print(f"Loading weights from {weights_path}...")
        model.load_weights(str(weights_path))
    else:
        print("Note: Running with initialized weights (demo/dry-run mode).")

    generator = GreedyGenerator(model=model, tokenizer=tokenizer, max_length=cfg.inference.max_decoded_length)
    caption = generator.generate(args.image)
    print(f"\nResult Caption: {caption}")


if __name__ == "__main__":
    main()
