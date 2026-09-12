import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import AppConfig, load_config
from src.data.dataset import decode_and_resize
from src.data.tokenizer import CaptionTokenizer
from src.models.captioner import ImageCaptioningModel, build_caption_model


class AttentionVisualizer:
    """Extracts cross-attention weights and overlays spatial heatmaps per generated word."""

    def __init__(
        self,
        model: ImageCaptioningModel,
        tokenizer: CaptionTokenizer,
        max_length: int = 18,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate_with_attention(
        self,
        image_input: Union[str, Path, np.ndarray, tf.Tensor],
    ) -> Tuple[List[str], List[np.ndarray], Image.Image]:
        """Generates a caption while recording the cross-attention spatial heatmap for each token.

        Returns:
            Tuple of:
                - words: List[str] of generated words
                - attention_maps: List[np.ndarray] of 2D normalized heatmaps (H, W)
                - original_pil_img: PIL Image of the input image
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            img_tensor = decode_and_resize(str(image_input), self.model.cnn_model.input_shape[1:3])
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input.astype(np.uint8))
            img_tensor = tf.convert_to_tensor(image_input, dtype=tf.float32)
        else:
            pil_img = Image.fromarray((image_input.numpy() * 255).astype(np.uint8))
            img_tensor = image_input

        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, 0)

        # 1. Visual Feature Extraction
        img_features = self.model.cnn_model(img_tensor, training=False)
        encoded_img = self.model.encoder(img_features, training=False)

        # Spatial grid dimensions: e.g. 10x10 = 100 patches for 299x299
        num_patches = int(encoded_img.shape[1])
        grid_dim = int(np.sqrt(num_patches))

        decoded_tokens = [self.tokenizer.START_TOKEN]
        generated_words: List[str] = []
        attention_maps: List[np.ndarray] = []

        for step in range(self.max_length):
            caption_str = " ".join(decoded_tokens)
            token_ids = self.tokenizer.text_to_sequence([caption_str])[:, :-1]
            mask = tf.math.not_equal(token_ids, 0)

            predictions, cross_attn = self.model.decoder(
                token_ids,
                encoded_img,
                training=False,
                mask=mask,
                return_attention_scores=True,
            )

            # cross_attn shape: (batch, num_heads, seq_len, num_patches)
            sampled_idx = int(np.argmax(predictions[0, step, :].numpy()))
            sampled_word = self.tokenizer.index_to_word.get(sampled_idx, self.tokenizer.UNK_TOKEN)

            if sampled_word == self.tokenizer.END_TOKEN:
                break

            # Average cross-attention weights across attention heads for current token step
            # attn_weights shape: (num_heads, num_patches) -> mean over heads -> (num_patches,)
            if cross_attn is not None:
                attn_heads = cross_attn[0, :, step, :].numpy()
                attn_vec = np.mean(attn_heads, axis=0)
            else:
                # Fallback uniform attention if attention scores not returned
                attn_vec = np.ones((num_patches,), dtype=np.float32) / num_patches

            # Reshape into 2D spatial grid (grid_dim x grid_dim)
            if len(attn_vec) == grid_dim * grid_dim:
                attn_grid = attn_vec.reshape((grid_dim, grid_dim))
            else:
                attn_grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)

            # Normalize heatmap
            min_val, max_val = np.min(attn_grid), np.max(attn_grid)
            norm_attn = (attn_grid - min_val) / (max_val - min_val + 1e-8)

            decoded_tokens.append(sampled_word)
            generated_words.append(sampled_word)
            attention_maps.append(norm_attn)

        return generated_words, attention_maps, pil_img

    def plot_and_save(
        self,
        image_input: Union[str, Path, np.ndarray, tf.Tensor],
        output_path: Union[str, Path] = "outputs/attention_heatmap.png",
        colormap: str = "plasma",
        alpha: float = 0.55,
    ) -> Tuple[str, Path]:
        """Generates attention heatmaps and saves a multi-panel visualization figure."""
        words, attns, pil_img = self.generate_with_attention(image_input)
        caption = " ".join(words)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not words:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(pil_img)
            ax.set_title("No caption generated", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=200)
            plt.close()
            return caption, output_path

        # Layout: 1 plot for the original image + 1 plot for each word
        total_plots = len(words) + 1
        cols = min(4, total_plots)
        rows = int(np.ceil(total_plots / cols))

        fig = plt.figure(figsize=(cols * 3.5, rows * 3.5), dpi=150)

        # Subplot 1: Original image
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(pil_img)
        ax0.set_title("Original Image", fontsize=11, fontweight="bold", pad=8)
        ax0.axis("off")

        # Resize PIL image for overlay dimensions
        img_w, img_h = pil_img.size

        # Subplots 2..N: Word-by-word cross-attention overlay
        for i, (word, attn_map) in enumerate(zip(words, attns), start=2):
            ax = fig.add_subplot(rows, cols, i)
            ax.imshow(pil_img)

            # Resize low-res attention grid to original image resolution with bicubic interpolation
            attn_img = Image.fromarray((attn_map * 255).astype(np.uint8))
            resized_attn = np.array(attn_img.resize((img_w, img_h), resample=Image.BICUBIC)) / 255.0

            ax.imshow(resized_attn, cmap=colormap, alpha=alpha)
            ax.set_title(f'"{word}"', fontsize=12, fontweight="bold", color="#1a237e", pad=8)
            ax.axis("off")

        fig.suptitle(f'Generated: "{caption}"', fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        return caption, output_path


def main():
    parser = argparse.ArgumentParser(description="Generate caption and visualize Cross-Attention Heatmaps")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--output", type=str, default="outputs/attention_heatmap.png", help="Output path for figure")
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

    weights_path = args.weights or cfg.paths.weights_path
    if weights_path and Path(weights_path).exists():
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}.")
    else:
        print("Note: Running with initialized weights (demo/dry-run mode).")

    visualizer = AttentionVisualizer(model=model, tokenizer=tokenizer)
    caption, saved_path = visualizer.plot_and_save(args.image, output_path=args.output)
    print(f"\nCaption: {caption}")
    print(f"Attention Heatmap Visualization saved to: {saved_path}")


if __name__ == "__main__":
    main()
