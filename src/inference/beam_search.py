import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from src.config import AppConfig, load_config, resolve_weights_path
from src.data.dataset import decode_and_resize
from src.data.tokenizer import CaptionTokenizer
from src.inference.greedy import GreedyGenerator
from src.models.captioner import ImageCaptioningModel, build_caption_model


@dataclass
class BeamCandidate:
    tokens: List[str]
    cumulative_log_prob: float
    is_finished: bool = False

    def length_penalized_score(self, alpha: float = 0.7) -> float:
        # Wu et al. (2016) Google NMT length penalty
        length = len(self.tokens)
        penalty = ((5.0 + length) / 6.0) ** alpha
        return self.cumulative_log_prob / penalty


class BeamSearchGenerator:
    """Generates captions using top-k Beam Search with length penalty normalization."""

    def __init__(
        self,
        model: ImageCaptioningModel,
        tokenizer: CaptionTokenizer,
        beam_width: int = 3,
        max_length: int = 24,
        length_penalty_alpha: float = 0.7,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty_alpha = length_penalty_alpha
        self.temperature = max(1e-5, temperature)

    def generate(
        self,
        image_input: Union[str, Path, np.ndarray, tf.Tensor],
        beam_width: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Runs Beam Search on input image.

        Returns:
            Tuple of (best_caption: str, all_candidates: List[(caption, score)])
        """
        k = beam_width or self.beam_width
        current_temp = max(1e-5, temperature if temperature is not None else self.temperature)

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

        # Initialize active beams
        beams: List[BeamCandidate] = [
            BeamCandidate(tokens=[self.tokenizer.START_TOKEN], cumulative_log_prob=0.0)
        ]

        for step in range(self.max_length):
            candidates: List[BeamCandidate] = []
            all_finished = True

            for beam in beams:
                if beam.is_finished:
                    candidates.append(beam)
                    continue

                all_finished = False
                caption_str = " ".join(beam.tokens)
                token_ids = self.tokenizer.text_to_sequence([caption_str])[:, :-1]
                mask = tf.math.not_equal(token_ids, 0)

                predictions = self.model.decoder(token_ids, encoded_img, training=False, mask=mask)
                probs = predictions[0, step, :].numpy()

                # Apply temperature
                if current_temp != 1.0:
                    probs = np.log(np.maximum(probs, 1e-12)) / current_temp
                    probs = np.exp(probs - np.max(probs))
                    probs = probs / np.sum(probs)

                log_probs = np.log(np.maximum(probs, 1e-12))
                top_k_indices = np.argsort(log_probs)[-k:]

                for idx in top_k_indices:
                    word = self.tokenizer.index_to_word.get(int(idx), self.tokenizer.UNK_TOKEN)
                    new_tokens = beam.tokens + [word]
                    new_log_prob = beam.cumulative_log_prob + float(log_probs[idx])
                    is_done = word == self.tokenizer.END_TOKEN
                    candidates.append(
                        BeamCandidate(
                            tokens=new_tokens,
                            cumulative_log_prob=new_log_prob,
                            is_finished=is_done,
                        )
                    )

            # Sort candidates by length-penalized score
            candidates.sort(
                key=lambda b: b.length_penalized_score(self.length_penalty_alpha),
                reverse=True,
            )
            beams = candidates[:k]

            if all_finished:
                break

        # Format results
        results = []
        for beam in beams:
            clean_str = self.tokenizer.clean_caption(" ".join(beam.tokens))
            score = beam.length_penalized_score(self.length_penalty_alpha)
            results.append((clean_str, score))

        best_caption = results[0][0] if results else ""
        return best_caption, results


def main():
    parser = argparse.ArgumentParser(description="Generate image caption using Beam Search")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--compare_greedy", action="store_true", help="Print Greedy vs Beam comparison")
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
        model.load_weights(str(weights_path))
        print(f"Loaded weights from {weights_path}.")
    else:
        print("Note: Running with initialized weights (demo/dry-run mode).")

    beam_gen = BeamSearchGenerator(
        model=model,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        max_length=cfg.inference.max_decoded_length,
        length_penalty_alpha=cfg.inference.length_penalty_alpha,
        temperature=args.temperature,
    )

    best_caption, candidates = beam_gen.generate(args.image)

    print("\n--- Beam Search Results ---")
    for rank, (cand, score) in enumerate(candidates, 1):
        print(f"Rank {rank} (Score: {score:+.3f}): {cand}")

    if args.compare_greedy:
        greedy_gen = GreedyGenerator(model=model, tokenizer=tokenizer)
        greedy_caption = greedy_gen.generate(args.image)
        print("\n--- Comparison ---")
        print(f"Greedy Search: {greedy_caption}")
        print(f"Beam Search:   {best_caption}")


if __name__ == "__main__":
    main()
