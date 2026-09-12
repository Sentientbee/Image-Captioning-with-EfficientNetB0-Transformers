import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import CaptionDatasetLoader
from src.data.tokenizer import CaptionTokenizer
from src.inference.beam_search import BeamSearchGenerator
from src.inference.greedy import GreedyGenerator
from src.models.captioner import build_caption_model


def compute_corpus_bleu(
    references: List[List[str]],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Computes corpus-level word tokenized BLEU-1 through BLEU-4.

    Fixes:
        Corrects the character-level bug by strictly tokenizing captions into
        lists of words prior to passing to NLTK corpus_bleu.

    Args:
        references: List of lists of ground truth strings: [[ref1, ref2, ...], ...]
        hypotheses: List of predicted candidate strings: [hyp1, hyp2, ...]

    Returns:
        Dict containing BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores (0.0 to 100.0).
    """
    tokenized_refs: List[List[List[str]]] = []
    tokenized_hyps: List[List[str]] = []

    for ref_list, hyp in zip(references, hypotheses):
        # Tokenize each reference into list of words
        tok_refs = [[w.lower() for w in r.strip().split() if w] for r in ref_list]
        tok_hyp = [w.lower() for w in hyp.strip().split() if w]

        tokenized_refs.append(tok_refs)
        tokenized_hyps.append(tok_hyp)

    smoother = SmoothingFunction().method1

    weights_dict = {
        "BLEU-1": (1.0, 0.0, 0.0, 0.0),
        "BLEU-2": (0.5, 0.5, 0.0, 0.0),
        "BLEU-3": (0.333, 0.333, 0.334, 0.0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }

    scores = {}
    for metric_name, weights in weights_dict.items():
        score = corpus_bleu(
            tokenized_refs,
            tokenized_hyps,
            weights=weights,
            smoothing_function=smoother,
        )
        scores[metric_name] = round(score * 100.0, 2)

    return scores


def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """Computes the length of the Longest Common Subsequence between two token sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l(
    references: List[List[str]],
    hypotheses: List[str],
) -> float:
    """Computes ROUGE-L (LCS-based F1) score across the corpus."""
    f1_scores = []

    for ref_list, hyp in zip(references, hypotheses):
        hyp_tokens = [w.lower() for w in hyp.strip().split() if w]
        if not hyp_tokens:
            f1_scores.append(0.0)
            continue

        best_f1 = 0.0
        for ref in ref_list:
            ref_tokens = [w.lower() for w in ref.strip().split() if w]
            if not ref_tokens:
                continue

            lcs = _lcs_length(ref_tokens, hyp_tokens)
            prec = lcs / len(hyp_tokens) if hyp_tokens else 0.0
            rec = lcs / len(ref_tokens) if ref_tokens else 0.0

            if prec + rec > 0:
                f1 = (2 * prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            best_f1 = max(best_f1, f1)

        f1_scores.append(best_f1)

    return round(float(np.mean(f1_scores)) * 100.0, 2) if f1_scores else 0.0


def compute_meteor_scores(
    references: List[List[str]],
    hypotheses: List[str],
) -> Optional[float]:
    """Computes METEOR score if NLTK meteor_score and WordNet resources are available."""
    try:
        from nltk.translate.meteor_score import meteor_score

        meteor_vals = []
        for ref_list, hyp in zip(references, hypotheses):
            tok_refs = [[w.lower() for w in r.strip().split() if w] for r in ref_list]
            tok_hyp = [w.lower() for w in hyp.strip().split() if w]
            score = meteor_score(tok_refs, tok_hyp)
            meteor_vals.append(score)
        return round(float(np.mean(meteor_vals)) * 100.0, 2)
    except Exception:
        # Graceful fallback if WordNet data is not downloaded in environment
        return None


def evaluate_dataset(
    model,
    tokenizer: CaptionTokenizer,
    test_data_map: Dict[str, List[str]],
    use_beam_search: bool = False,
    beam_width: int = 3,
) -> Dict[str, float]:
    """Evaluates the model over an entire test split and reports full benchmark metrics."""
    if use_beam_search:
        generator = BeamSearchGenerator(model=model, tokenizer=tokenizer, beam_width=beam_width)
        gen_fn = lambda img: generator.generate(img)[0]
    else:
        generator = GreedyGenerator(model=model, tokenizer=tokenizer)
        gen_fn = lambda img: generator.generate(img)

    all_refs: List[List[str]] = []
    all_hyps: List[str] = []

    clean_refs_list = []
    for img_path, raw_refs in tqdm(test_data_map.items(), desc="Evaluating Captions"):
        clean_refs = [tokenizer.clean_caption(r) for r in raw_refs]
        pred_caption = gen_fn(img_path)

        clean_refs_list.append(clean_refs)
        all_hyps.append(pred_caption)

    metrics = compute_corpus_bleu(clean_refs_list, all_hyps)
    metrics["ROUGE-L"] = compute_rouge_l(clean_refs_list, all_hyps)

    meteor = compute_meteor_scores(clean_refs_list, all_hyps)
    if meteor is not None:
        metrics["METEOR"] = meteor

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model on benchmark split")
    parser.add_argument("--images_dir", type=str, default="data/Images", help="Path to images directory")
    parser.add_argument("--captions_file", type=str, default="data/captions.txt", help="Path to captions file")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file (.h5)")
    parser.add_argument("--use_beam_search", action="store_true", help="Evaluate using Beam Search")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width if beam search is used")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
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

    loader = CaptionDatasetLoader(
        images_dir=args.images_dir,
        captions_path=args.captions_file,
        tokenizer=tokenizer,
        image_size=cfg.model.image_size,
        seq_length=cfg.model.seq_length,
    )

    try:
        mapping, _ = loader.parse_captions_file()
        _, _, test_map = loader.split_data(mapping)
    except Exception as e:
        print(f"Cannot load dataset from {args.images_dir}: {e}")
        return

    method_name = f"Beam Search (k={args.beam_width})" if args.use_beam_search else "Greedy Search"
    print(f"\nEvaluating on {len(test_map)} test samples using {method_name}...")
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data_map=test_map,
        use_beam_search=args.use_beam_search,
        beam_width=args.beam_width,
    )

    print("\n" + "=" * 40)
    print("      EVALUATION BENCHMARK RESULTS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
