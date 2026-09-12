import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


def custom_standardization(input_string: tf.Tensor) -> tf.Tensor:
    """Standardizes text: lowercase and strip punctuation while preserving special tokens.

    Preserves '<start>' and '<end>' tokens.
    """
    lowercase = tf.strings.lower(input_string)
    # Strip standard punctuation excluding angle brackets '<' and '>'
    strip_chars = r"!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"
    escaped_strip = re.escape(strip_chars)
    return tf.strings.regex_replace(lowercase, f"[{escaped_strip}]", "")


class CaptionTokenizer:
    """Manages text tokenization, vocabulary mapping, and sequence vectorization.

    Uses clean JSON serialization to avoid pickle incompatibilities across
    different TensorFlow/Keras versions.
    """

    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = ""

    def __init__(
        self,
        max_tokens: int = 10000,
        seq_length: int = 25,
        vocab: Optional[List[str]] = None,
    ):
        self.max_tokens = max_tokens
        self.seq_length = seq_length
        self.vocab = vocab or []
        self.index_to_word: Dict[int, str] = {}
        self.word_to_index: Dict[str, int] = {}
        self.vectorizer: Optional[TextVectorization] = None

        if self.vocab:
            self._build_mappings(self.vocab)
            self._init_vectorizer()

    def _build_mappings(self, vocab_list: List[str]) -> None:
        self.vocab = vocab_list
        self.index_to_word = {idx: word for idx, word in enumerate(vocab_list)}
        self.word_to_index = {word: idx for idx, word in enumerate(vocab_list)}

    def _init_vectorizer(self) -> None:
        """Initializes TensorFlow TextVectorization layer with the loaded vocabulary."""
        self.vectorizer = TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.seq_length,
            standardize=custom_standardization,
            vocabulary=self.vocab,
        )

    @classmethod
    def from_vocab_file(
        cls,
        vocab_path: Union[str, Path] = "data/vocab.json",
        max_tokens: int = 10000,
        seq_length: int = 25,
    ) -> "CaptionTokenizer":
        """Load tokenizer vocabulary from a JSON file."""
        resolved = Path(vocab_path)
        if not resolved.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if (repo_root / vocab_path).exists():
                resolved = repo_root / vocab_path
            elif (repo_root / "data" / "vocab.json").exists():
                resolved = repo_root / "data" / "vocab.json"
            else:
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(resolved, "r", encoding="utf-8") as f:
            vocab_list = json.load(f)

        return cls(max_tokens=max_tokens, seq_length=seq_length, vocab=vocab_list)

    def fit_on_texts(self, text_list: List[str]) -> None:
        """Fits TextVectorization on an input list of caption strings."""
        self.vectorizer = TextVectorization(
            max_tokens=self.max_tokens,
            output_mode="int",
            output_sequence_length=self.seq_length,
            standardize=custom_standardization,
        )
        self.vectorizer.adapt(text_list)
        vocab_list = self.vectorizer.get_vocabulary()
        self._build_mappings(vocab_list)

    def save_vocab(self, output_path: Union[str, Path]) -> None:
        """Saves current vocabulary to JSON format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    def text_to_sequence(self, text: Union[str, List[str], tf.Tensor]) -> tf.Tensor:
        """Converts text string, list of strings, or string Tensor to integer sequence tensor."""
        if self.vectorizer is None:
            raise ValueError("Tokenizer has not been fitted or loaded with a vocabulary.")
        if tf.is_tensor(text):
            return self.vectorizer(text)
        if isinstance(text, str):
            text = [text]
        return self.vectorizer(tf.convert_to_tensor(text))

    def sequence_to_text(self, sequence: Union[List[int], tf.Tensor]) -> str:
        """Converts an integer token sequence back into a clean readable string."""
        if isinstance(sequence, tf.Tensor):
            sequence = sequence.numpy().tolist()

        words = []
        for idx in sequence:
            word = self.index_to_word.get(int(idx), self.UNK_TOKEN)
            if word in [self.PAD_TOKEN, self.START_TOKEN]:
                continue
            if word == self.END_TOKEN:
                break
            words.append(word)

        return " ".join(words).strip()

    def format_caption(self, caption: str) -> str:
        """Wraps caption with <start> and <end> tokens."""
        clean = caption.strip()
        return f"{self.START_TOKEN} {clean} {self.END_TOKEN}"

    def clean_caption(self, caption: str) -> str:
        """Strips <start> and <end> markers from a caption string."""
        cleaned = caption.replace(self.START_TOKEN, "").replace(self.END_TOKEN, "")
        return " ".join(cleaned.split()).strip()

    def __len__(self) -> int:
        return len(self.vocab)
