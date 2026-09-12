import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data.tokenizer import CaptionTokenizer


def get_image_augmentation() -> keras.Sequential:
    """Builds visual augmentation pipeline for training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomContrast(0.2),
        ],
        name="image_augmentation",
    )


def decode_and_resize(
    img_path: tf.Tensor,
    image_size: Tuple[int, int] = (299, 299),
) -> tf.Tensor:
    """Reads an image from disk, decodes JPEG/PNG, and resizes to target shape."""
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape((*image_size, 3))
    return img


class CaptionDatasetLoader:
    """Loads, splits, and compiles tf.data.Dataset pipelines for image-caption pairs."""

    def __init__(
        self,
        images_dir: Union[str, Path],
        captions_path: Union[str, Path],
        tokenizer: CaptionTokenizer,
        image_size: Tuple[int, int] = (299, 299),
        seq_length: int = 25,
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.captions_path = Path(captions_path)

        # Autodetect fallback paths relative to repository root if missing
        if not self.images_dir.exists() or not self.captions_path.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            if not self.images_dir.exists():
                alt_images = [
                    repo_root / images_dir,
                    repo_root / "data" / "Flick8k Dataset" / "Images",
                    repo_root / "data" / "Images",
                ]
                for p in alt_images:
                    if p.exists():
                        self.images_dir = p
                        break

            if not self.captions_path.exists():
                alt_captions = [
                    repo_root / captions_path,
                    repo_root / "data" / "Flick8k Dataset" / "captions.txt",
                    repo_root / "data" / "captions.txt",
                ]
                for p in alt_captions:
                    if p.exists():
                        self.captions_path = p
                        break

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.seq_length = seq_length
        self.seed = seed
        self.augmentation = get_image_augmentation()

    def parse_captions_file(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Parses Flickr8k captions file supporting CSV or tab-separated formats.

        Returns:
            Tuple of (mapping: {image_path: [5 captions]}, all_captions: [str])
        """
        import csv

        if not self.captions_path.exists():
            raise FileNotFoundError(f"Captions file not found: {self.captions_path}")

        caption_mapping: Dict[str, List[str]] = {}
        all_text: List[str] = []

        # Detect delimiter: check if tab-separated
        with open(self.captions_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        is_tsv = "\t" in first_line

        with open(self.captions_path, "r", encoding="utf-8") as f:
            if is_tsv:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    parts = line.split("\t", 1)
                    img_file = parts[0].strip().split("#")[0]
                    raw_caption = parts[1].strip().strip('"')

                    tokens = raw_caption.split()
                    if len(tokens) < 3 or len(tokens) > (self.seq_length - 2):
                        continue

                    full_img_path = str(self.images_dir / img_file)
                    formatted_caption = self.tokenizer.format_caption(raw_caption)

                    if full_img_path not in caption_mapping:
                        caption_mapping[full_img_path] = []

                    caption_mapping[full_img_path].append(formatted_caption)
                    all_text.append(formatted_caption)
            else:
                reader = csv.reader(f)
                header_skipped = False
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    if not header_skipped and ("image" in row[0].lower() and "caption" in row[1].lower()):
                        header_skipped = True
                        continue

                    img_file = row[0].strip().split("#")[0]
                    raw_caption = row[1].strip().strip('"')

                    tokens = raw_caption.split()
                    if len(tokens) < 3 or len(tokens) > (self.seq_length - 2):
                        continue

                    full_img_path = str(self.images_dir / img_file)
                    formatted_caption = self.tokenizer.format_caption(raw_caption)

                    if full_img_path not in caption_mapping:
                        caption_mapping[full_img_path] = []

                    caption_mapping[full_img_path].append(formatted_caption)
                    all_text.append(formatted_caption)

        # Ensure only images with at least 1 caption are retained
        clean_mapping = {
            img: caps for img, caps in caption_mapping.items() if len(caps) >= 1
        }
        return clean_mapping, all_text

    def split_data(
        self,
        caption_mapping: Dict[str, List[str]],
        val_size: float = 0.125,
        test_size: float = 0.125,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Performs deterministic train/val/test split with fixed random seed.

        Default split produces approximately 6000 train / 1000 val / 1000 test
        on standard Flickr8k.
        """
        rng = np.random.default_rng(self.seed)
        all_keys = list(caption_mapping.keys())
        rng.shuffle(all_keys)

        total = len(all_keys)
        n_test = int(total * test_size)
        n_val = int(total * val_size)

        test_keys = all_keys[:n_test]
        val_keys = all_keys[n_test : n_test + n_val]
        train_keys = all_keys[n_test + n_val :]

        train_data = {k: caption_mapping[k] for k in train_keys}
        val_data = {k: caption_mapping[k] for k in val_keys}
        test_data = {k: caption_mapping[k] for k in test_keys}

        return train_data, val_data, test_data

    def create_dataset(
        self,
        data_mapping: Dict[str, List[str]],
        batch_size: int = 64,
        is_training: bool = True,
        num_captions_per_image: int = 5,
    ) -> tf.data.Dataset:
        """Constructs an optimized tf.data pipeline with vectorized prefetching."""
        image_paths = []
        caption_lists = []

        for img_path, captions in data_mapping.items():
            # Pad or truncate captions to match expected num_captions_per_image
            if len(captions) < num_captions_per_image:
                padded = captions + [captions[0]] * (num_captions_per_image - len(captions))
            else:
                padded = captions[:num_captions_per_image]

            image_paths.append(img_path)
            caption_lists.append(padded)

        def _process_sample(img_path, caps):
            img = decode_and_resize(img_path, self.image_size)
            if is_training:
                img = self.augmentation(tf.expand_dims(img, 0), training=True)[0]
                img.set_shape((*self.image_size, 3))
            vectorized_caps = self.tokenizer.text_to_sequence(caps)
            vectorized_caps.set_shape((num_captions_per_image, self.seq_length))
            return img, vectorized_caps

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, caption_lists))
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1024), seed=self.seed)

        dataset = dataset.map(_process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
