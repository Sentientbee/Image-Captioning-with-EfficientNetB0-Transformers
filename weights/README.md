# Model Weights

The trained weights file (`model_weights.weights.h5`, ~255 MB) is excluded from regular git commits via `.gitignore` to keep the repository lightweight and adhere to GitHub's file size policies.

### Expected Location
```
weights/model_weights.weights.h5
```

### Automatic Path Resolution
The inference engine automatically detects either `weights/model_weights.weights.h5` (Keras 3) or legacy `weights/model_weights.h5` (Keras 2).

### Training Weights
To train your own model weights on Flickr8k or a custom dataset:
```bash
python -m src.train \
    --images_dir path/to/Images \
    --captions_file path/to/captions.txt \
    --epochs 25 \
    --batch_size 64
```
