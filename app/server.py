import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import AppConfig, load_config, resolve_weights_path
from src.data.dataset import decode_and_resize
from src.data.tokenizer import CaptionTokenizer
from src.inference.attention_map import AttentionVisualizer
from src.inference.beam_search import BeamSearchGenerator
from src.inference.greedy import GreedyGenerator
from src.models.captioner import ImageCaptioningModel, build_caption_model

# App State
app_state = {
    "cfg": None,
    "model": None,
    "tokenizer": None,
    "beam_gen": None,
    "greedy_gen": None,
    "attn_vis": None,
    "weights_loaded": False,
    "weights_path": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load configuration, tokenizer, and model
    cfg = load_config()
    app_state["cfg"] = cfg

    print(f"[Server] Loading vocabulary from {cfg.paths.vocab_path}...")
    tokenizer = CaptionTokenizer.from_vocab_file(
        cfg.paths.vocab_path,
        max_tokens=cfg.model.vocab_size,
        seq_length=cfg.model.seq_length,
    )
    app_state["tokenizer"] = tokenizer

    print("[Server] Building caption model architecture...")
    model = build_caption_model(
        image_size=cfg.model.image_size,
        channels=cfg.model.channels,
        seq_length=cfg.model.seq_length,
        vocab_size=cfg.model.vocab_size,
        embed_dim=cfg.model.embed_dim,
        ff_dim=cfg.model.ff_dim,
        num_heads_encoder=cfg.model.num_heads_encoder,
        num_heads_decoder=cfg.model.num_heads_decoder,
    )
    app_state["model"] = model

    weights_file = resolve_weights_path(cfg.paths.weights_path)
    if weights_file and weights_file.exists():
        print(f"[Server] Successfully loaded trained weights from {weights_file}")
        model.load_weights(str(weights_file))
        app_state["weights_loaded"] = True
        app_state["weights_path"] = str(weights_file)
    else:
        print("[Server] Warning: Trained weights not found. Operating in demo mode.")
        app_state["weights_loaded"] = False
        app_state["weights_path"] = None

    app_state["beam_gen"] = BeamSearchGenerator(
        model=model,
        tokenizer=tokenizer,
        beam_width=cfg.inference.beam_width,
        max_length=cfg.inference.max_decoded_length,
        length_penalty_alpha=cfg.inference.length_penalty_alpha,
    )
    app_state["greedy_gen"] = GreedyGenerator(
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.inference.max_decoded_length,
    )
    app_state["attn_vis"] = AttentionVisualizer(
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.inference.max_decoded_length,
    )

    yield
    print("[Server] Shutting down.")


app = FastAPI(
    title="Vision Scribe API",
    description="Interactive Image Captioning with EfficientNetB0 and Transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "weights_loaded": app_state["weights_loaded"],
        "weights_path": app_state["weights_path"],
        "vocab_size": len(app_state["tokenizer"]) if app_state["tokenizer"] else 0,
        "device": "GPU" if len(tf.config.list_physical_devices("GPU")) > 0 else "CPU",
    }


@app.get("/api/samples")
async def get_samples():
    """Returns metadata for 1-click sample image presets."""
    return [
        {
            "id": "sample1",
            "name": "Playful Dogs in Grass",
            "description": "Two dogs running and playing outdoors in green grass",
            "tag": "Animals",
            "url": "/static/samples/sample1.jpg",
        },
        {
            "id": "sample2",
            "name": "Children Activity",
            "description": "Children engaged in active outdoor play",
            "tag": "People",
            "url": "/static/samples/sample2.jpg",
        },
        {
            "id": "sample3",
            "name": "Sports & Movement",
            "description": "Action shot capturing athletic activity",
            "tag": "Sports",
            "url": "/static/samples/sample3.jpg",
        },
        {
            "id": "sample4",
            "name": "Outdoor Scene",
            "description": "Scenic landscape with subjects interacting with nature",
            "tag": "Nature",
            "url": "/static/samples/sample4.jpg",
        },
    ]


@app.post("/api/caption")
async def generate_caption(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    decoding_mode: str = Form("beam"),
    beam_width: int = Form(3),
    temperature: float = Form(1.0),
):
    """Generates a caption and extracts cross-attention heatmaps per word token."""
    start_time = time.time()

    # 1. Acquire Image Bytes
    image_bytes = None
    if file and file.filename:
        image_bytes = await file.read()
    elif sample_id:
        sample_path = static_dir / "samples" / f"{sample_id}.jpg"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        with open(sample_path, "rb") as f:
            image_bytes = f.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Please upload an image file or choose a preset sample.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Prepare base64 Data URL for frontend preview
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_b64}"

    # 2. Resize and Preprocess for TensorFlow
    cfg: AppConfig = app_state["cfg"]
    img_resized = pil_img.resize(cfg.model.image_size, Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)

    # 3. Caption Generation (Greedy or Beam Search)
    beam_candidates = []
    if decoding_mode == "beam":
        beam_gen: BeamSearchGenerator = app_state["beam_gen"]
        beam_gen.temperature = max(0.1, min(temperature, 2.0))
        caption, raw_candidates = beam_gen.generate(img_tensor, beam_width=max(1, min(beam_width, 5)))
        beam_candidates = [{"caption": c[0], "score": round(float(c[1]), 3)} for c in raw_candidates]
    else:
        greedy_gen: GreedyGenerator = app_state["greedy_gen"]
        caption = greedy_gen.generate(img_tensor)
        beam_candidates = [{"caption": caption, "score": 1.0}]

    # 4. Cross-Attention Heatmaps Extraction per Word
    attn_vis: AttentionVisualizer = app_state["attn_vis"]
    words, attention_maps, _ = attn_vis.generate_with_attention(img_tensor)

    # If greedy/beam gave caption, ensure words align
    if not words and caption:
        words = caption.split()
        grid_dim = int(np.sqrt(100))
        attention_maps = [np.ones((grid_dim, grid_dim), dtype=np.float32) / (grid_dim * grid_dim) for _ in words]

    # Convert 2D numpy arrays to clean JSON-serializable lists
    attn_grids_json = [arr.tolist() for arr in attention_maps]

    elapsed = round((time.time() - start_time) * 1000.0, 1)

    return JSONResponse(
        {
            "success": True,
            "caption": caption if caption else " ".join(words),
            "words": words,
            "attention_grids": attn_grids_json,
            "beam_candidates": beam_candidates,
            "image_data_url": data_url,
            "image_dimensions": {"width": pil_img.width, "height": pil_img.height},
            "decoding_mode": decoding_mode,
            "beam_width": beam_width,
            "latency_ms": elapsed,
            "weights_loaded": app_state["weights_loaded"],
        }
    )
