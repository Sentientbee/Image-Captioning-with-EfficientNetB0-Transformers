/**
 * Vision Scribe - Attention Heatmap Renderer
 * Bicubic/Bilinear interpolation of 10x10 cross-attention weights onto an HTML5 Canvas.
 * Supports scientific colormaps (Plasma, Viridis, Inferno, Turbo).
 */

class HeatmapRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.baseImage = null;
    this.attentionGrid = null; // 10x10 2D float array
    this.colormapName = 'plasma';
    this.alpha = 0.65; // Overlay blend opacity
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
  }

  setBaseImage(imageElement) {
    this.baseImage = imageElement;
    if (imageElement) {
      this.canvas.width = imageElement.naturalWidth || imageElement.width || 640;
      this.canvas.height = imageElement.naturalHeight || imageElement.height || 480;
      this.offscreenCanvas.width = this.canvas.width;
      this.offscreenCanvas.height = this.canvas.height;
    }
  }

  setAttentionGrid(grid) {
    this.attentionGrid = grid;
  }

  setColormap(name) {
    this.colormapName = name || 'plasma';
  }

  setAlpha(alpha) {
    this.alpha = Math.max(0.0, Math.min(1.0, alpha));
  }

  // Linear interpolation helper
  static lerp(a, b, t) {
    return a + (b - a) * t;
  }

  // Colormap evaluators: maps normalized value t in [0, 1] to [r, g, b]
  static sampleColormap(name, t) {
    const clampedT = Math.max(0, Math.min(1, t));

    if (name === 'viridis') {
      // Perceptually uniform Viridis approximation
      const r = Math.floor(255 * (0.28 + 0.72 * clampedT * Math.sin(clampedT * Math.PI)));
      const g = Math.floor(255 * (0.05 + 0.90 * Math.pow(clampedT, 0.7)));
      const b = Math.floor(255 * (0.35 + 0.65 * (1.0 - clampedT)));
      return [r, g, b];
    } else if (name === 'inferno') {
      // Inferno approximation: Black -> Purple -> Orange -> Yellow
      const r = Math.floor(255 * Math.min(1.0, 1.8 * Math.pow(clampedT, 1.2)));
      const g = Math.floor(255 * Math.max(0.0, 1.4 * Math.pow(clampedT, 2.2) - 0.1));
      const b = Math.floor(255 * Math.max(0.0, 0.8 * Math.sin(clampedT * Math.PI) * (1 - clampedT * 0.5)));
      return [r, g, b];
    } else if (name === 'turbo') {
      // Turbo rainbow approximation
      const r = Math.floor(255 * Math.sin(clampedT * Math.PI * 0.9 + 0.2));
      const g = Math.floor(255 * Math.sin(clampedT * Math.PI * 0.9 + 1.2));
      const b = Math.floor(255 * Math.sin(clampedT * Math.PI * 0.9 + 2.4));
      return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
    } else {
      // Default: Vibrant Plasma (Deep Blue -> Magenta -> Sunset Orange -> Golden Yellow)
      const r = Math.floor(255 * (0.05 + 0.95 * Math.pow(clampedT, 0.65)));
      const g = Math.floor(255 * (0.02 + 0.90 * Math.pow(clampedT, 1.8)));
      const b = Math.floor(255 * (0.55 * (1.0 - clampedT) + 0.2 * Math.sin(clampedT * Math.PI)));
      return [r, g, b];
    }
  }

  // Bilinear interpolation on a 2D grid
  sampleGridBilinear(grid, normX, normY) {
    const rows = grid.length;
    const cols = grid[0].length;

    const gx = normX * (cols - 1);
    const gy = normY * (rows - 1);

    const x0 = Math.floor(gx);
    const x1 = Math.min(cols - 1, x0 + 1);
    const y0 = Math.floor(gy);
    const y1 = Math.min(rows - 1, y0 + 1);

    const fx = gx - x0;
    const fy = gy - y0;

    const top = HeatmapRenderer.lerp(grid[y0][x0], grid[y0][x1], fx);
    const bottom = HeatmapRenderer.lerp(grid[y1][x0], grid[y1][x1], fx);

    return HeatmapRenderer.lerp(top, bottom, fy);
  }

  render() {
    const { ctx, canvas, baseImage, attentionGrid, colormapName, alpha } = this;
    if (!ctx || !canvas) return;

    // 1. Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Draw Base Image
    if (baseImage && baseImage.complete && baseImage.naturalWidth > 0) {
      ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // 3. If no attention grid provided, finish with original image
    if (!attentionGrid || !attentionGrid.length || alpha <= 0.01) {
      return;
    }

    // 4. Find min and max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < attentionGrid.length; r++) {
      for (let c = 0; c < attentionGrid[r].length; c++) {
        const val = attentionGrid[r][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const valRange = Math.max(1e-6, maxVal - minVal);

    // 5. Render smoothly interpolated heatmap onto an offscreen canvas at lower resolution then scale up
    // For crisp real-time performance (60 FPS during scrubbing), interpolate onto 160x120 grid
    const heatW = 160;
    const heatH = 120;
    if (this.offscreenCanvas.width !== heatW || this.offscreenCanvas.height !== heatH) {
      this.offscreenCanvas.width = heatW;
      this.offscreenCanvas.height = heatH;
    }
    const heatCanvas = this.offscreenCanvas;
    const heatCtx = this.offscreenCtx;
    const imgData = heatCtx.createImageData(heatW, heatH);
    const data = imgData.data;

    for (let y = 0; y < heatH; y++) {
      const normY = y / (heatH - 1);
      for (let x = 0; x < heatW; x++) {
        const normX = x / (heatW - 1);
        const rawVal = this.sampleGridBilinear(attentionGrid, normX, normY);
        const normVal = (rawVal - minVal) / valRange;

        // Apply non-linear contrast stretch to emphasize peaks
        const contrastT = Math.pow(normVal, 1.35);

        const [r, g, b] = HeatmapRenderer.sampleColormap(colormapName, contrastT);
        const idx = (y * heatW + x) * 4;

        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        // Non-linear alpha: higher attention regions have more opacity
        data[idx + 3] = Math.floor(255 * (0.2 + 0.8 * contrastT));
      }
    }

    heatCtx.putImageData(imgData, 0, 0);

    // 6. Draw smoothly scaled heatmap with user alpha and overlay blend
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(heatCanvas, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }
}

window.HeatmapRenderer = HeatmapRenderer;
