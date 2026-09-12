/**
 * Vision Scribe - Interactive Web Studio Controller
 * Handles UI interactions, API communications, XAI attention scrubbing, and state management.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    selectedFile: null,
    selectedSampleId: 'sample1',
    imageDataUrl: null,
    currentImage: null,
    decodingMode: 'beam',
    beamWidth: 3,
    temperature: 1.0,
    heatmapAlpha: 0.65,
    colormap: 'plasma',
    captionResult: null,
    currentWordIndex: 0,
    isPlaying: false,
    playInterval: null,
  };

  // DOM Elements
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('fileInput');
  const presetsContainer = document.getElementById('presetsContainer');
  const btnGenerate = document.getElementById('btnGenerate');
  const canvas = document.getElementById('attentionCanvas');
  const canvasWordBadge = document.getElementById('canvasWordBadge');
  const captionOutput = document.getElementById('captionOutput');
  const latencyBadge = document.getElementById('latencyBadge');
  const tokenPillsContainer = document.getElementById('tokenPillsContainer');
  const scrubSlider = document.getElementById('scrubSlider');
  const scrubCurrentWord = document.getElementById('scrubCurrentWord');
  const btnPlayScrub = document.getElementById('btnPlayScrub');
  const colormapSelect = document.getElementById('colormapSelect');
  const alphaSlider = document.getElementById('alphaSlider');
  const alphaVal = document.getElementById('alphaVal');
  const beamWidthGroup = document.getElementById('beamWidthGroup');
  const beamWidthSlider = document.getElementById('beamWidthSlider');
  const beamWidthVal = document.getElementById('beamWidthVal');
  const tempSlider = document.getElementById('tempSlider');
  const tempVal = document.getElementById('tempVal');
  const beamCandidatesSection = document.getElementById('beamCandidatesSection');
  const beamCandidatesList = document.getElementById('beamCandidatesList');
  const serverStatusBadge = document.getElementById('serverStatusBadge');
  const weightsStatusBadge = document.getElementById('weightsStatusBadge');

  // Initialize Heatmap Renderer
  const renderer = new HeatmapRenderer(canvas);
  renderer.setColormap(state.colormap);
  renderer.setAlpha(state.heatmapAlpha);

  // 1. Initial Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        serverStatusBadge.innerHTML = `<span class="dot-pulse"></span> FastAPI Online (${data.device})`;
        if (data.weights_loaded) {
          weightsStatusBadge.innerHTML = `<i class="fa-solid fa-check-circle" style="color: #10b981;"></i> Model Weights Loaded (25 Epochs)`;
        } else {
          weightsStatusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color: #f59e0b;"></i> Weights Not Found (Demo Mode)`;
        }
      }
    } catch (err) {
      serverStatusBadge.innerHTML = `<span style="color: #f43f5e;"><i class="fa-solid fa-circle-xmark"></i> Offline</span>`;
    }
  }

  // 2. Load Sample Presets
  async function loadSamplePresets() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) return;
      const samples = await res.json();
      presetsContainer.innerHTML = '';

      samples.forEach((sample, idx) => {
        const card = document.createElement('div');
        card.className = `preset-card ${sample.id === state.selectedSampleId ? 'active' : ''}`;
        card.dataset.id = sample.id;
        card.title = `${sample.name}: ${sample.description}`;

        card.innerHTML = `
          <img src="${sample.url}" alt="${sample.name}" loading="lazy" />
          <div class="preset-tag">${sample.tag}</div>
        `;

        card.addEventListener('click', () => {
          selectSample(sample.id, sample.url);
        });

        presetsContainer.appendChild(card);

        // Preload the default selected sample
        if (idx === 0) {
          selectSample(sample.id, sample.url);
        }
      });
    } catch (err) {
      console.error('Error fetching sample presets:', err);
    }
  }

  function selectSample(id, url) {
    state.selectedSampleId = id;
    state.selectedFile = null;
    if (fileInput) fileInput.value = '';

    // Update active UI cards
    document.querySelectorAll('.preset-card').forEach(card => {
      card.classList.toggle('active', card.dataset.id === id);
    });

    // Load preview onto canvas
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = url;
    img.onload = () => {
      state.currentImage = img;
      state.imageDataUrl = url;
      renderer.setBaseImage(img);
      renderer.setAttentionGrid(null);
      renderer.render();
      canvasWordBadge.textContent = 'Original Input';
      resetCaptionState();
    };
  }

  // 3. Drag and Drop File Upload
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPG, PNG, WebP).');
      return;
    }

    state.selectedFile = file;
    state.selectedSampleId = null;

    // Deselect sample presets
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (event) => {
      state.imageDataUrl = event.target.result;
      const img = new Image();
      img.onload = () => {
        state.currentImage = img;
        renderer.setBaseImage(img);
        renderer.setAttentionGrid(null);
        renderer.render();
        canvasWordBadge.textContent = 'Custom Upload';
        resetCaptionState();
      };
      img.src = state.imageDataUrl;
    };
    reader.readAsDataURL(file);
  }

  // 4. Mode Selection (Beam Search vs Greedy)
  const segButtons = document.querySelectorAll('.seg-btn');
  segButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      segButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.decodingMode = btn.dataset.mode;

      if (state.decodingMode === 'beam') {
        beamWidthGroup.style.display = 'flex';
      } else {
        beamWidthGroup.style.display = 'none';
      }
    });
  });

  // Slider adjustments
  beamWidthSlider.addEventListener('input', (e) => {
    state.beamWidth = parseInt(e.target.value, 10);
    beamWidthVal.textContent = state.beamWidth;
  });

  tempSlider.addEventListener('input', (e) => {
    state.temperature = parseFloat(e.target.value);
    tempVal.textContent = state.temperature.toFixed(1);
  });

  alphaSlider.addEventListener('input', (e) => {
    state.heatmapAlpha = parseFloat(e.target.value);
    alphaVal.textContent = `${Math.round(state.heatmapAlpha * 100)}%`;
    renderer.setAlpha(state.heatmapAlpha);
    updateActiveWordRender();
  });

  colormapSelect.addEventListener('change', (e) => {
    state.colormap = e.target.value;
    renderer.setColormap(state.colormap);
    updateActiveWordRender();
  });

  // 5. Generate Caption API Call
  btnGenerate.addEventListener('click', async () => {
    if (!state.selectedFile && !state.selectedSampleId) {
      alert('Please upload an image or choose a preset sample.');
      return;
    }

    stopScrubPlayback();
    setGeneratingState(true);

    const formData = new FormData();
    if (state.selectedFile) {
      formData.append('file', state.selectedFile);
    } else if (state.selectedSampleId) {
      formData.append('sample_id', state.selectedSampleId);
    }

    formData.append('decoding_mode', state.decodingMode);
    formData.append('beam_width', state.beamWidth);
    formData.append('temperature', state.temperature);

    try {
      const response = await fetch('/api/caption', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }

      const data = await response.json();
      state.captionResult = data;
      renderCaptionResults(data);
    } catch (err) {
      console.error('Caption generation error:', err);
      captionOutput.textContent = `Error: ${err.message || 'Failed to generate caption'}`;
      captionOutput.style.color = '#ef4444';
    } finally {
      setGeneratingState(false);
    }
  });

  function setGeneratingState(isGenerating) {
    if (isGenerating) {
      btnGenerate.disabled = true;
      btnGenerate.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Synthesizing Caption & Attention...`;
      captionOutput.textContent = 'Analyzing image features with EfficientNetB0 and decoding with Transformer...';
      captionOutput.classList.add('skeleton');
      latencyBadge.textContent = 'Processing...';
    } else {
      btnGenerate.disabled = false;
      btnGenerate.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Caption & Heatmap`;
      captionOutput.classList.remove('skeleton');
    }
  }

  function resetCaptionState() {
    stopScrubPlayback();
    state.captionResult = null;
    captionOutput.textContent = 'Click "Generate Caption & Heatmap" above to begin inference.';
    captionOutput.style.color = '';
    latencyBadge.textContent = '0 ms';
    tokenPillsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Token pills will appear here once caption is generated.</span>';
    scrubSlider.max = 0;
    scrubSlider.value = 0;
    scrubCurrentWord.textContent = 'None';
    beamCandidatesSection.style.display = 'none';
    beamCandidatesList.innerHTML = '';
  }

  // 6. Render Results (Typewriter + Token Pills + Beam Candidates)
  function renderCaptionResults(data) {
    captionOutput.style.color = '';
    latencyBadge.textContent = `${data.latency_ms} ms (${data.decoding_mode.toUpperCase()})`;

    // Typewriter effect for main caption
    typewriterEffect(captionOutput, data.caption || 'No caption generated.');

    // Build Word Pills
    buildTokenPills(data.words, data.attention_grids);

    // Render Beam Candidates if Beam Search was used
    if (data.decoding_mode === 'beam' && data.beam_candidates && data.beam_candidates.length > 0) {
      beamCandidatesSection.style.display = 'block';
      beamCandidatesList.innerHTML = '';
      data.beam_candidates.forEach((cand, rank) => {
        const item = document.createElement('div');
        item.className = 'beam-candidate-item';
        item.innerHTML = `
          <div>
            <span class="beam-rank">#${rank + 1}</span>
            <span>${cand.caption}</span>
          </div>
          <span class="beam-score">log-p: ${cand.score}</span>
        `;
        beamCandidatesList.appendChild(item);
      });
    } else {
      beamCandidatesSection.style.display = 'none';
    }

    // Set scrubber max
    if (data.words && data.words.length > 0) {
      scrubSlider.max = data.words.length - 1;
      scrubSlider.value = 0;
      setActiveWord(0);
    }
  }

  function typewriterEffect(element, text) {
    element.textContent = '';
    let i = 0;
    const speed = 25; // ms per char
    const timer = setInterval(() => {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
      } else {
        clearInterval(timer);
      }
    }, speed);
  }

  function buildTokenPills(words, grids) {
    tokenPillsContainer.innerHTML = '';
    if (!words || words.length === 0) return;

    words.forEach((word, index) => {
      const pill = document.createElement('button');
      pill.className = `token-pill ${index === 0 ? 'active' : ''}`;
      pill.textContent = word;
      pill.dataset.index = index;

      pill.addEventListener('click', () => {
        stopScrubPlayback();
        setActiveWord(index);
      });

      tokenPillsContainer.appendChild(pill);
    });
  }

  function setActiveWord(index) {
    if (!state.captionResult || !state.captionResult.words) return;
    const words = state.captionResult.words;
    if (index < 0 || index >= words.length) return;

    state.currentWordIndex = index;
    scrubSlider.value = index;
    scrubCurrentWord.textContent = `"${words[index]}" (${index + 1}/${words.length})`;
    canvasWordBadge.textContent = `Attending: "${words[index]}"`;

    // Highlight active pill
    document.querySelectorAll('.token-pill').forEach((pill, idx) => {
      pill.classList.toggle('active', idx === index);
    });

    updateActiveWordRender();
  }

  function updateActiveWordRender() {
    if (!state.captionResult) {
      renderer.render();
      return;
    }

    const grid = state.captionResult.attention_grids
      ? state.captionResult.attention_grids[state.currentWordIndex]
      : null;

    renderer.setAttentionGrid(grid);
    renderer.render();
  }

  // 7. Scrubber and Playback
  scrubSlider.addEventListener('input', (e) => {
    stopScrubPlayback();
    setActiveWord(parseInt(e.target.value, 10));
  });

  btnPlayScrub.addEventListener('click', () => {
    if (state.isPlaying) {
      stopScrubPlayback();
    } else {
      startScrubPlayback();
    }
  });

  function startScrubPlayback() {
    if (!state.captionResult || !state.captionResult.words || state.captionResult.words.length === 0) return;

    state.isPlaying = true;
    btnPlayScrub.innerHTML = '<i class="fa-solid fa-pause"></i>';
    btnPlayScrub.title = 'Pause attention scrub animation';

    // If at the end, restart from beginning
    if (state.currentWordIndex >= state.captionResult.words.length - 1) {
      setActiveWord(0);
    }

    state.playInterval = setInterval(() => {
      const nextIndex = state.currentWordIndex + 1;
      if (nextIndex < state.captionResult.words.length) {
        setActiveWord(nextIndex);
      } else {
        stopScrubPlayback();
      }
    }, 750);
  }

  function stopScrubPlayback() {
    state.isPlaying = false;
    btnPlayScrub.innerHTML = '<i class="fa-solid fa-play"></i>';
    btnPlayScrub.title = 'Play attention scrub animation';
    if (state.playInterval) {
      clearInterval(state.playInterval);
      state.playInterval = null;
    }
  }

  // Initialize
  checkHealth();
  loadSamplePresets();
});
