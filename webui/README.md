# MaxGPT Web UI

Streamlit interface for chatting with the three from-scratch MaxGPT models and
comparing them side-by-side.

## Setup

From the project root, install dependencies into the existing venv:

```bash
pip install -r requirements.txt
```

The UI expects each model's trained files at:

- `maxgpt-1/checkpoints/final.pt` and `maxgpt-1/data/tokenizer.json`
- `maxgpt-2/checkpoints/final.pt` and `maxgpt-2/data/tokenizer.json`
- `maxgpt-3/checkpoints/final.pt` and `maxgpt-3/data/tokenizer.json`

Models with missing files appear in the picker tagged `(no checkpoint)` and
are disabled. The UI re-checks on every page load — just drop the files in
and refresh.

## Run

```bash
cd webui
streamlit run app.py
```

Opens at <http://localhost:8501>.

## Features

- **Chat mode** — single-model conversation with token-by-token streaming.
  Multi-turn history is kept per model (switching models doesn't wipe each
  other's history). Oldest turns are dropped automatically when the context
  window would overflow.
- **Compare mode** — multi-select any 2 or 3 models. Each runs in its own
  thread so you can watch them race; the caption under each column shows
  elapsed time and rough throughput.
- **Sampling controls** in the sidebar — temperature, top-K, repetition
  penalty, max new tokens.
- **Auto device detection** — `cuda` -> `mps` -> `cpu`. The active device is
  shown at the bottom of the sidebar.

## Implementation notes

- Each `maxgpt-N/` folder has identical class names (`Config`, `Transformer`,
  `BPETokenizer`). `app.py` uses `importlib.util` to register each model's
  three files under a namespaced module name (`maxgpt_1.model` vs
  `maxgpt_2.model` ...) so all three coexist in one Python process. See the
  "Import collision handling" section near the top of `app.py`.
- Models are loaded lazily on first selection and cached via
  `@st.cache_resource`, so they stay resident in VRAM after the first use.
- The streaming generator buffers the last few characters of decoded text so
  the `USER:` stop string never flashes on screen before generation cuts off.
