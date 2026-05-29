# MaxGPT — Demo Video Script (~2.5 minutes)

The rubric requires a short pre-recorded demo showing the project **works**. This
is a shot-list + narration. Aim for under 3 minutes.

## Before you record
1. Start the web app (from the working setup on your Mac):
   `cd webui && streamlit run app.py`  (it should open in the browser)
2. Make the browser full-screen.
3. Screen-record with mic: **Cmd+Shift+5** → "Record Entire Screen" → make sure
   microphone is on. (Or QuickTime → File → New Screen Recording.)
4. Do one practice run first. Use the **"Reset to demo defaults"** button so
   sampling is consistent (temp 0.8 · top-k 50 · rep 1.2 · 200 tokens).

## Shot list

**0:00–0:20 — Intro** (home screen)
> "This is MaxGPT — four conversational language models I built from scratch in
> PyTorch. No pretrained weights, no transformers library. I wrote the tokenizer,
> the transformer, and the training loop myself. Here's what they can do."

**0:20–1:10 — Compare mode (the money shot)**
- Switch to **Compare** mode. Add all four models (MaxGPT-1, 2, 3, 3.5) for the
  full scaling ladder, or just keep 3.5 vs 3.
- Prompt: `can you give me a recipe for brownies?`
- Narrate as they stream:
> "Same prompt, four models. MaxGPT-1 — 23 million parameters — produces word
> salad. MaxGPT-2 is grammatical but nonsense. MaxGPT-3 writes a real-looking
> recipe. And MaxGPT-3.5, the fine-tuned one, stays on task like an assistant.
> Same architecture, scaled ten times — you can watch coherence emerge."

**1:10–1:55 — Chat mode + multi-turn (the fine-tuning win)**
- Switch to **Chat** mode, model **MaxGPT-3.5**.
- Turn 1: `I adopted a puppy named Rex.`
- Turn 2: `What should I feed him?`
- Narrate:
> "In chat mode it has multi-turn memory. Notice it resolves 'him' back to the
> puppy from my previous message — that cross-turn understanding is exactly what
> the supervised fine-tuning improved."

**1:55–2:20 — Feature callouts**
- Point at the token-by-token streaming, the model picker, an "About" panel.
> "Everything streams token-by-token, generated live by my own model on this
> laptop — no API calls."

**2:20–2:40 — Close**
> "Four models, 23 to 235 million parameters, all trained on a single gaming GPU —
> about ten-thousand times less compute than GPT-3, but demonstrating every
> concept that makes it work. Thanks for watching."

## Tips
- Pick prompts you've tested. If a small model loops or says something odd, that's
  fine — it's honest; you can even say "small models do this."
- Keep it tight; the grader wants to see it **works**, not a feature tour.
- Export and double-check the audio is audible before submitting.
