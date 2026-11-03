# Training curves

Each picture is plotted from its run's own log, which sits beside it (converted to UTF-8 text; the training scripts
printed one line every 50 to 500 steps and a train/validation evaluation at intervals).

| Run | Picture | Log | Last evaluation |
|---|---|---|---|
| MaxGPT-1 (23M, 15,000 steps, about 34 minutes) | [maxgpt-1.png](maxgpt-1.png) | [maxgpt-1.log](maxgpt-1.log) | train 3.41, validation 3.44 (step 14,500) |
| MaxGPT-2 (110M, 244,000 steps, about 9.4 hours) | [maxgpt-2.png](maxgpt-2.png) | [maxgpt-2.log](maxgpt-2.log) | train 2.38, validation 2.48 (step 243,500) |
| MaxGPT-3 (235M, 2.44M steps, about 6.6 days) | [maxgpt-3.png](maxgpt-3.png) | [maxgpt-3.log](maxgpt-3.log) | train 2.02, validation 2.13 (step 2,435,000) |
| MaxGPT-3.5 (MaxGPT-3 chat fine-tuned, 300,000 steps) | [maxgpt-3.5.png](maxgpt-3.5.png) | [maxgpt-3.5.log](maxgpt-3.5.log) | train 1.91, validation 1.94 (step 297,500) |

[classic-pretraining.png](classic-pretraining.png) puts the three pretraining runs on one axis (each run's steps as a
fraction of the run): each larger model reaches a lower loss.

All four ran on one RTX 5070 (12 GB). The weights are on Hugging Face: https://huggingface.co/MaxMillion404
