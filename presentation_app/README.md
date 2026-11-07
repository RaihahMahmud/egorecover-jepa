# EgoRecover-JEPA presentation app

A local Streamlit app for presenting the EPIC-KITCHENS-100 V-JEPA 2 / V-JEPA 2.1 subset experiments. It brings together:

- A clear subset-experiment summary and training/validation counts
- Saved Recall@5 and top-5 accuracy values for verb, noun, and action
- Contextual V-JEPA 2 values transcribed from the teammate's supplied full-validation figure, clearly marked as non-comparable
- A training-process summary and a plot of only the loss records that are actually present
- An architecture diagram and run protocol details
- A video showcase that can play local short clips and display ground truth plus saved probe predictions

The app is a **viewer**, not a training runner. It does not load V-JEPA model weights or perform new model inference. No source videos, checkpoints, credentials, or Drive tokens belong in this GitHub folder.

## Run locally on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:EGO_JEPA_RESULTS_DIR = "C:\path\to\results of training experiment\vjepa21_ek100_sunday_milestone_v1"
$env:EGO_JEPA_MEDIA_DIR = "C:\path\to\short_demo_clips"
streamlit run app.py
```

The results folder should contain the loose run files from Drive, especially `manifest.json`, `metrics.json`, `training_log.json`, `training_loss_records.csv`, and `validation_log.json`. If `EGO_JEPA_RESULTS_DIR` is not set, the app loads the bundled summary values. The original detailed artifacts remain your source of truth.

## Video examples

The current run's aggregate validation log contains per-video counts, not per-clip logits. Therefore the app does not invent video predictions. To create a presentation example, add a short permitted excerpt and the matching saved top predictions in the **Video showcase** tab. The upload form writes clip files under `user_media/` and metadata under `user_data/`; these paths are excluded from Git.

Alternatively, create `demo_examples.json` in the results folder using the schema in `demo_examples_template.json`, and put the local clip files in the folder pointed to by `EGO_JEPA_MEDIA_DIR`. A prediction may be a label string or a list of top-k label strings. Include only outputs that came from the trained probe.

## Accuracy and scope labels

- The V-JEPA 2.1 results shown by default are from 340 clips in 8 selected validation videos.
- The older V-JEPA 2 chart is from the supplied project image and covers 9,248 clips in 138 videos; it is shown as context only.
- The stored loss history can be incomplete because the training notebook reset its in-memory log after resume. The app labels this and never fills gaps with fabricated values.
- The manifest describes the implemented validation sampler as anchoring at the action stop timestamp minus one second. Verify the timestamps of each selected clip before presenting it as strict action anticipation.

## Suggested short demo order

1. Overview: subset scope and completed work.
2. Video showcase: one success, one partial result, and one miss, if those examples are available.
3. Metrics: subset class-mean Recall@5 and the contextual V-JEPA 2 chart values.
4. Training and architecture: frozen backbone, attentive probe, checkpoint/resume, and honest loss-history coverage.

Keep this repository private and do not commit the original dataset videos or user-specific credentials.

