# EgoRecover-JEPA — Context and Project Handoff to Codex

**Last reconciled:** 2026-10-28. This describes past completed work and the current proposed downloader. It is not evidence that the new downloader has been executed live.

## 1. Research objective and fixed architecture
Working title: **Reliability-Aware V-JEPA for Robust Egocentric Action Anticipation under Transient Visual Degradation** (EgoRecover-JEPA), a CSE senior course project. Research problem: fast head motion and hand-object occlusion corrupt recent visual evidence, while action anticipation predicts a **future** verb/noun pair from an observation window ending one second before the action starts.

Proposed method (still future work): frozen V-JEPA 2.1 encoder E(X_corrupt); estimate per-token reliability q; use a latent predictor / recovery module P to estimate missing reliable embeddings; combine as q*z_obs + (1-q)*z_pred; pass to verb/noun/action anticipation heads. Evaluate directional motion blur, persistent occlusion and both under varied severity, duration and temporal placement. Do **not** equate restoring semantic latents with restoring RGB pixels.

Do not add major architectural modules solely to chase a number. User's aspirational goal is better verb, noun and action Recall@5, potentially approaching 35 mean-class action Recall@5, but no guarantee. This goal does not replace the robustness question. The current task is **dataset download only**.

## 2. Completed Baseline 1 (run baseline1_20261020_091040)
- Dataset: EPIC-KITCHENS-100, 69 annotated source recordings, 7,644 pre-action clips. Custom train 49 recordings / 5,390 clips; held-out 12 recordings / 1,700 clips; eight recordings from official validation / 554 clips. Do not conflate 69 recordings with 69 short video clips. Custom held-out split is recording-disjoint but not confirmed participant-disjoint.
- Feature extractor: community HF port `apiantonio/vjepa2.1-vit-large-384` (ViT-L ~300M), frozen. Each clip is 4 seconds ending 1 second before action start, 16 RGB frames at 4fps, 384x384 center crop, 8 temporal by 36 spatial pooled tokens (288x1024), feature shards cached fp16. Pretrained future predictor was **not** used in this experiment.
- Two downstream probe types: linear mean-pooled token probe; custom attentive probe with two self-attention blocks and three learned verb/noun/action queries. 97 verbs, 300 nouns, 3568 action categories. 30 epochs, batch 64, AdamW LR 1e-3, wd 0.05, 3 random seeds. Custom attentive head is **not an exact reproduction** of Meta's official head.
- Epoch-30 held-out sample R@5 (%): linear 68.0 / 39.0 / 20.4; attentive 67.2 / 38.7 / 19.1 (verb/noun/action). Held-out **mean-class** R@5 (%): linear 18.9 / 19.0 / 7.6; attentive 20.1 / 22.5 / 9.1. Frequency prior sample: 62.6 / 22.8 / 9.7 and mean-class 8.8 / 5.4 / 1.3.
- Eight-video official validation **subset** sample R@5 (%): linear 70.4 / 28.9 / 15.2; attentive 65.2 / 30.5 / 15.8. Mean-class R@5: linear 22.4 / 17.2 / 6.7; attentive 22.9 / 21.0 / 8.8.
- Validation action R@5 peaked around epoch 9 (~21.7 held-out sample, three-seed mean) then dropped by epoch 30 (~19.1), while train loss kept falling; possible overfit/training mismatch. Context ablation (one seed per duration) was token slicing on pretrained full-context representations, not re-running the encoder on shorter source clips.
- Published V-JEPA 2.1 ViT-G result of **40.8 is mean-class action Recall@5** using full paper protocol, 2B model and a 32-frame/8fps encoder+predictor+attentive evaluation pipeline. It is not directly comparable to either of our partial-subset results. Our current pilot does not establish blur robustness; no corrupted-video Baseline 2 has been run.
- CSV of final run metrics is in `results/epoch_metrics_all_runs_full.csv`; recompute metrics from actual CSV for any numerical report.

## 3. Current data acquisition goal
The user has roughly **120 GB of video data** already on their personal Google Drive (self-reported, not re-inventoried). Wants all required EPIC-KITCHENS raw MP4 recordings, possibly 700 GB–1.5 TB of storage depending on file scope and originals, saved into Drive via Kaggle CPU and rclone. **Inventory the actual official train/validation/test videos and source sizes first; do not assert the total is exactly 1.5 TB.** Keep existing `gdrive:EgoRecover-JEPA/EPIC-KITCHENS/Pxx/videos/...` layout; no new duplicate full-video tree. Keep original source MP4 to avoid erasing real blur/compression evidence. 256px (V-JEPA 2) and 384px (2.1) are inference preprocess settings; resize at training/inference, not on raw storage.

Current Kaggle secret: `RCLONE_TOKEN`. Never ask the user to disclose it, never print it. If the raw-auth token mechanism needs repair, audit old notebook implementation first and retain same secret name. Do not assume rclone is connected from the Codex cloud environment. Actual transfer runs must happen on authorized Kaggle with mounted secret and Drive access.

## 4. Code/data artifacts in this handoff
- `notebooks/egorecover_baseline1_original.ipynb`: previous Kaggle notebook (original code, outputs removed from transfer bundle). This is the reference for existing rclone auth and Drive folder conventions.
- `notebooks/EgoRecover_EPIC100_Full_Dataset_Kaggle_rclone.ipynb`: **candidate** six-code-cell download-only notebook. Static checks do **not** mean it has successfully run on Kaggle; treat it as unverified until live inventory and pilot.
- `references/EgoRecover_super_prompt.md`: exhaustive previous assistant project handoff, dataset selection, source pipeline and known problems.
- `references/CSE465_EgoRecover_JEPA_Proposal.pdf`: research architecture and planned experiments.
- `references/Research-related-details-in-portait.pdf`: preliminary brainstorming about blur recovery (contains unverified numbers and proposed scores—do not cite without checking papers).
- `results/epoch_metrics_all_runs_full.csv`: direct authoritative epoch-level metrics.
- `references/25th_OCT_baseline1_progress_update.pptx`: historical progress slides; may contain an imprecise paper-vs-pilot metric comparison.
- `references/prompt_suggestion_for_rayhan.md`: optional broader Claude Code and Kaggle workflow; includes steps inconsistent with immediate download-only scope. Do not adopt the instruction to re-encode the only original MP4 copies.

## 5. Notebook audit to repeat before deployment
Original notebook was designed for download -> decode -> frozen encoder -> cache shard -> train probe, rather than pure raw-video acquisition. It selects a limited video set and may classify a video as done based on `.pt` feature shard. Need full official MP4 inventory and Drive MP4 verification. Inspect upload cleanup: local file must **never** be deleted in a `finally` block if upload might have failed. Inspect background upload futures and rclone statuses. Bound scratch staging in bytes, not just number of videos. Detect partial files and HTTP range consistency. Mark existing remote file as verified only after size/hash checks; keep uncertain files for inspection without overwrite. Use official checksums when supplied. Preserve historical custom held-out split metadata. Avoid assuming all Hugging Face mirrors contain the full dataset. Avoid running GPU / feature extraction / model training in this notebook.

## 6. Immediate next task for Codex
1. Read `AGENTS.md`, this handoff, both notebooks and source PDFs and relevant official metadata.
2. Do a full read-only code audit of download-only notebook: compile every cell, check source inventory, Kaggle secret `RCLONE_TOKEN` compatibility, rclone remote setup, URL validity, hash verification, duplicate avoidance, Xet caching, staging disk, upload cleanup, progress persistence, dry-run, and full official dataset coverage. Never assume a notebook is perfect because an earlier assistant claimed it was tested.
3. Produce a risk table with exact cell/function/line information and a minimal correction plan. DO NOT start remote downloads or modify Drive without approval.
4. Once approved, patch download-only notebook while keeping at most 6 code cells, run offline unit tests, generate safe Kaggle instructions and a 2–3-video pilot. The user runs or explicitly authorizes the pilot on Kaggle. Use Drive as the sole storage destination and only authorized Kaggle accounts. For teammate work partition by stable video IDs; no credentials in notebook.
5. Only after the pilot results are reviewed may you design/approve the large transfer or later training tasks.
