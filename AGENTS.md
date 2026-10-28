# Codex Instructions — EgoRecover-JEPA

## Read before editing
Read `PROJECT_HANDOFF.md` and the original proposal first. The project must **not** be redesigned without explicit approval. The active task is a **download-only Kaggle notebook**, not model training.

## Unchanged research decisions
- Primary task: EPIC-KITCHENS-100 future verb/noun/action anticipation; retain a 4-second observation window ending 1 second before the next action.
- Existing Baseline 1: frozen V-JEPA 2.1 ViT-L 384 model, 16 RGB frames sampled at 4 fps, 384x384 preprocessed input, spatial pooling to 288 tokens, custom 2-self-attention-block / three-query attentive probe plus mean-pooled linear baseline.
- Proposed architecture: frozen V-JEPA encoder -> reliability estimator -> latent recovery/predictor -> reliability-gated fusion -> anticipation heads. The reliability/recovery components are planned, not implemented. Preserve the distinction between plan and completed work.
- V-JEPA 2 and 2.1 may require different pre-processing (256 vs 384 resolution). Download **original MP4s once**; never degrade the original video to 256 or 384 as the only stored copy. The 256/384 values describe model input pipelines, not the source-video download format.
- Do not train or change models during the download phase.

## Kaggle and storage constraints
- Work on Kaggle CPU for downloading; Google Drive via rclone. No local workstation commands required or recommended.
- User already has a Kaggle secret called `RCLONE_TOKEN`; do not request or display it. Never print OAuth tokens, configs, cookies or secrets.
- Preserve existing raw MP4s at `gdrive:EgoRecover-JEPA/EPIC-KITCHENS/...` and all existing feature/result files; current Drive inventory must be checked live rather than assumed from previous run.
- Never overwrite, rename, move or delete a Drive file without permission. Never delete the staging file until its upload is verified. Do not mark existing unverified files as validated.
- Use official dataset manifests, URL checks, per-video expected sizes and hashes when available. Record unverified/missing sources honestly. Handle sessions, disk caps, daily Drive upload cap and resume failures.
- Default should be read-only inventory/dry run; require approval to launch pilot or bulk transfer. Keep the new Kaggle notebook to **six Python code cells maximum**.
- If sharing a code notebook with teammates to execute on their Kaggle quota, distribute only notebook code; each collaborator should use permitted account access to the user's Drive. Never share the owner's `RCLONE_TOKEN` or personal rclone configuration. Keep deterministic nonoverlapping work partitions and shared verified manifest, if feasible.

## Evaluation integrity
- Never claim observed progress without actual Kaggle/Drive output.
- Official EPIC validation must never be included in training. Preserve Baseline 1 recording-separated custom held-out split for reproducibility.
- Report sample Recall@5 and **mean-class** Recall@5 separately. Paper 40.8 is mean-class action R@5 for the full-benchmark ViT-G configuration; current pilot action mean-class R@5 is 9.1 held-out (three-seed mean), 8.8 on an eight-video official validation subset. Not directly comparable.
- Mark findings as VERIFIED, FROM CODE, PLAN or HYPOTHESIS when scientifically relevant. No invented benchmark results or imagined upload speeds.

## Quality and approval
- Audit notebooks first, report bugs with exact cells/functions, propose minimal fixes, then edit code.
- Validate Python syntax, notebook structure and pure helper routines locally/offline before user-visible delivery. Real Kaggle/rclone access requires user-run pilot or explicitly authorized tools.
- Do not start downloading hundreds of GB, connect to accounts, use paid services, publish datasets to GitHub, or modify a remote account without explicit user approval.
- Never add video files, source data, Drive credentials or private access tokens to this repository. The GitHub project contains notebook code, documentation and small metrics only.
