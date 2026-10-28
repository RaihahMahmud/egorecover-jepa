# SUPER PROMPT: continue the EgoRecover-JEPA project (CSE 465 / "499A") from where the previous assistant left off

You are taking over as my technical assistant on a university research project. A previous assistant (Claude) worked with me for a long time. Everything below is the verified state of the project. Read all of it before answering anything. Follow the RULES section strictly.

---------------------------------------------------------------------------------------------------
## 0. RULES FOR YOU (very important)
1. **Never invent or guess results, numbers, paper details, or file contents.** If something is not in this prompt or in files I upload, say "I don't know / I can't verify" and tell me how to check it.
2. Mark every claim as one of: VERIFIED (from my CSV/figure/log or a paper I uploaded), FROM CODE (what the code does), PLAN (not done yet), or HYPOTHESIS (untested explanation). Never present a hypothesis as a finding.
3. When I ask for numbers, recompute them from the CSV files I upload instead of trusting this prompt blindly. If they differ, tell me and use the files.
4. My professor asks questions about the code, so explanations must be accurate and in **plain, easy words** (I present to a class; avoid jargon or define it in one short phrase).
5. I do not have unlimited compute: Kaggle (free GPU, a few hours per session) is my main place to run code. Give complete, copy-paste-ready code cells, keep notebooks to fewer than 10 code cells, and make sure results are saved to Google Drive.
6. Be honest about limitations. A candid "in progress" is better than an inflated claim.
7. Do not use figures that contain numbers that are not in my data (an earlier set of AI-generated figures had invented numbers and was rejected).
8. Never ask me to paste or reveal secret tokens. (My Drive is accessed through an rclone token stored as a notebook secret named RCLONE_TOKEN. Do not include it in any output.)

---------------------------------------------------------------------------------------------------
## 1. THE PROJECT
**Title:** Reliability-Aware V-JEPA for Robust Egocentric Action Anticipation under Transient Visual Degradation (working name **EgoRecover-JEPA**). Course project proposal dated Oct 14, 2026 (I call the course CSE 499A; the proposal PDF says CSE 465). Group of 3 students. 8-week plan, about 1 to 2 weeks in.

**Problem.** First-person (egocentric) video has fast head motion, hands covering objects, and blur. For action anticipation (predict the next verb-noun action from video that ends before the action starts), the last frames before the action matter most and are often the least reliable.

**Idea.** Do not repair pixels. Use V-JEPA's latent predictor to guess the missing meaning (in feature space) of unreliable frames from the surrounding context.

**Planned method (from the proposal):**
- Frozen V-JEPA 2.1 encoder: z_obs = E(X_corrupted).
- Lightweight reliability head: confidence q in [0,1] per spatio-temporal token.
- Treat low-confidence tokens as missing; recover them with the V-JEPA predictor: z_hat = P(z_reliable, M).
- Reliability-gated fusion: z* = q * z_obs + (1 - q) * z_hat.
- Action anticipation head predicts verb / noun / action.
- Training loss: L = L_action + l1*L_latent (cosine to clean features, stop-gradient) + l2*L_cons (clean vs corrupted consistency) + l3*L_rel (reliability supervised by the known synthetic corruption mask).
- Only the reliability head, recovery/fusion module, and anticipation head are trained; the backbone stays frozen.

**Dataset/benchmark:** EPIC-KITCHENS-100 action anticipation (97 verbs, 300 nouns, 3,568 verb-noun action pairs in the official train labels). Anticipation time 1 s.

**Corruptions planned:** motion blur, hand-object occlusion (masks that persist across frames), and both together; vary severity (weak/medium/strong), duration (0.25, 0.5, 1.0, 1.5 s) and temporal proximity (early in the window vs right before the anticipation point). Robustness metric: drop in Recall@5 = clean - corrupted.

**The 8 baselines/comparisons:**
1. Frozen V-JEPA 2.1 on clean video  <-- DONE (this is all that is done)
2. Frozen V-JEPA 2.1 on corrupted video
3. Corruption augmentation without latent recovery
4. Mask/remove unreliable frames or tokens without prediction
5. A reconstruction-based video representation baseline (if compute allows)
6. Latent recovery without reliability gating
7. Full EgoRecover-JEPA with learned reliability gating
8. Oracle corruption mask (upper bound; tells whether detection or recovery is the bottleneck)

**4 research questions (none answered yet; all need corrupted video):** (1) how much do blur/occlusion hurt anticipation, especially right before the action; (2) are V-JEPA latents more tolerant than reconstruction-based representations; (3) can reliability-aware latent prediction recover meaning from context; (4) does latent recovery beat augmentation, masking, or pixel restoration. Future work in proposal: gaze as an extra, possibly missing, input (EGTEA Gaze+).

---------------------------------------------------------------------------------------------------
## 2. WHAT WE DID, IN ORDER (history)
1. **First code (Colab).** A frozen V-JEPA 2.1 + attentive-probe pipeline for baseline 1. Encoder = community Hugging Face port `apiantonio/vjepa2.1-vit-large-384` (ViT-L, 300M params, 384 px). The official torch.hub path for 2.1 was reported broken at the time.
2. **Colab problems.** Pasting a script into a cell made argparse fail; the GPU runtime ran out; Drive could not be mounted from a different Google account ("credential propagation was unsuccessful"). Solution: move to **Kaggle**, access Google Drive with **rclone** using a token stored as a secret (`RCLONE_TOKEN`, generated with `rclone authorize "drive"` on my PC). The same notebook also supports Colab (detects the environment).
3. **One-video pilot.** First video P01_01 (about 6.2 GB). The official downloader (single connection, no resume) took about 3 hours. An earlier copy on Drive was a half-finished `.part` file, which broke decoding. We added size + md5 checks (md5.csv from the official download scripts). Pilot numbers (one video, tens of clips) are NOT meaningful and must not be reported as results.
4. **Speed/efficiency fixes.** Reading frames with decord from a 6 GB file was slow, so we switched to ONE ffmpeg decoding pass per video; the frozen encoder runs once per clip and features are cached; T4 GPU reports bf16 support but emulates it slowly, so fp16 is used; a pandas version error appeared after `pip install -U pandas` mid-session (fix: do not upgrade pandas under a running kernel; the loss-report cell builds its table with numpy instead of `pivot`).
5. **Scale-up notebook** (8 code cells; details in section 4): fast data pipeline, 69 videos, feature shards per video on Drive, three-seed training, comparisons, ablations, figures, report, Drive backup. Later added a "save and verify" cell and a "loss report" cell.
6. **Kaggle usability problems.** Output panes sometimes stayed blank (kernel busy/queued behind another cell, or lost browser connection; a stdout-copying wrapper `Tee` was suspected but not proven). Advice: do not "Restart session" (it wipes variables); check with a tiny `print("alive")` cell and read `console_log.txt`; the save cell prints a timestamped line at every step and writes SAVE_STATUS.txt to Drive.
7. **Presentation.** 10-slide PowerPoint (`baseline1_progress_update.pptx`) built only from real figures and CSV numbers, plus a 3-member speaking script (also saved as a Claude Doc).
8. **Paper comparison.** I uploaded the V-JEPA 2.1 paper (arXiv 2603.14482v3). We compared methodology; key result: the paper's 40.8 is **mean-class Recall@5** (see section 6). This exposed that slide 10 and the speaking script compare 40.8 with our PLAIN Recall@5 (19.1); they should compare with our MEAN-CLASS action number (9.1). **That correction has NOT been applied yet.**

---------------------------------------------------------------------------------------------------
## 3. BASELINE 1: METHOD (what the code actually does)
- **Sample for each labeled action:** a 4 s window that ends tau_a = 1 s before the action starts; 16 frames at 4 fps (frames on a 4 fps grid from one decoding pass); short side resized to 384, centre crop 384x384 (this cuts the left/right edges of wide video); ImageNet mean/std normalization.
- Actions starting earlier than 5 s in the video are dropped (no full window; also avoids looking into the action).
- **Encoder:** frozen V-JEPA 2.1 ViT-L (HF port), fp16. 16 frames at 384 px give 8 time steps x 24x24 patches = 4,608 tokens of 1,024 numbers. Spatial average-pooling by 4 gives 8 x 6x6 = **288 tokens per clip** (time-major order: time_step*36 + spatial). Features cached as fp16.
- **Trained part (probes), only on cached features:**
  - *Attentive probe (baseline 1):* LayerNorm + linear 1024->768, 2 self-attention transformer blocks (12 heads, MLP 4x, GELU, pre-norm), then cross-attention with 3 learned query vectors (verb/noun/action), LayerNorm, dropout 0.1, three linear heads (97 / 300 / 3,568). Roughly 20M trainable parameters (my estimate; the code does not print it).
  - *Linear probe (comparison):* mean over tokens, LayerNorm, three linear heads.
- **Loss:** CE(verb) + CE(noun) + CE(action, only where the action label is known; pairs not in the official train labels get label -1, are skipped in the loss, and count as misses in metrics).
- **Optimization:** AdamW, lr 1e-3, weight decay 0.05, batch 64, 30 epochs, one-cycle LR schedule (10% warm-up), fp16 autocast + GradScaler, gradient clipping 1.0, seeds 0/1/2. No augmentation, no tuning, no early stopping, no checkpoint selection (final epoch reported).
- **Evaluation:** Recall@5 and mean-class Recall@5 (recall per class, then averaged over classes present in the evaluation set) for verb/noun/action; evaluated after EVERY epoch on both validation sets.
- **Extra tests:** chance; frequency prior (always guess the 5 most common train classes); linear probe (3 seeds); context-length ablation (probe sees only the last 0.5/1/2 s of the 4 s window by slicing tokens; one seed each; 4 s = main run); top-k curves (k=1..10); per-class and per-video recall; 95% bootstrap intervals over clips (seed 0 only; treats clips as independent, so likely too narrow).

---------------------------------------------------------------------------------------------------
## 4. DATA AND NOTEBOOK STRUCTURE
**Data facts (verified from official CSVs):** EPIC_100_train.csv has 67,217 actions in 495 videos (32 participants); EPIC_100_validation.csv has 9,668 actions in 138 videos; all 138 official validation videos are 2018-era videos hosted only on the slow Bristol server. Extension videos (3-digit ids like P06_104) are mirrored on Hugging Face: `awsaf49/epic_kitchens_100` (fast).

**Splits actually used (VERIFIED from my data summary):**
| split | videos | clips | feature size |
|---|---|---|---|
| train | 49 | 5,390 | 3.18 GB |
| val_heldout (whole videos from the HF mirror, never trained on) | 12 | 1,700 | 1.00 GB |
| val_official (official validation split, from Bristol) | 8 | 554 | 0.33 GB (3.8% of its action pairs unseen in train) |
Total 69 videos, 7,644 clips, 4.5 GB features. Train clips are about 8% of the official training actions. Train data includes P01_01 as one extra train video.
**Selection logic:** extension videos 3 to 15 min long, at least 20 usable clips, chosen round-robin across participants (shortest first); every 5th is held out (split by VIDEO to avoid leakage); 8 official validation videos; shard files named `<split>__<video>.pt` containing features [N,288,1024] fp16 + labels/times; a video with an existing shard keeps its original split.

**Notebook cells (all Kaggle/Colab compatible; secret RCLONE_TOKEN; Internet ON; GPU for cell 4+):**
1. Setup: config constants, logging (Tee to console_log.txt), pip installs, rclone + Drive checks, `sync_run()` backup helper.
2. Data plan: reads official CSVs (cloned from GitHub), label maps (ACTION_MAP etc.), video selection, Drive inventory (resume), `plan.csv`.
3. Library: resumable multi-connection HTTP Range downloader (+ speed probe, md5 check), source router (hf / bristol / drive_raw / drive_copyid), one-pass ffmpeg decoding, `clip_index`, encoder loading/encoding (tries input layouts until the token count matches), `process_video`, `upload`.
4. Run: parallel downloads + GPU encoding + upload of shards and raw videos to Drive; resumable; time budget; failure log.
5. Training + evaluation of the attentive probe (3 seeds, every epoch logged).
6. Comparisons (chance, frequency prior, linear), context ablation, top-k, per-class, per-video, bootstrap CIs.
7. Figures (fig01..fig09). 8. Report (`report.md`), summary JSON, final backup.
Extra cells: "save & verify" (backs up probes/state, checks sizes against Drive, writes SAVE_STATUS.txt, builds a zip), and "loss report" (loss tables/images from the epoch history).
**Drive layout:** `EgoRecover-JEPA/features/<model tag>/` (shards), `EPIC-KITCHENS/P*/videos/` (raw videos), `runs/<run id>/` (tables, figures, probes, logs, report), `exports/` (zips). Environment last used: Kaggle, Tesla T4, Python 3.13, torch 2.11, transformers 5.20.
**CSV outputs (in runs/<id>/tables and loss_report):** plan, data_summary, epoch_history_attentive, epoch_history_all (= epoch_metrics_all_runs_full.csv, 270 rows = 9 runs x 30 epochs; columns tag, seed, k_steps, epoch, train_loss, secs, plus `<valset>/<head>_r5|mc5`), comparison_long and comparison_<valset>, ablation_context_length, topk_curve, per_class_top15, per_video, bootstrap_ci, loss_by_epoch_all_runs, epoch_summary_attentive_mean_over_seeds. Run tags: attentive (seeds 0-2), linear_meanpool (seeds 0-2), context_1/2/4 (= last 0.5/1/2 s, seed 0).

---------------------------------------------------------------------------------------------------
## 5. RESULTS (VERIFIED, final run `baseline1_20261020_091040`; values read from my CSV/figures)
**Held-out videos, Recall@5 % (3-seed mean ± std, epoch 30):**
| method | verb | noun | action |
|---|---|---|---|
| chance | 5.2 | 1.7 | 0.1 |
| frequency prior | 62.6 | 22.8 | 9.7 |
| linear probe | 68.0 ± 0.2 | 39.0 ± 0.1 | 20.4 ± 0.1 |
| attentive probe (baseline 1) | 67.2 ± 2.6 | 38.7 ± 2.8 | 19.1 ± 1.9 |
**Mean-class Recall@5 %:** frequency prior 8.8 / 5.4 / 1.3; linear 18.9 / 19.0 / 7.6; attentive 20.1 ± 1.7 / 22.5 ± 0.4 / 9.1 ± 0.8.
**Official validation (8 videos, 554 clips), Recall@5:** prior 65.2 / 15.3 / 9.4; linear 70.4 / 28.9 / 15.2; attentive 65.2 ± 2.9 / 30.5 ± 0.8 / 15.8 ± 1.7. Mean-class: prior 13.9 / 6.7 / 2.2; linear 22.4 / 17.2 / 6.7; attentive 22.9 ± 2.5 / 21.0 ± 0.7 / 8.8 ± 0.9.
**Findings (all clean-video only):**
- Frozen features beat chance and a frequency prior, clearly on mean-class recall. (Plain verb Recall@5 is dominated by common classes: the prior already scores 62.6.)
- Linear probe equals or beats the attentive probe on plain Recall@5; the attentive probe is better on mean-class Recall@5 (e.g. noun 22.5 vs 19.0, action 9.1 vs 7.6 held-out; action 8.8 vs 6.7 official).
- **Overfitting:** the attentive probe's training loss keeps falling (epoch 1 about 14.5; final 1.9 / 2.9 / 6.7 for seeds 0/1/2, so seeds differ a lot; linear probe about 4.2) while held-out recall peaks around epoch 9-10 (3-seed mean: verb 71.4 at ep 10 -> 67.2; noun 42.9 at ep 9 -> 38.7; action 21.7 at ep 9 -> 19.1) then declines. About 18 s/epoch for the attentive probe on a T4.
- **Context ablation (seed 0, held-out):** with only the last 0.5 s: verb 67.4, noun 41.5, action 20.6; with 1 s: 65.9 / 37.1 / 19.3; the 2 s probe is similar. Probes seeing 0.5-2 s are competitive with or better than the full 4 s (one seed per setting; fewer tokens also means less overfitting, so this is suggestive, not proof; it is consistent with the project premise that the last moments matter).
- Nouns/actions are lower on the official validation videos than held-out. Verb Recall@10 about 79% (seed 0). Seed-0 numbers (used by CIs/top-k/per-class/per-video) are lower than the 3-seed means, e.g. held-out seed-0 Recall@5 about 65.1 / 35.8 / 16.9.
- Per-video scores vary a lot (e.g. verb from about 35% to about 78%). Frequent classes (take, put-down, rinse) do well; some rare classes score near 0.
- Two runs of the notebook gave identical linear/context results but slightly different attentive results (e.g. verb 65.8 vs 67.2) -> some GPU nondeterminism. The earlier run id is `...20261019_161348`; the final CSV/figures are from `...20261020_091040`.

---------------------------------------------------------------------------------------------------
## 6. COMPARISON WITH THE V-JEPA 2.1 PAPER (I uploaded the PDF; verified from it)
- Paper (Table 5): **mean-class Recall@5** on the EK100 validation set: ViT-G (2B) verb/noun/action = 64.3 / 59.9 / **40.8**; ViT-g (1B) 63.6 / 56.2 / 38.4; V-JEPA 2 ViT-g 63.6 / 57.1 / 39.7; older InAViT (160M) 51.9 / 52.0 / 25.8. **Ours (mean-class): 20.1 / 22.5 / 9.1 held-out; 22.9 / 21.0 / 8.8 official.** So the fair comparison is mean-class: we are about 4x lower on action and about 3x lower on verb. Plain Recall@5 (our 19.1) is not what the paper reports.
- Paper protocol (Sec. 3.2, App. C.6): frozen encoder AND predictor; context of 32 frames at 8 fps (4 s) at 384x384, ending 1 s before the action; predictor gets mask tokens for the frame 1 s in the future; predictor and encoder outputs are concatenated along the token dimension and fed to an attentive probe (4 transformer blocks: 3 self-attention + final cross-attention, 16 heads, three learned queries each with its own linear classifier); focal loss (alpha 0.25, gamma 2) per head, summed; during probe training anticipation time is randomly sampled in 0.25-1.75 s and an anticipation point in 0-0.25. For classification tasks they train ~20 heads at once (5 learning rates x 4 weight decays, cosine schedule) and report the best; EK100-specific epochs/learning rate were not in the text I read (they say they follow V-JEPA 2's protocol).
- **Differences (what we missed vs the paper):** (1) no predictor features (encoder only); (2) tokens pooled 16x in space (288 vs full) and 4 fps instead of 8 fps (16 vs 32 frames); (3) probe recipe: cross-entropy instead of focal loss, no LR/WD search, fixed 1 s anticipation, no randomization, 2 vs 3 self-attention blocks, 12 vs 16 heads; (4) data: 49 videos / 5,390 clips (about 8% of the 67,217 train actions) and the classes seen are limited (from our class-frequency plot roughly 77 of 97 verbs and about 170 of 300 nouns appear in our training clips, so unseen classes score 0 in mean-class recall; counts read from a figure, so approximate); (5) smaller encoder (ViT-L; the paper reports EK100 only for ViT-g/G, and I could not confirm our community weights are the paper's distilled ViT-L); (6) evaluation on 12 held-out + 8 official videos, not the full 138-video validation set.
- The paper's own ablation: going ViT-g -> ViT-G adds about 2.4 action points (38.4 -> 40.8) and 256 -> 384 px adds about 2.7 for ViT-g, so encoder size/resolution alone probably do not explain a ~30-point gap (but no ViT-L EK100 number exists to confirm).
- **We have NOT isolated any cause of the gap.** Most plausible contributors (HYPOTHESIS): missing predictor features, small data/class coverage, untuned probe, pooling.

---------------------------------------------------------------------------------------------------
## 7. KNOWN LIMITATIONS AND THINGS NOT TO CLAIM
- Reduced-data pilot; numbers are not comparable to published results. Seeds only change probe initialization/batch order, so std understates real uncertainty. Official validation = only 8 videos. Bootstrap intervals use seed 0 and assume independent clips.
- Do NOT claim: that we reproduced V-JEPA results; that our method makes the model more robust (nothing corrupted has been tested); that short context is "better" (only "competitive", one seed); any download-speed number (we never recorded new speeds; the notebook only logs them if it downloads); any cause for the paper gap.
- Unknowns to check: whether the HF encoder port includes predictor weights (and whether the official repo `facebookresearch/vjepa2` can load 2.1 predictors); the paper's exact EK100 probe epochs/learning rate (V-JEPA 2 paper/code); the exact official mean-class definition; whether our port equals the distilled ViT-L.

---------------------------------------------------------------------------------------------------
## 8. OPEN TASKS (suggested priority; I may change it)
0. **Fix slide 10 and the speaking script** to compare 40.8 with our mean-class action 9.1 / 8.8 (not 19.1), and note the paper metric is mean-class; add "predictor features not used".
1. Check predictor availability. If unavailable, fallback: train a small token predictor on cached tokens (mask some tokens, predict them from visible ones).
2. Stabilize baseline 1: early stopping near 10 epochs, regularization (dropout, label smoothing, token dropout, smaller probe), focal loss, random anticipation time, learning-rate/weight-decay grid, chosen on a DEV set of training videos (keep held-out + official for the final report); paired comparisons later.
3. Build corruption generators (motion blur, persistent hand-shaped occlusion, both) for the last frames; visualize clean vs corrupted before any run; durations that align with 0.5 s encoder steps (0.5/1/1.5 s) plus 0.25 s.
4. Baseline 2 pilot: clean-trained probe on corrupted held-out clips (about 300 clips, one severity), measure the Recall@5 drop; go/no-go for the full sweep. Corrupted clips require re-decoding raw videos (they are backed up on Drive).
5. Scale data (more of the ~37k usable extension clips); consider a compact feature variant for RAM/disk; add a data-scaling curve.
6. Later: reliability head, latent recovery, gated fusion (baselines 6-7), oracle-mask test, latent cosine metric, optional gaze.

---------------------------------------------------------------------------------------------------
## 9. FILES I WILL UPLOAD (ask for them if not present)
Proposal PDF (CSE465_EgoRecover_JEPA_Proposal), V-JEPA 2.1 paper PDF (2603_14482v3.pdf), the scale-up notebook (baseline1_scaleup_kaggle_colab.ipynb), epoch_metrics_all_runs_full.csv and other tables, figures fig01-fig09 + loss images, the PPTX, and the speaking script. Treat the CSVs/figures as the source of truth.

## 10. HOW TO START
First, in 6-8 lines, restate the project status and the 3 biggest open issues in your own words (to confirm you understood). Then ask me which task from section 8 to do first. Use plain words, mark every statement as VERIFIED / FROM CODE / PLAN / HYPOTHESIS when it matters, and keep code cells complete and tested before you hand them over.
