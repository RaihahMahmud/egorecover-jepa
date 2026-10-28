Claude Docs can't be created in an Incognito chat because they stay in your account after the chat ends, so I can't make the PDF here. You can open a new chat with Incognito off and I'll build it there. Or the quick option: copy the corrected guide below into any editor, or print this page with Ctrl+P → "Save as PDF".

**What I fixed after re-checking**

1. **Kaggle secrets.** A secret has to be attached to a specific notebook, and a notebook pushed fresh from the CLI may not have it attached. The fix: create the downloader notebook once in Kaggle's web interface, attach the secret there, and have Claude push new versions to that same notebook. Claude should also run a tiny test that only checks whether the secret is readable.
2. **rclone rate limits.** rclone's built-in Google Drive login is shared by all rclone users and gets throttled on large transfers. For hundreds of GB, create your own Google OAuth client ID (rclone's Drive docs explain how) and use it during `rclone config`.
3. **Reading data on GPU machines.** Reading videos one by one from Drive during inference or training is slow. Each GPU job should first copy the files it needs to the Kaggle machine's local disk.
4. **Kaggle notebook settings.** Pushed notebooks need internet (and a GPU where needed) turned on in their metadata file. I've added this to the prompts.

The rest checked out. The Claude Code apt install matches Anthropic's documentation, and the Miniforge, rclone, Kaggle and Colab steps are correct.

---

# V-JEPA 2 / 2.1 EK100 Action Anticipation: Setup and Workflow Guide

## Part 1: Base Ubuntu 22.04 setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget tmux ffmpeg build-essential

# Miniforge (conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh    # accept defaults, then open a new terminal

# Latest rclone
curl https://rclone.org/install.sh | sudo bash
```

If your laptop has an NVIDIA GPU, check it with `nvidia-smi`. Otherwise all GPU work happens on Kaggle or Colab.

## Part 2: Install Claude Code

```bash
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL https://downloads.claude.ai/keys/claude-code.asc \
  -o /etc/apt/keyrings/claude-code.asc
echo "deb [signed-by=/etc/apt/keyrings/claude-code.asc] https://downloads.claude.ai/claude-code/apt/stable stable main" \
  | sudo tee /etc/apt/sources.list.d/claude-code.list
sudo apt update && sudo apt install -y claude-code

claude --version
claude doctor
```

Run `claude` and log in with your Pro account. Claude Desktop on Windows keeps working at the same time.

## Part 3: Connect the services

**Kaggle.** In Kaggle, go to Settings → API, create the legacy API credential (`kaggle.json`), and verify your phone number, which notebooks need for internet access and GPUs.

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install kaggle && kaggle kernels list --mine
```

**Google Drive through rclone.** First create your own Google OAuth client ID by following the Google Drive page of the rclone docs. Then run `rclone config`, create a remote named `gdrive`, enter your client ID and secret, and finish the browser login. Test it with `rclone lsd gdrive:`.

**Kaggle secret for Drive.** In Kaggle's web interface, create a notebook named `ek100-downloader`. Open Add-ons → Secrets, add a secret named `RCLONE_CONF` containing your `~/.config/rclone/rclone.conf`, and attach it to this notebook. Do this by hand, and never paste the token into a chat.

**Colab (optional).** Claude Code sets this up in Phase 2.

**Modal (optional, later).** Run `pip install modal` and then `modal setup`.

## Part 4: Project folder and CLAUDE.md

```bash
mkdir -p ~/vjepa-ek100 && cd ~/vjepa-ek100 && git init
```

Create `CLAUDE.md` in that folder:

```markdown
# Project: V-JEPA 2 / 2.1 action anticipation on EPIC-KITCHENS-100

## Goal

Reproduce EK100 action anticipation with frozen ViT-L backbones and attentive probes.

- V-JEPA 2 ViT-L/16 (300M, 256px). Meta's released EK100 probe reports 32.7 action R@5.
- V-JEPA 2.1 ViT-L/16 (300M, 384px). No released probe assumed; we train our own.
- Target for our own probes: roughly 28-30 action R@5, using the official EK100 anticipation metric (class-mean top-5 recall).

## Key facts

- Official code: facebookresearch/vjepa2 in third_party/vjepa2. Do not edit it; our code goes in src/ and scripts/.
- Each model uses its OWN official eval settings (resolution, normalization, frame sampling). Never share preprocessing between them.
- Dataset stored once in Google Drive (rclone remote "gdrive"), re-encoded to short side >= 384px, original fps kept.
- Google Drive has a daily upload cap of about 750 GB. Plan transfers around it.
- GPU work runs on Kaggle (CLI) or Colab. The laptop is for coding, orchestration and small tests.
- GPU jobs copy the data they need from Drive to the machine's local disk before starting.
- Kaggle notebooks that need Drive push new versions to the existing notebook "ek100-downloader" (or another notebook I created in the web UI), because the RCLONE_CONF secret is attached there.

## Rules

- Never print, log or commit secrets (kaggle.json, rclone config, tokens).
- Ask me before deleting data, starting anything that costs money, or launching jobs longer than about 2 hours.
- Test every pipeline on a tiny subset before running it at scale.
- All long jobs must be resumable (manifests or checkpoints saved to Drive), because Kaggle and Colab sessions time out.
- Record every run in EXPERIMENTS.md: date, model, config, data used, result, notes.
- Commit to git after each completed step.

## Compute rules (laptop is an old CPU-only machine)

- The laptop only runs Claude Code, git, the kaggle CLI and rclone listing/config. No PyTorch, no conda envs, no video downloads, no ffmpeg jobs, no data processing locally.
- All tests, even tiny ones, run in Kaggle notebooks pushed with the kaggle CLI.
- Kaggle CPU notebooks: dataset download and re-encoding. They don't use GPU quota.
- Kaggle GPU notebooks (T4 x2): inference, feature extraction, probe training. Use both GPUs (two parallel workers or data parallel). Kaggle GPU quota is about 30 h/week, so before each GPU job, estimate its hours and tell me.
- Colab (one account, via colab-mcp in my browser): overflow GPU work only, when the Kaggle quota is used up. Every Colab job must checkpoint to Drive often, because free sessions can end without warning.
- One account per service. Never ask for or use other people's accounts.
```

## Part 5: Prompts, phase by phase

Start inside tmux so the session survives a closed terminal: run `tmux new -s vjepa`, then `cd ~/vjepa-ek100 && claude`. Before big phases, press Shift+Tab to switch to plan mode, so Claude shows its plan before acting.

**Phase 0: Environment check**

```
Read CLAUDE.md. Check this laptop's specs without installing anything: CPU model and core count, RAM, free disk space, swap, and CPU temperature if available (sensors or /sys/class/thermal). Based on that, propose which light tasks are safe to run locally (for example editing code, git, kaggle CLI, rclone listing, reading small log files) and which must always go to Kaggle. Add your conclusions to the "Compute rules" section of CLAUDE.md, and wait for my approval before saving.
```

**Phase 1: Official code and config comparison**

```
Clone facebookresearch/vjepa2 into third_party/vjepa2. Read its README, the evals/action_anticipation_frozen code, and the EK100 eval configs. Find the ViT-L EK100 config for V-JEPA 2, and find or derive the equivalent for V-JEPA 2.1 ViT-L.

Write docs/config_diff.md comparing the two: input resolution, frames per clip, frame sampling (fps or frame step), anticipation time, normalization, probe architecture, optimizer, learning rate, epochs, batch size, and which checkpoint and probe files are needed. Mark anything you inferred rather than read directly.

Do NOT install the environment locally. Instead, write a setup cell for Kaggle notebooks that installs the repo's dependencies, and push a tiny Kaggle GPU test notebook that loads the V-JEPA 2 ViT-L checkpoint, runs one dummy clip, and prints the output shape.
```

Review `config_diff.md` carefully. It's a good document to give Gemini for an independent check.

**Phase 2: Colab connection (optional)**

```
Read the README of github.com/googlecolab/colab-mcp and set it up for Claude Code in this project. Tell me exactly what I need to do in my browser.
```

**Phase 3: Data pipeline, validation split first**

```
Build a resumable EK100 download pipeline:
1. Use the official epic-kitchens download scripts and the EK100 annotations repo. Check their READMEs for licence terms and tell me anything I need to accept.
2. Re-encode each video to a short side between 384 and 432px (pick a value and justify it from config_diff.md), keeping the original fps. Use H.264 with quality good enough for 384px crops.
3. Upload each finished video to gdrive:EK100/videos/ with rclone, and keep a manifest on Drive so a restarted job skips completed files.
4. Run it as a Kaggle CPU notebook by pushing new versions to my existing notebook "ek100-downloader" with the kaggle CLI, with internet enabled in kernel-metadata.json. Read the rclone config from the attached Kaggle secret RCLONE_CONF.

First push a tiny test version that only checks the secret is readable and that rclone can list gdrive:, without printing the secret. Then test the re-encoding on 2 videos in a Kaggle CPU notebook and report output size and quality. Check how many CPU notebooks Kaggle lets me run at once, and split the video list across that many parallel workers, each with its own manifest and show me output size and quality. Download ONLY the videos needed for the validation split, and wait for my approval before launching the full validation run.
```

**Phase 4: Key checkpoint, Meta's released probe**

```
Download the V-JEPA 2 ViT-L checkpoint and Meta's released ViT-L EK100 probe. Run validation inference on a Kaggle GPU notebook (GPU and internet enabled in its metadata) using the official eval code. Copy the validation videos from Drive to the notebook's local disk before starting. Report action, verb and noun R@5 with the official metric.

Expected action R@5 is about 32.7. If we are more than about 1.5 points below that, stop and debug the preprocessing (resolution, fps and frame sampling, normalization, annotation timestamps) before anything else. Log the result in EXPERIMENTS.md.
```

Don't move on until this matches. It proves your data pipeline is correct.

**Phase 5: Full dataset download**

```
Extend the download pipeline to the full training split. Respect the Drive daily upload cap, run in resumable Kaggle sessions, and give me a status summary (done / remaining / estimated days) each time I ask.
```

**Phase 6: Measure before caching features**

```
On 100 training clips, extract frozen features (encoder plus predictor outputs, as the anticipation probe uses them) for V-JEPA 2 ViT-L at 256px and V-JEPA 2.1 ViT-L at 384px, in fp16. Report per-clip size, time per clip on the Kaggle GPU, and extrapolated totals for the full training and validation sets. Recommend either caching features in shards on Drive or training from video, with the tradeoffs, including Kaggle local disk limits. Wait for my decision.

Also give a GPU-hour budget for all remaining work (feature extraction for both models plus probe training), split into weekly Kaggle quota blocks, and say what overflow should go to Colab.
```

**Phase 7: Train your own probes**

```
Train the attentive anticipation probe for V-JEPA 2 ViT-L first, then V-JEPA 2.1 ViT-L, following config_diff.md and the approach we chose in Phase 6. Copy each data shard to local disk before use. Save checkpoints to Drive every epoch so training resumes after Kaggle session timeouts. Evaluate on validation with the official metric and log everything in EXPERIMENTS.md, including every deviation from the official protocol.
```

**Phase 8: Supervisor report**

```
Write docs/report.md for my supervisor: setup, a results table (released probe reproduction, our V-JEPA 2 probe, our V-JEPA 2.1 probe) next to the paper numbers, training curves, and an honest list of deviations from the official protocol and their likely effect.
```

## Part 6: Working across several days

When you come back, reattach with `tmux attach -t vjepa` and run `claude --continue`. If you start a fresh session, use this prompt:

```
Read CLAUDE.md, EXPERIMENTS.md, the manifests and the latest logs. Summarize where we are, what's running, and the next step.
```

When you hit Pro's usage limit, Kaggle jobs keep running on their own, so just continue once the limit resets. Approve permission requests one by one instead of turning permission checks off, especially for anything touching Drive. Phases 4 and 6 are where your own review matters most, because a mistake caught there saves days of compute. When the project is finished, revoke the rclone token and delete the `RCLONE_CONF` Kaggle secret.
