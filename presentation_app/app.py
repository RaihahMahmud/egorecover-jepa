from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLED_DATA = APP_DIR / "demo_data"
DEFAULT_RESULTS = os.environ.get("EGO_JEPA_RESULTS_DIR", str(BUNDLED_DATA))
DEFAULT_MEDIA = os.environ.get("EGO_JEPA_MEDIA_DIR", "")

st.set_page_config(
    page_title="EgoRecover JEPA | EK100 demonstration",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 1.4rem; padding-bottom: 2rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px; color: #f8fafc;
              background: linear-gradient(115deg, #0f172a 0%, #173a5e 58%, #256b73 100%);}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
      .hero p {margin: .25rem 0; color: #dbeafe;}
      .scope {padding: .8rem 1rem; border-left: 5px solid #e59c28; background: #fff8e8;
              border-radius: 8px; color: #47330f; margin: .7rem 0 1rem 0;}
      .small-note {color: #64748b; font-size: .9rem;}
      div[data-testid="stMetric"] {background: #f8fafc; padding: .8rem 1rem; border-radius: 12px;
                                    border: 1px solid #e2e8f0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return default


def read_first_json(root: Path, names: list[str], fallback: Any) -> Any:
    for name in names:
        value = read_json(root / name)
        if value is not None:
            return value
    return fallback


def safe_child(root: Path, relative: str) -> Path | None:
    """Resolve a user-provided relative media path without allowing path traversal."""
    try:
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        candidate.relative_to(resolved_root)
        return candidate if candidate.is_file() else None
    except (OSError, ValueError):
        return None


def as_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def normalize_metrics(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for head in ("verb", "noun", "action"):
        item = value.get(head, {})
        if not isinstance(item, dict):
            continue
        result[head] = {
            "top5_accuracy_percent": item.get("top5_accuracy_percent"),
            "class_mean_recall_at5_percent": item.get("class_mean_recall_at5_percent"),
            "examples": item.get("examples"),
            "classes_seen": item.get("classes_seen"),
        }
    return result


def normalize_history(training_log: Any, csv_path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if csv_path.is_file():
        try:
            frame = pd.read_csv(csv_path)
            if not frame.empty:
                records = frame.to_dict(orient="records")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            pass

    if not records and isinstance(training_log, list):
        records.extend(row for row in training_log if isinstance(row, dict))
    elif not records and isinstance(training_log, dict):
        for key in ("records", "training_log", "history", "loss_records"):
            candidate = training_log.get(key)
            if isinstance(candidate, list):
                records.extend(row for row in candidate if isinstance(row, dict))

    if not records:
        return pd.DataFrame(columns=["epoch", "video_id", "mean_loss", "status"])

    frame = pd.DataFrame(records)
    aliases = {
        "epoch": ("epoch", "Epoch"),
        "video_id": ("video_id", "video", "Video"),
        "mean_loss": ("mean_loss", "loss", "combined_loss", "mean combined loss"),
        "status": ("status",),
        "clips_updated": ("clips_updated", "clips", "num_clips"),
    }
    for target, options in aliases.items():
        if target not in frame.columns:
            source = next((column for column in options if column in frame.columns), None)
            if source is not None:
                frame[target] = frame[source]
    if "epoch" in frame:
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    if "mean_loss" in frame:
        frame["mean_loss"] = pd.to_numeric(frame["mean_loss"], errors="coerce")
    frame = frame.dropna(subset=[column for column in ("epoch", "mean_loss") if column in frame.columns])
    if "status" in frame:
        frame = frame[frame["status"].astype(str).isin(["completed", "complete", "nan", "None", ""])]
    if "video_id" not in frame:
        frame["video_id"] = [f"record {i + 1}" for i in range(len(frame))]
    return frame.reset_index(drop=True)


def draw_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    boxes = [
        (0.25, 1.45, 2.05, 1.55, "EPIC-100 clip\n32 frames • 8 fps\n384 × 384", "#e7f0fb"),
        (2.85, 1.45, 2.35, 1.55, "Frozen V-JEPA 2.1\nViT-L/384 encoder\n+ predictor", "#d8e9ff"),
        (5.75, 1.45, 2.2, 1.55, "Meta attentive pooler\n3 learned queries\n4 blocks • 16 heads", "#e7e1ff"),
        (8.5, 1.45, 1.8, 1.55, "Trainable\nclassification\nheads", "#e0f2e9"),
        (10.85, 2.65, 1.8, 1.0, "Verb\n97 classes", "#fff1d6"),
        (10.85, 1.45, 1.8, 1.0, "Noun\n289 classes", "#fff1d6"),
        (10.85, 0.25, 1.8, 1.0, "Action\n3,568 classes", "#fff1d6"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color="#152238", zorder=3)
    for y in (2.95, 1.95, 0.95):
        pass
    arrows = [((2.30, 2.22), (2.82, 2.22)), ((5.20, 2.22), (5.72, 2.22)), ((7.95, 2.22), (8.47, 2.22)),
              ((10.3, 2.22), (10.68, 3.15)), ((10.3, 2.22), (10.68, 1.95)), ((10.3, 2.22), (10.68, 0.75))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#52657b"})
    ax.text(3.98, 1.12, "Frozen", ha="center", color="#315c89", fontsize=9, weight="bold")
    ax.text(9.38, 1.12, "Updated during probe training", ha="center", color="#26734d", fontsize=9, weight="bold")
    fig.tight_layout()
    return fig


def display_prediction_panel(example: dict[str, Any]) -> None:
    gt = example.get("ground_truth", {}) if isinstance(example.get("ground_truth"), dict) else {}
    pred = example.get("predictions", {}) if isinstance(example.get("predictions"), dict) else {}
    rows = []
    for head in ("verb", "noun", "action"):
        gt_label = str(gt.get(head, "—"))
        value = pred.get(head, [])
        if isinstance(value, str):
            top = [value]
        elif isinstance(value, list):
            top = [str(x.get("label", x.get("class", ""))) if isinstance(x, dict) else str(x) for x in value]
        elif isinstance(value, dict):
            top = [str(value.get("label", value.get("class", "—")))]
        else:
            top = []
        top = [x for x in top if x and x != "—"]
        hit = gt_label in top
        rows.append({
            "Head": head.capitalize(),
            "Ground truth": gt_label,
            "Top prediction": top[0] if top else "Not provided",
            "In shown top-k": "Yes" if hit else ("No" if top else "—"),
            "Other shown predictions": ", ".join(top[1:]) if len(top) > 1 else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.sidebar.title("Demo controls")
results_dir_text = st.sidebar.text_input(
    "Run results folder",
    value=st.session_state.get("results_dir", DEFAULT_RESULTS),
    help="Select the folder containing manifest.json, metrics.json, and training_log.json. It can be your local copy of the Drive presentation folder.",
)
st.session_state["results_dir"] = results_dir_text
results_dir = Path(results_dir_text).expanduser()
if not results_dir.exists():
    results_dir = BUNDLED_DATA
media_dir_text = st.sidebar.text_input(
    "Optional demo clips folder",
    value=st.session_state.get("media_dir", DEFAULT_MEDIA),
    help="Use short local MP4 excerpts. Keep video files out of GitHub.",
)
st.session_state["media_dir"] = media_dir_text
media_dir = Path(media_dir_text).expanduser() if media_dir_text else APP_DIR / "demo_clips"

manifest = read_first_json(results_dir, ["manifest.json", "presentation_summary.json"], {})
metrics = normalize_metrics(read_first_json(results_dir, ["metrics.json"], read_json(BUNDLED_DATA / "metrics.json", {})))
training_log = read_first_json(results_dir, ["training_log.json"], read_json(BUNDLED_DATA / "training_log.json", []))
validation_log = read_first_json(results_dir, ["validation_log.json"], read_json(BUNDLED_DATA / "validation_log.json", []))
examples_doc = read_first_json(
    results_dir,
    ["demo_examples.json"],
    read_json(APP_DIR / "user_data" / "demo_examples.json", read_json(APP_DIR / "demo_examples_template.json", {"examples": []})),
)
examples = examples_doc.get("examples", []) if isinstance(examples_doc, dict) else []
if not isinstance(examples, list):
    examples = []
history = normalize_history(training_log, results_dir / "training_loss_records.csv")

run_name = str(manifest.get("run_name", "vjepa21_ek100_sunday_milestone_v1"))
train_info = manifest.get("training", {}) if isinstance(manifest.get("training"), dict) else {}
val_info = manifest.get("validation", {}) if isinstance(manifest.get("validation"), dict) else {}
protocol = manifest.get("protocol", {}) if isinstance(manifest.get("protocol"), dict) else {}
train_videos = train_info.get("selected_train_videos", 12)
train_clips = train_info.get("selected_train_actions", 1179)
val_videos = val_info.get("selected_val_videos", 8)
val_clips = val_info.get("selected_val_actions", 340)
epochs = train_info.get("target_epochs", 3)

st.markdown(
    f"""
    <div class="hero">
      <h1>EgoRecover-JEPA</h1>
      <p>EPIC-KITCHENS-100 • V-JEPA 2 and 2.1 • attentive-probe demonstration</p>
      <p><b>Run:</b> {run_name}</p>
    </div>
    <div class="scope"><b>Subset experiment:</b> this demonstration covers {val_clips} validation clips from {val_videos} videos. It is not a full EK100 benchmark result.</div>
    """,
    unsafe_allow_html=True,
)

overview_tab, video_tab, metrics_tab, training_tab, pipeline_tab = st.tabs(
    ["Overview", "Video showcase", "Metrics", "Training", "Architecture & protocol"]
)

with overview_tab:
    st.subheader("What was run")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train subset", f"{train_clips:,} clips", f"{train_videos} videos")
    c2.metric("Validation subset", f"{val_clips:,} clips", f"{val_videos} videos")
    c3.metric("Probe training", f"{epochs} epochs", "Meta attentive classifier")
    c4.metric("Backbone", "Frozen", "V-JEPA 2.1 ViT-L/384")

    st.markdown("#### Work completed")
    st.markdown(
        """
        - Loaded the released V-JEPA 2.1 ViT-L/384 encoder and predictor.
        - Kept the representation model frozen and trained Meta's attentive probe on an EPIC-100 subset.
        - Resumed training from checkpoints across Kaggle sessions and synced run artifacts to Drive.
        - Evaluated the trained probe on the selected validation videos and saved per-head Recall@5 metrics.
        """
    )
    st.info("This app displays saved outputs. It does not start training or load the multi-gigabyte foundation checkpoint.")
    status = manifest.get("status", "run artifacts loaded")
    st.caption(f"Run status in manifest: {status} · Results folder: {results_dir}")

with video_tab:
    st.subheader("Predictions on selected validation clips")
    st.caption("The app shows precomputed predictions and local video excerpts. It does not perform live model inference.")
    if examples:
        titles = [str(row.get("title", row.get("example_id", f"Example {i + 1}"))) for i, row in enumerate(examples)]
        choice = st.selectbox("Choose an example", range(len(examples)), format_func=lambda i: titles[i])
        example = examples[choice]
        clip_relative = str(example.get("clip_file", example.get("video_path", "")))
        clip_path = safe_child(media_dir, clip_relative) if clip_relative else None
        if clip_path is None and clip_relative:
            clip_path = safe_child(APP_DIR / "user_media", Path(clip_relative).name)
        if clip_path:
            st.video(str(clip_path))
        else:
            st.warning(f"Video excerpt not found: {clip_relative or 'no clip_file set'}. Add it to the clips folder or upload an example below.")
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**Video:** {example.get('video_id', '—')}  \n**Narration:** {example.get('narration_id', '—')}")
            st.markdown(f"**Ground-truth action:** {example.get('ground_truth', {}).get('action', '—') if isinstance(example.get('ground_truth'), dict) else '—'}")
        with right:
            if example.get("observation_start") is not None or example.get("observation_end") is not None:
                st.markdown(f"**Observed window:** {example.get('observation_start', '?')}–{example.get('observation_end', '?')} s")
            if example.get("anticipation_gap_seconds") is not None:
                st.markdown(f"**Gap before target:** {example['anticipation_gap_seconds']} s")
        display_prediction_panel(example)
        if example.get("note"):
            st.caption(str(example["note"]))
    else:
        st.info("No per-clip examples are included in the run metrics. Add a few short excerpts and their saved probe predictions below; no training rerun is needed.")

    with st.expander("Add a presentation clip and its saved predictions"):
        st.caption("Upload a short excerpt that you are allowed to use, then enter the ground truth and predictions from your saved inference output. This form records labels; it does not run the model.")
        uploaded_clip = st.file_uploader("Short MP4 excerpt", type=["mp4", "mov", "webm"], key="clip_upload")
        title = st.text_input("Example title", placeholder="Example 1 — verb and noun correct")
        video_id = st.text_input("EPIC video ID", placeholder="P08_17")
        narration_id = st.text_input("Narration ID (optional)")
        gt_cols = st.columns(3)
        gt_verb = gt_cols[0].text_input("Ground-truth verb")
        gt_noun = gt_cols[1].text_input("Ground-truth noun")
        gt_action = gt_cols[2].text_input("Ground-truth action")
        p_cols = st.columns(3)
        pred_verb = p_cols[0].text_input("Top verb prediction")
        pred_noun = p_cols[1].text_input("Top noun prediction")
        pred_action = p_cols[2].text_input("Top action prediction")
        topk_cols = st.columns(3)
        top_verb = topk_cols[0].text_input("Other top verbs (comma-separated)")
        top_noun = topk_cols[1].text_input("Other top nouns (comma-separated)")
        top_action = topk_cols[2].text_input("Other top actions (comma-separated)")
        window_cols = st.columns(3)
        obs_start = window_cols[0].text_input("Observed start time (s, optional)")
        obs_end = window_cols[1].text_input("Observed end time (s, optional)")
        gap = window_cols[2].text_input("Target gap (s, optional)", value="1.0")
        if st.button("Save example locally", type="primary", disabled=uploaded_clip is None):
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.strip()).strip("_") or "clip_example"
            local_media = APP_DIR / "user_media"
            local_media.mkdir(parents=True, exist_ok=True)
            extension = Path(uploaded_clip.name).suffix.lower() or ".mp4"
            clip_name = f"{slug}{extension}"
            (local_media / clip_name).write_bytes(uploaded_clip.getvalue())
            local_data = APP_DIR / "user_data"
            local_data.mkdir(parents=True, exist_ok=True)
            data_file = local_data / "demo_examples.json"
            existing = read_json(data_file, {"examples": []})
            if not isinstance(existing, dict) or not isinstance(existing.get("examples"), list):
                existing = {"examples": []}
            def pred_list(first: str, rest: str) -> list[str]:
                return [item.strip() for item in [first, *rest.split(",")] if item.strip()]
            item = {
                "example_id": slug,
                "title": title.strip() or slug,
                "video_id": video_id.strip(),
                "narration_id": narration_id.strip(),
                "clip_file": clip_name,
                "ground_truth": {"verb": gt_verb, "noun": gt_noun, "action": gt_action},
                "predictions": {
                    "verb": pred_list(pred_verb, top_verb),
                    "noun": pred_list(pred_noun, top_noun),
                    "action": pred_list(pred_action, top_action),
                },
                "observation_start": obs_start or None,
                "observation_end": obs_end or None,
                "anticipation_gap_seconds": gap or None,
                "note": "Precomputed output from the saved attentive probe; this app does not run inference.",
            }
            existing["examples"].append(item)
            data_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            st.success(f"Saved clip and labels locally: {clip_name}. The files are excluded from Git by this app's ignore rules.")
            st.rerun()

with metrics_tab:
    st.subheader("Validation results — V-JEPA 2.1 subset")
    st.caption("Class-mean Recall@5 is averaged across classes represented in this selected subset. It is not the full EK100 validation benchmark.")
    metric_rows = []
    for head in ("verb", "noun", "action"):
        item = metrics.get(head, {})
        metric_rows.append({
            "Head": head.capitalize(),
            "Class-mean Recall@5 (%)": item.get("class_mean_recall_at5_percent"),
            "Top-5 accuracy (%)": item.get("top5_accuracy_percent"),
            "Validation clips": item.get("examples"),
            "Classes represented": item.get("classes_seen"),
        })
    metric_frame = pd.DataFrame(metric_rows)
    st.dataframe(metric_frame, use_container_width=True, hide_index=True)
    if metric_frame["Class-mean Recall@5 (%)"].notna().any():
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        valid = metric_frame.dropna(subset=["Class-mean Recall@5 (%)"])
        bars = ax.bar(valid["Head"], valid["Class-mean Recall@5 (%)"], color=["#3978a8", "#7d68a6", "#df9636"][:len(valid)])
        ax.set_ylabel("Class-mean Recall@5 (%)")
        ax.set_ylim(0, max(10, float(valid["Class-mean Recall@5 (%)"].max()) * 1.25))
        ax.set_title("V-JEPA 2.1 attentive-probe results on the subset")
        ax.grid(axis="y", alpha=.22)
        for bar, value in zip(bars, valid["Class-mean Recall@5 (%)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .25, f"{float(value):.2f}%", ha="center", va="bottom", weight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Context from the earlier V-JEPA 2 run")
    st.warning("The V-JEPA 2 values below are from the supplied full-validation chart (9,248 clips / 138 videos); they are context only, not an apples-to-apples comparison with this 340-clip V-JEPA 2.1 subset.")
    ref = read_json(BUNDLED_DATA / "vjepa2_reference.json", {})
    if ref:
        ref_rows = []
        for head in ("Action", "Verb", "Noun"):
            ref_rows.append({
                "Head": head,
                "V-JEPA 2 code window (%)": ref.get("code_window", {}).get(head.lower()),
                "V-JEPA 2 paper window (%)": ref.get("paper_window", {}).get(head.lower()),
                "Meta reported (%)": ref.get("meta_reported", {}).get(head.lower()),
            })
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    st.caption("Reference values were read from the teammate chart shared in this project; they are not recomputed by this app.")

with training_tab:
    st.subheader("Training progress and saved loss records")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Epoch target", str(epochs))
    t2.metric("Selected training videos", str(train_videos))
    t3.metric("Selected training clips", f"{train_clips:,}")
    t4.metric("Frozen backbone", "Yes")

    st.markdown("#### Training actions")
    st.markdown(
        """
        1. Stage one source video at a time from Drive to limit temporary storage use.
        2. Sample the configured video clip and pass it through the frozen V-JEPA 2.1 encoder and predictor.
        3. Update the Meta attentive probe using the combined verb, noun, and action sigmoid-focal objective.
        4. Save and sync probe, optimizer, scaler, and resume cursor checkpoints so a Kaggle session can continue.
        5. Run validation on the selected videos and save per-head top-5 metrics.
        """
    )

    if not history.empty and {"epoch", "mean_loss"}.issubset(history.columns):
        st.markdown("#### Saved combined loss records")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for epoch, group in history.groupby("epoch", dropna=True):
            x = np.arange(1, len(group) + 1)
            ax.plot(x, group["mean_loss"].to_numpy(), marker="o", markersize=3, linewidth=1.6, label=f"Epoch {int(epoch)}")
        ax.set_xlabel("Saved completed-video record within epoch")
        ax.set_ylabel("Mean combined focal loss")
        ax.set_title("Saved training-loss records (available records only)")
        ax.grid(alpha=.22)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        epoch_counts = history.groupby("epoch").size().rename("saved video records").reset_index()
        st.dataframe(epoch_counts, use_container_width=True, hide_index=True)
        st.warning("Cell 3 reset its in-memory log on resumed sessions. This chart reflects only records present in the saved JSON/CSV; it may not cover every epoch or every training update.")
    else:
        st.warning("No structured combined-loss records were found in the selected results folder. Check that training_log.json or training_loss_records.csv is present. The checkpoint does not contain a historical loss curve.")
    st.caption("Separate per-head loss curves and per-optimizer-step history were not saved by this run.")

with pipeline_tab:
    st.subheader("Model and data path")
    st.pyplot(draw_architecture(), use_container_width=True)
    st.markdown("#### Run configuration")
    config_rows = [
        ("Backbone", "V-JEPA 2.1 ViT-L/384; encoder and predictor frozen"),
        ("Probe", train_info.get("probe", "Meta AttentiveClassifier: 4 blocks, 16 heads, 3 learned queries")),
        ("Clip input", f"{protocol.get('frames', 32)} frames at {protocol.get('fps', 8)} fps · {protocol.get('resolution', 384)} px"),
        ("Train subset", f"{train_videos} videos · {train_clips} clips · {epochs} epochs"),
        ("Validation subset", f"{val_videos} videos · {val_clips} clips"),
        ("Validation sampler note", protocol.get("val_sampler_anchor", "See run manifest for sampler timing")),
        ("Code revision", manifest.get("repository_commit", "See run manifest")),
    ]
    st.dataframe(pd.DataFrame(config_rows, columns=["Item", "Run detail"]), use_container_width=True, hide_index=True)
    st.info("The run manifest describes validation sampling as anchored at the action stop timestamp minus one second. Treat this as the implemented sampler behavior, and verify the selected example's actual timestamps before calling the demo strict action anticipation.")

    with st.expander("Show loaded run manifest"):
        st.json(manifest)
    with st.expander("Show validation-video coverage log"):
        st.json(validation_log)

st.divider()
st.caption("EgoRecover-JEPA presentation app · local artifact viewer · no videos, checkpoints, tokens, or model weights are uploaded by the app.")

