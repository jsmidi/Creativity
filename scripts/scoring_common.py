"""Shared loading and list parsing for response-level evaluation."""
from pathlib import Path
import re
import pandas as pd
from experiment import PROTOCOL_VERSION, CONDITION_TEXT, canonical_item

# Scored tables keep pairing and validity; full request details stay in the raw CSV.
RESPONSE_COLUMNS = (
    "Response_ID", "Run_ID", "Model", "Task", "Item", "Split", "Condition",
    "Paraphrase", "Repeat", "Block_ID", "Generation_Seed", "Temperature",
    "Top_P", "Top_K", "Intervention", "Alpha", "Random_Vector", "Artifact", "Scope",
    "Response", "Status", "Error", "Finish_Reason",
)


def parse_numbered_list(text):
    """Extract newline-delimited numbered entries, allowing optional Markdown bold markers."""
    if not isinstance(text, str):
        return []
    entries = re.findall(r"(?:^|\n)\s*(?:\*\*)?(\d+)[.)](?:\*\*)?\s+([^\n]+)", text)
    return [value.strip().strip("*").strip() for _, value in entries]


def parse_dat_words(text):
    # A phrase is invalid, not silently converted to its first word.
    """Return lowercase single-word entries; reject phrases instead of taking their first word."""
    return [word.lower() for word in parse_numbered_list(text)
            if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", word)]


def load_responses(input_paths, task, include_legacy=False):
    """Read raw task CSVs, exclude scored exports, validate protocol and reject duplicate response IDs.

    Legacy inputs require explicit opt-in; raw source paths and failure status are retained."""
    files = sorted({file.resolve() for path in input_paths
                    for file in (path.rglob("*.csv") if path.is_dir() else [path])})
    frames = []
    for path in files:
        frame = pd.read_csv(path, keep_default_na=False)
        if not {"Task", "Response", "Model", "Condition", "Item"}.issubset(frame.columns):
            continue
        if "Parsed_Ideas" in frame or "Parsed_Words" in frame:
            continue  # Do not accidentally ingest scored exports.
        if "Protocol" not in frame:
            # New self-contained runs have a recognizable schema, without a sidecar.
            compact = {"Run_ID", "Response_ID", "Instruction", "Generation_Seed",
                       "Prompt", "Max_Tokens", "Top_P", "Top_K", "Timestamp_UTC"}.issubset(frame)
            if not compact and not include_legacy:
                continue
            frame["Protocol"] = PROTOCOL_VERSION if compact else "legacy"
        if not include_legacy:
            frame = frame[frame["Protocol"] == PROTOCOL_VERSION].copy()
        frame = frame[frame["Task"] == task].copy()
        if frame.empty:
            continue
        frame["Source_File"] = str(path)
        if "Response_ID" not in frame:
            frame["Response_ID"] = [f"{path.name}:{i}" for i in frame.index]
        if "Status" not in frame:
            frame["Status"] = frame["Response"].apply(
                lambda x: "error" if str(x).lstrip().startswith("Error:") else "ok" if str(x).strip() else "empty")
            if "Error" in frame:
                frame.loc[frame["Error"].astype(str).str.strip().ne(""), "Status"] = "error"
        if "Block_ID" not in frame and "Generation_Seed" in frame:
            frame["Block_ID"] = frame["Item"].map(canonical_item) + "|seed=" + frame["Generation_Seed"].astype(str)
        keep = [column for column in (*RESPONSE_COLUMNS, "Protocol", "Source_File") if column in frame]
        frames.append(frame[keep])
    if not frames:
        raise ValueError("No matching raw responses. Legacy data require --include-legacy.")
    result = pd.concat(frames, ignore_index=True)
    if result["Response_ID"].duplicated().any():
        raise ValueError("Duplicate response IDs: select each raw run only once.")
    if result["Protocol"].nunique() > 1:
        raise ValueError("Do not pool legacy and v2 prompts. Evaluate them separately.")
    return result


def descriptive_plot(frame, metrics, output):
    """Response means with +/- one sample SD, not inferential confidence intervals."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    if "Response_ID" in frame and frame["Response_ID"].duplicated().any():
        raise ValueError("Plot one row per response; aggregate AUT ideas before plotting.")
    present = frame["Condition"].dropna().unique().tolist()
    order = [condition for condition in CONDITION_TEXT if condition in present]
    order += [condition for condition in present if condition not in order]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6), squeeze=False)
    for ax, (column, title) in zip(axes[0], metrics.items()):
        valid = frame.dropna(subset=[column])
        if valid.empty:
            ax.text(.5, .5, "No available scores", ha="center")
        else:
            sns.barplot(data=valid, x="Model", y=column, hue="Condition",
                        hue_order=order, errorbar="sd", capsize=0.12, ax=ax)
            ax.tick_params(axis="x", rotation=25)
        ax.set_title(title + "\nMean +/- 1 SD across responses (not a confidence interval)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
