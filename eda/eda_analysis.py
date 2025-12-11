import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


DATA_FILES = ["train.csv", "validation.csv", "test.csv"]
OUTPUT_DIR = Path("eda")
FIG_DIR = OUTPUT_DIR / "figures"
SUMMARY_JSON = OUTPUT_DIR / "eda_summary.json"


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV file and treat the first unnamed column as index if present.
    """
    df = pd.read_csv(path)
    # If there is an unnamed index column, drop it
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    return df


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def describe_sequences(df: pd.DataFrame, sample_for_bases: int = 5000) -> dict:
    """
    Compute basic statistics on the nucleotide sequences:
    - length distribution
    - base composition (approximate, on a sample to keep it fast)
    """
    seqs = df["NucleotideSequence"].astype(str)
    lengths = seqs.str.len()

    # Strip potential angle brackets <...>
    inner = seqs.str.strip().str.strip("<>").str.upper()

    if sample_for_bases and len(inner) > sample_for_bases:
        inner_sample = inner.sample(sample_for_bases, random_state=42)
    else:
        inner_sample = inner

    base_counter: Counter = Counter()
    for s in inner_sample:
        base_counter.update(list(s))

    total_bases = sum(base_counter.values()) or 1
    base_freqs = {b: c / total_bases for b, c in base_counter.items()}

    return {
        "length_stats": lengths.describe().to_dict(),
        "base_freqs": base_freqs,
    }


def _svg_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 800,
    height: int = 400,
) -> str:
    """Create a simple SVG bar chart string without external plotting libraries."""
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 50, 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_val = max(values) if values else 1.0
    n = len(values)
    bar_width = chart_width / max(n, 1)

    svg_parts: List[str] = [
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Background
    svg_parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />'
    )

    # Title
    svg_parts.append(
        f'<text x="{width/2}" y="{margin_top/2}" text-anchor="middle" '
        f'font-family="Arial" font-size="16">{title}</text>'
    )

    # Axes
    x0 = margin_left
    y0 = height - margin_bottom
    svg_parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + chart_width}" y2="{y0}" '
        f'stroke="black" stroke-width="1" />'
    )
    svg_parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{margin_top}" '
        f'stroke="black" stroke-width="1" />'
    )

    # Bars and x-labels
    for i, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value / max_val) * chart_height if max_val else 0
        x = x0 + i * bar_width
        y = y0 - bar_height
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" '
            f'height="{bar_height}" fill="#4C72B0" />'
        )

        # X labels (rotated if many)
        text_x = x + bar_width * 0.4
        text_y = y0 + 15
        if n <= 10:
            svg_parts.append(
                f'<text x="{text_x}" y="{text_y}" text-anchor="end" '
                f'font-family="Arial" font-size="10" '
                f'transform="rotate(45 {text_x} {text_y})">{label}</text>'
            )
        else:
            # If many labels, only show every few
            if i % max(1, n // 10) == 0:
                svg_parts.append(
                    f'<text x="{text_x}" y="{text_y}" text-anchor="end" '
                    f'font-family="Arial" font-size="9" '
                    f'transform="rotate(60 {text_x} {text_y})">{label}</text>'
                )

    # Axis labels
    svg_parts.append(
        f'<text x="{width/2}" y="{height - 20}" text-anchor="middle" '
        f'font-family="Arial" font-size="12">{xlabel}</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{(margin_top + y0)/2}" text-anchor="middle" '
        f'font-family="Arial" font-size="12" transform="rotate(-90 20 {(margin_top + y0)/2})">{ylabel}</text>'
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def plot_gene_type_distribution(df: pd.DataFrame, split_name: str) -> None:
    if "GeneType" not in df.columns:
        return
    counts = df["GeneType"].value_counts().sort_values(ascending=False)
    labels = [str(k) for k in counts.index.tolist()]
    values = [float(v) for v in counts.values.tolist()]

    svg = _svg_bar_chart(
        labels=labels,
        values=values,
        title=f"GeneType distribution ({split_name})",
        xlabel="GeneType",
        ylabel="Count",
    )
    out_path = FIG_DIR / f"{split_name}_gene_type_distribution.svg"
    out_path.write_text(svg, encoding="utf-8")


def _length_histogram_bins(lengths: Iterable[int], bins: int = 20) -> Tuple[List[str], List[int]]:
    values = list(lengths)
    if not values:
        return [], []
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        # All lengths equal, single bin
        return [str(min_v)], [len(values)]

    step = (max_v - min_v) / float(bins)
    if step <= 0:
        step = 1.0

    edges = [min_v + i * step for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for v in values:
        # Find bin index
        idx = int((v - min_v) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(bins)]
    return labels, counts


def plot_sequence_length_hist(df: pd.DataFrame, split_name: str) -> None:
    seqs = df["NucleotideSequence"].astype(str)
    lengths = seqs.str.len().tolist()

    labels, counts = _length_histogram_bins(lengths, bins=20)
    if not labels:
        return

    svg = _svg_bar_chart(
        labels=labels,
        values=[float(c) for c in counts],
        title=f"Nucleotide sequence length distribution ({split_name})",
        xlabel="Sequence length bins (characters, including brackets)",
        ylabel="Number of genes",
    )
    out_path = FIG_DIR / f"{split_name}_sequence_length_hist.svg"
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    overall_summary = {}

    for path in DATA_FILES:
        try:
            df = load_dataset(path)
        except FileNotFoundError:
            print(f"Warning: file not found: {path}")
            continue

        split_name = Path(path).stem  # train / validation / test

        split_summary = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
        }

        # Label / GeneType stats
        if "GeneType" in df.columns:
            gt_counts = df["GeneType"].value_counts()
            split_summary["gene_type_counts"] = gt_counts.to_dict()
            split_summary["n_gene_types"] = int(gt_counts.shape[0])

        # Sequence stats and plots
        if "NucleotideSequence" in df.columns:
            seq_stats = describe_sequences(df)
            split_summary["sequence"] = seq_stats

            plot_gene_type_distribution(df, split_name)
            plot_sequence_length_hist(df, split_name)

        overall_summary[split_name] = split_summary

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"EDA summary written to {SUMMARY_JSON}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

