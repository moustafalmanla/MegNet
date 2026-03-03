#!/usr/bin/env python3
"""Simple wrapper for MEGnet-neuro end-to-end ICA labeling and cleanup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as op
import re
import sys
from pathlib import Path

import matplotlib
import mne
import numpy as np
from scipy import signal


def _bootstrap_package_root() -> None:
    """Allow running this file directly from any CWD.

    When executed as `python /path/to/easy_megnet.py`, Python only adds that
    script directory to `sys.path`. We add the repo/package root (the parent
    that contains the `MEGnet/` package directory) so absolute imports work.
    """
    if __package__:
        return

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "MEGnet").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_bootstrap_package_root()

from MEGnet.megnet_init import main as megnet_init
from MEGnet.prep_inputs.ICA import classify_ica
from MEGnet.prep_inputs.ICA import load_data
from MEGnet.prep_inputs.ICA import main as run_ica_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_ID_TO_NAME = {
    0: "Neural/other",
    1: "Eye blink (VEOG)",
    2: "Cardiac (ECG/EKG)",
    3: "Horizontal eye movement (saccade/HEOG)",
}


def _file_base(filename: str, outbasename: str | None) -> str:
    if outbasename:
        return outbasename
    return Path(filename).stem


def _to_int_list(values) -> list[int]:
    arr = np.asarray(values).reshape(-1)
    return [int(x) for x in arr]


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr))
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _pick_reference_channels(raw: mne.io.BaseRaw, kind: str) -> np.ndarray:
    if kind == "ecg":
        picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True, eog=False, ref_meg=False)
        if len(picks) > 0:
            return picks
        keys = ("ECG", "EKG")
    elif kind == "eog":
        picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False, eog=True, ref_meg=False)
        if len(picks) > 0:
            return picks
        keys = ("EOG", "HEOG", "VEOG")
    else:
        return np.array([], dtype=int)

    fallback = [idx for idx, name in enumerate(raw.ch_names) if any(key in name.upper() for key in keys)]
    return np.asarray(fallback, dtype=int)


def _best_reference_match(component_signal: np.ndarray, reference_signals: np.ndarray) -> tuple[int, float] | None:
    if reference_signals.size == 0:
        return None

    comp_z = _zscore(component_signal)
    best_idx = -1
    best_corr = 0.0
    for ref_idx in range(reference_signals.shape[0]):
        ref_z = _zscore(reference_signals[ref_idx])
        if np.all(ref_z == 0) or np.all(comp_z == 0):
            corr = 0.0
        else:
            corr = float(np.corrcoef(comp_z, ref_z)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        if best_idx < 0 or abs(corr) > abs(best_corr):
            best_idx = ref_idx
            best_corr = corr

    if best_idx < 0:
        return None
    return best_idx, best_corr


def _save_trace_panel_plot(
    t: np.ndarray,
    traces: list[np.ndarray],
    labels: list[str],
    out_path: str,
    title: str,
    colors: list[str] | None = None,
) -> None:
    if len(traces) == 0:
        return
    fig, axes = plt.subplots(len(traces), 1, figsize=(13, max(3.0, 2.2 * len(traces))), sharex=True)
    if len(traces) == 1:
        axes = [axes]

    if colors is None:
        colors = ["tab:blue"] * len(traces)
    if len(colors) < len(traces):
        colors = colors + [colors[-1]] * (len(traces) - len(colors))

    for idx, (ax, trace, label) in enumerate(zip(axes, traces, labels)):
        ax.plot(t, trace, lw=1.1, color=colors[idx])
        ax.set_ylabel("z-score")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_reference_comparisons(
    raw_fif_path: str,
    ica_path: str,
    classes: list[int],
    output_dir: str,
    max_seconds: float = 60.0,
) -> dict:
    if not op.exists(raw_fif_path):
        return {"status": "skipped", "reason": f"missing_data_file:{raw_fif_path}"}
    if not op.exists(ica_path):
        return {"status": "skipped", "reason": f"missing_ica_file:{ica_path}"}

    os.makedirs(output_dir, exist_ok=True)

    if raw_fif_path.endswith(".fif"):
        raw = mne.io.read_raw_fif(raw_fif_path, preload=True, allow_maxshield=True)
    elif raw_fif_path.endswith(".ds"):
        raw = mne.io.read_raw_ctf(raw_fif_path, preload=True)
    else:
        raw = mne.io.read_raw(raw_fif_path, preload=True)

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, ref_meg=False)
    ica = mne.preprocessing.read_ica(ica_path)
    sources = ica.get_sources(raw_meg).get_data()
    sfreq = float(raw.info["sfreq"])
    n_show = min(sources.shape[1], max(1, int(max_seconds * sfreq)))
    t = np.arange(n_show) / sfreq

    rows = []
    plot_paths = []

    # Cardiac handling
    ecg_picks = _pick_reference_channels(raw, "ecg")
    ecg_comp_indices = [idx for idx, class_id in enumerate(classes) if class_id == 2]
    if len(ecg_comp_indices) > 0 and len(ecg_picks) > 0:
        cardiac_rows_before = len(rows)
        ecg_data = raw.get_data(picks=ecg_picks)
        ecg_names = [raw.ch_names[p] for p in ecg_picks]
        all_ecg_names = ";".join(ecg_names)
        for comp_idx in ecg_comp_indices:
            match = _best_reference_match(sources[comp_idx], ecg_data)
            if match is None:
                continue
            ref_idx, corr = match
            ref_name = ecg_names[ref_idx]

            ic_trace = _zscore(sources[comp_idx][:n_show])
            ref_trace = _zscore(ecg_data[ref_idx][:n_show])

            out_name = f"IC{comp_idx + 1:02d}_cardiac_{_sanitize_name(ref_name)}.png"
            out_path = op.join(output_dir, out_name)
            _save_trace_panel_plot(
                t=t,
                traces=[ic_trace, ref_trace],
                labels=[
                    f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})",
                    f"ECG channel {ref_name} (corr={corr:.3f})",
                ],
                out_path=out_path,
                title=f"Cardiac comparison: IC{comp_idx + 1} vs {ref_name}",
                colors=["tab:blue", "tab:red"],
            )

            plot_paths.append(out_path)
            rows.append(
                {
                    "artifact_type": "cardiac",
                    "component_index_0based": comp_idx,
                    "component_number_1based": comp_idx + 1,
                    "predicted_class_id": classes[comp_idx],
                    "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                    "best_reference_channel": ref_name,
                    "all_reference_channels": all_ecg_names,
                    "correlation": f"{corr:.6f}",
                    "plot_file": out_path,
                }
            )
        # If matching fails for all components, still output IC-only cardiac plots.
        if len(rows) == cardiac_rows_before:
            traces = []
            labels = []
            for comp_idx in ecg_comp_indices:
                traces.append(_zscore(sources[comp_idx][:n_show]))
                labels.append(f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})")
            out_path = op.join(output_dir, "IC_cardiac_only.png")
            _save_trace_panel_plot(
                t=t,
                traces=traces,
                labels=labels,
                out_path=out_path,
                title="Cardiac IC-only plot (ECG match unavailable)",
                colors=["tab:blue"] * len(traces),
            )
            plot_paths.append(out_path)
            for comp_idx in ecg_comp_indices:
                rows.append(
                    {
                        "artifact_type": "cardiac_ic_only",
                        "component_index_0based": comp_idx,
                        "component_number_1based": comp_idx + 1,
                        "predicted_class_id": classes[comp_idx],
                        "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                        "best_reference_channel": "",
                        "all_reference_channels": all_ecg_names,
                        "correlation": "",
                        "plot_file": out_path,
                    }
                )
    elif len(ecg_comp_indices) > 0:
        # If ECG is unavailable, still provide IC-only cardiac plots.
        traces = []
        labels = []
        for comp_idx in ecg_comp_indices:
            traces.append(_zscore(sources[comp_idx][:n_show]))
            labels.append(f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})")

        out_path = op.join(output_dir, "IC_cardiac_only.png")
        _save_trace_panel_plot(
            t=t,
            traces=traces,
            labels=labels,
            out_path=out_path,
            title="Cardiac IC-only plot (no ECG channel available)",
            colors=["tab:blue"] * len(traces),
        )
        plot_paths.append(out_path)
        for comp_idx in ecg_comp_indices:
            rows.append(
                {
                    "artifact_type": "cardiac_ic_only",
                    "component_index_0based": comp_idx,
                    "component_number_1based": comp_idx + 1,
                    "predicted_class_id": classes[comp_idx],
                    "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                    "best_reference_channel": "",
                    "all_reference_channels": "",
                    "correlation": "",
                    "plot_file": out_path,
                }
            )

    # Ocular handling
    eog_picks = _pick_reference_channels(raw, "eog")
    ocular_candidates = {1: [idx for idx, cls in enumerate(classes) if cls == 1], 3: [idx for idx, cls in enumerate(classes) if cls == 3]}
    if len(ocular_candidates[1]) > 0 or len(ocular_candidates[3]) > 0:
        selected = {}
        class_meta = {
            1: {"label": "vEOG", "color": "tab:green"},
            3: {"label": "hEOG", "color": "tab:orange"},
        }

        eog_data = np.array([])
        eog_names: list[str] = []
        all_eog_names = ""
        if len(eog_picks) > 0:
            eog_data = raw.get_data(picks=eog_picks)
            eog_names = [raw.ch_names[p] for p in eog_picks]
            all_eog_names = ";".join(eog_names)

        for class_id in (1, 3):
            if len(ocular_candidates[class_id]) == 0:
                continue
            if len(eog_picks) > 0:
                best = None
                for comp_idx in ocular_candidates[class_id]:
                    match = _best_reference_match(sources[comp_idx], eog_data)
                    if match is None:
                        continue
                    ref_idx, corr = match
                    entry = {
                        "comp_idx": comp_idx,
                        "corr": corr,
                        "ref_name": eog_names[ref_idx],
                    }
                    if best is None or abs(corr) > abs(best["corr"]):
                        best = entry
                if best is None:
                    comp_idx = ocular_candidates[class_id][0]
                    best = {"comp_idx": comp_idx, "corr": None, "ref_name": ""}
                selected[class_id] = best
            else:
                comp_idx = ocular_candidates[class_id][0]
                selected[class_id] = {"comp_idx": comp_idx, "corr": None, "ref_name": ""}

        if selected:
            traces = []
            labels = []
            colors = []
            for class_id in (1, 3):
                if class_id not in selected:
                    continue
                entry = selected[class_id]
                comp_idx = entry["comp_idx"]
                meta = class_meta[class_id]
                traces.append(_zscore(sources[comp_idx][:n_show]))
                if entry["corr"] is None:
                    labels.append(f"{meta['label']} IC{comp_idx + 1} (no EOG channel)")
                else:
                    labels.append(
                        f"{meta['label']} IC{comp_idx + 1} (best {entry['ref_name']}, corr={entry['corr']:.3f})"
                    )
                colors.append(meta["color"])

            # Add each available EOG channel in its own panel.
            for idx, eog_name in enumerate(eog_names):
                traces.append(_zscore(eog_data[idx][:n_show]))
                labels.append(f"EOG channel {eog_name}")
                colors.append("tab:red")

            if len(eog_names) > 0:
                ocular_out_path = op.join(output_dir, "IC_ocular_combined_all_EOG.png")
                ocular_title = "Ocular comparison: vEOG/hEOG ICs + all EOG channels"
            else:
                ocular_out_path = op.join(output_dir, "IC_ocular_only.png")
                ocular_title = "Ocular IC-only plot (no EOG channel available)"

            _save_trace_panel_plot(
                t=t,
                traces=traces,
                labels=labels,
                out_path=ocular_out_path,
                title=ocular_title,
                colors=colors,
            )
            plot_paths.append(ocular_out_path)

            for class_id in (1, 3):
                if class_id not in selected:
                    continue
                entry = selected[class_id]
                comp_idx = entry["comp_idx"]
                rows.append(
                    {
                        "artifact_type": "ocular_combined" if len(eog_names) > 0 else "ocular_ic_only",
                        "component_index_0based": comp_idx,
                        "component_number_1based": comp_idx + 1,
                        "predicted_class_id": classes[comp_idx],
                        "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                        "best_reference_channel": entry["ref_name"],
                        "all_reference_channels": all_eog_names,
                        "correlation": "" if entry["corr"] is None else f"{entry['corr']:.6f}",
                        "plot_file": ocular_out_path,
                    }
                )

    if not rows:
        return {"status": "skipped", "reason": "no_matching_components_or_reference_channels", "plots": []}

    csv_path = op.join(output_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact_type",
                "component_index_0based",
                "component_number_1based",
                "predicted_class_id",
                "predicted_class_name",
                "best_reference_channel",
                "all_reference_channels",
                "correlation",
                "plot_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"status": "ok", "plots": plot_paths, "summary_csv": csv_path}


def _build_report(file_base: str, classes: list[int], bads_idx: list[int]) -> dict:
    bads = set(bads_idx)
    records = []
    for idx, class_id in enumerate(classes):
        records.append(
            {
                "component_index_0based": idx,
                "component_number_1based": idx + 1,
                "class_id": class_id,
                "class_name": CLASS_ID_TO_NAME.get(class_id, "Unknown"),
                "remove_component": idx in bads,
            }
        )

    unique, counts = np.unique(np.asarray(classes), return_counts=True)
    class_counts = {
        CLASS_ID_TO_NAME.get(int(class_id), str(int(class_id))): int(count)
        for class_id, count in zip(unique, counts)
    }

    return {
        "file_base": file_base,
        "class_id_map": CLASS_ID_TO_NAME,
        "class_counts": class_counts,
        "remove_components_0based": bads_idx,
        "remove_components_1based": [idx + 1 for idx in bads_idx],
        "components": records,
    }


def get_ic_probabilities(class_output: dict) -> list[dict[str, float]]:
    """Return per-IC class probabilities from ``classify_ica`` output.

    Parameters
    ----------
    class_output
        Output dictionary returned by :func:`MEGnet.prep_inputs.ICA.classify_ica`.

    Returns
    -------
    list of dict
        One entry per IC component. Each dictionary maps class labels
        (``Neural/other``, ``Eye blink (VEOG)``, ``Cardiac (ECG/EKG)``,
        ``Horizontal eye movement (saccade/HEOG)``) to probability values.
    """
    probs = class_output.get("class_probabilities")
    if probs is None:
        classes = _to_int_list(class_output.get("classes", []))
        if len(classes) == 0:
            return []
        probs = np.eye(len(CLASS_ID_TO_NAME), dtype=float)[classes]

    prob_arr = np.asarray(probs, dtype=float)
    if prob_arr.ndim != 2:
        raise ValueError(f"Expected a 2D probability array, got shape {prob_arr.shape}")

    ordered_class_ids = sorted(CLASS_ID_TO_NAME)
    if prob_arr.shape[1] != len(ordered_class_ids):
        raise ValueError(
            "Probability output shape does not match expected class count: "
            f"{prob_arr.shape[1]} vs {len(ordered_class_ids)}"
        )

    return [
        {
            CLASS_ID_TO_NAME[class_id]: float(prob_arr[comp_idx, class_id])
            for class_id in ordered_class_ids
        }
        for comp_idx in range(prob_arr.shape[0])
    ]


def get_ic_class_probabilities(results_dir: str, filename: str, outbasename: str | None = None) -> list[dict[str, float]]:
    """Classify ICA outputs and return per-IC class probabilities.

    This is a convenience API for programmatic use of the wrapper when only
    component probabilities are needed.
    """
    class_output = classify_ica(
        results_dir=results_dir,
        outbasename=outbasename,
        filename=filename,
    )
    return get_ic_probabilities(class_output)



def get_artifact_probabilities(ic_probabilities: list[dict[str, float]]) -> list[dict[str, float]]:
    """Return per-IC ocular (EOG) and cardiac (ECG) probabilities.

    EOG is the sum of vertical and horizontal ocular classes.
    """
    out: list[dict[str, float]] = []
    for comp_idx, prob_map in enumerate(ic_probabilities):
        eog_prob = float(prob_map.get("Eye blink (VEOG)", 0.0) + prob_map.get("Horizontal eye movement (saccade/HEOG)", 0.0))
        ecg_prob = float(prob_map.get("Cardiac (ECG/EKG)", 0.0))
        out.append(
            {
                "component_index_0based": comp_idx,
                "component_number_1based": comp_idx + 1,
                "eog_probability": eog_prob,
                "ecg_probability": ecg_prob,
            }
        )
    return out


def rank_ic_candidates(
    artifact_probs: list[dict[str, float]], artifact_key: str, top_k: int = 2, min_probability: float | None = None
) -> list[dict[str, float]]:
    """Rank ICs by a chosen artifact probability key and return top candidates."""
    ranked = sorted(artifact_probs, key=lambda row: row[artifact_key], reverse=True)
    if min_probability is not None:
        ranked = [row for row in ranked if row[artifact_key] >= float(min_probability)]
    return ranked[: max(0, int(top_k))]



def summarize_probability_quality(ic_probabilities: list[dict[str, float]], atol: float = 1e-6) -> dict[str, float | int | bool]:
    """Summarize whether per-IC probabilities look like soft distributions."""
    if len(ic_probabilities) == 0:
        return {
            "n_components": 0,
            "row_sum_min": float("nan"),
            "row_sum_max": float("nan"),
            "one_hot_like_count": 0,
            "all_one_hot_like": False,
        }

    arr = np.array([list(row.values()) for row in ic_probabilities], dtype=float)
    row_sums = arr.sum(axis=1)

    one_hot_like = 0
    for row in arr:
        close_to_one = np.isclose(row, 1.0, atol=atol)
        close_to_zero = np.isclose(row, 0.0, atol=atol)
        if np.sum(close_to_one) == 1 and np.all(np.logical_or(close_to_one, close_to_zero)):
            one_hot_like += 1

    return {
        "n_components": int(arr.shape[0]),
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
        "one_hot_like_count": int(one_hot_like),
        "all_one_hot_like": bool(one_hot_like == arr.shape[0]),
    }


def _apply_ica_cleanup(
    raw_dataset: str, ica_path: str, bads_idx: list[int], out_clean_path: str, out_ica_applied_path: str
) -> None:
    raw = load_data(raw_dataset)
    ica = mne.preprocessing.read_ica(ica_path)
    ica.exclude = bads_idx
    ica.apply(raw)
    raw.save(out_clean_path, overwrite=True)
    ica.save(out_ica_applied_path, overwrite=True)


def _safe_fig_save(fig, out_path: str) -> None:
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_qc_out_dir(ica_file: str, data_file: str) -> str:
    data_stem = Path(data_file).stem
    return op.join(op.dirname(ica_file), "MEGnetExtPlots", data_stem)


def _ensure_megnet_qc_compat() -> None:
    # MEGnet monkey patches MNE topomap internals and misses this symbol in some envs.
    import MEGnet.prep_inputs.ICA as megnet_ica
    from mne.viz.topomap import _topomap_plot_sensors

    if not hasattr(megnet_ica, "_topomap_plot_sensors"):
        megnet_ica._topomap_plot_sensors = _topomap_plot_sensors


def _run_qc_plotting_fallback(ica_file: str, data_file: str, apply_filter: bool) -> None:
    out_dir = _resolve_qc_out_dir(ica_file=ica_file, data_file=data_file)
    os.makedirs(out_dir, exist_ok=True)

    ica = mne.preprocessing.read_ica(ica_file)
    if data_file.endswith(".fif"):
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    elif data_file.endswith(".ds"):
        raw = mne.io.read_raw_ctf(data_file, preload=True)
    else:
        raw = mne.io.read_raw(data_file, preload=True)

    if apply_filter:
        lfreq, hfreq = 1.0, 98.0
        line_freq = raw.info.get("line_freq", None)
        if line_freq:
            notch_freqs = np.arange(line_freq, hfreq, line_freq)
            if len(notch_freqs):
                raw.notch_filter(
                    notch_freqs,
                    picks=["meg", "eeg", "eog", "ecg"],
                    filter_length="auto",
                    notch_widths=2,
                    trans_bandwidth=1.0,
                    verbose=False,
                )
        raw.filter(
            lfreq,
            hfreq,
            picks=["meg", "eeg", "eog", "ecg"],
            filter_length="auto",
            method="fir",
            phase="zero",
            verbose=False,
        )

    exp_var = ica.get_explained_variance_ratio(raw)
    sen_types = set(raw.get_channel_types())
    with open(op.join(out_dir, "Explained_variance_ratio.csv"), "w", encoding="utf-8") as f:
        if "grad" not in sen_types and "mag" in exp_var:
            f.write("data_file,\tmag\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('mag', np.nan)}\n")
        elif "mag" not in sen_types and "grad" in exp_var:
            f.write("data_file,\tgrad\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('grad', np.nan)}\n")
        else:
            f.write("data_file,\tgrad,\tmag\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('grad', np.nan)},\t{exp_var.get('mag', np.nan)}\n")

    src_fig = ica.plot_sources(raw, picks=range(ica.n_components_), show=False, block=False)
    _safe_fig_save(src_fig, op.join(out_dir, "all_comp_time_series_plot.png"))

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, ref_meg=True)
    sources = ica.get_sources(raw_meg).get_data()
    sfreq = float(raw_meg.info["sfreq"])
    n_show = min(sources.shape[1], max(1, int(sfreq * 60)))
    t = np.arange(n_show) / sfreq
    for comp_idx in range(sources.shape[0]):
        sig = sources[comp_idx, :n_show]
        freqs, psd = signal.welch(sig, fs=sfreq, nperseg=min(2048, len(sig)))

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
        axes[0].plot(t, sig, lw=0.8, color="tab:blue")
        axes[0].set_title(f"ICA component {comp_idx} time series (first {n_show / sfreq:.1f}s)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("a.u.")
        axes[0].grid(alpha=0.2)

        axes[1].plot(freqs, psd, lw=0.8, color="tab:purple")
        axes[1].set_xlim(0, min(120, freqs.max()))
        axes[1].set_title("Welch PSD")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Power")
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        _safe_fig_save(fig, op.join(out_dir, f"properties_plot_IC{comp_idx:03d}.png"))

    for check in ("eog", "ecg"):
        try:
            _, scores = getattr(ica, f"find_bads_{check}")(raw)
        except Exception:
            continue
        try:
            score_arr = np.asarray(scores, dtype=float)
            if score_arr.size == 0:
                continue
            score_arr = np.squeeze(score_arr)
            n_comp = int(ica.n_components_)

            # Handle score arrays from single or multiple reference channels.
            # For multi-channel outputs (e.g., shape (2, n_components)),
            # aggregate to one value per component using max absolute score.
            if score_arr.ndim == 1:
                plot_scores = score_arr
            elif score_arr.ndim == 2:
                if score_arr.shape[1] == n_comp:
                    plot_scores = np.max(np.abs(score_arr), axis=0)
                elif score_arr.shape[0] == n_comp:
                    plot_scores = np.max(np.abs(score_arr), axis=1)
                else:
                    plot_scores = np.max(np.abs(score_arr), axis=0).reshape(-1)
            else:
                reduce_axes = tuple(range(score_arr.ndim - 1))
                plot_scores = np.max(np.abs(score_arr), axis=reduce_axes).reshape(-1)

            if plot_scores.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(np.arange(plot_scores.size), plot_scores, color="tab:gray")
            ax.set_title(f"ICA component scores ({check.upper()})")
            ax.set_xlabel("ICA component index")
            ax.set_ylabel("Score")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            _safe_fig_save(fig, op.join(out_dir, f"score_plot_{check.upper()}.png"))
        except Exception as score_exc:
            print(f"Skipping {check.upper()} score plot in QC fallback: {score_exc}")

    print(f"QC fallback plots written to: {out_dir}")


def _run_qc_plotting(ica_file: str, data_file: str, apply_filter: bool, block: bool) -> None:
    from MEGnet.megnet_qc_plots import plot_all

    try:
        _ensure_megnet_qc_compat()
        plot_all(
            results_dir=None,
            ica_file=ica_file,
            data_file=data_file,
            apply_filter=apply_filter,
            block=block,
            apply_ica=False,
        )
    except Exception as exc:
        print(f"MEGnet QC plotting failed ({exc}); running wrapper fallback QC plotting.")
        _run_qc_plotting_fallback(ica_file=ica_file, data_file=data_file, apply_filter=apply_filter)
    finally:
        plt.close("all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MEGnet preprocessing + ICA + classification + optional ICA application "
            "in one command."
        )
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="Path to MEG dataset (.fif, .ds, .sqd, BTI path) used by MEGnet.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory where MEGnet output folder will be created.",
    )
    parser.add_argument(
        "--line-freq",
        type=float,
        choices=[50.0, 60.0],
        required=True,
        help="Mains frequency (50 or 60 Hz).",
    )
    parser.add_argument(
        "--filename-raw",
        default=None,
        help="Optional non-SSS raw FIF file for MEGIN bad-channel detection.",
    )
    parser.add_argument(
        "--outbasename",
        default=None,
        help="Optional output folder name inside --results-dir.",
    )
    parser.add_argument(
        "--bad-channels",
        default="",
        help="Comma-separated channels to drop before ICA (example: MEG0113,MEG2443).",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Skip preprocessing/ICA generation and classify existing MEGnet outputs only.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip megnet_init weight check/download.",
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Do not apply/remove predicted bad components from the raw data.",
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        help="Run MEGnet QC plotting after classification.",
    )
    parser.add_argument(
        "--qc-apply-filter",
        action="store_true",
        help="Pass --apply_filter behavior to QC plotting.",
    )
    parser.add_argument(
        "--qc-block",
        action="store_true",
        help="Block QC plotting windows (interactive mode).",
    )
    parser.add_argument(
        "--run-ref-compare",
        action="store_true",
        help="Plot predicted cardiac/ocular ICs against ECG/EOG channels when available.",
    )
    parser.add_argument(
        "--compare-max-seconds",
        type=float,
        default=60.0,
        help="Duration in seconds to show per comparison plot.",
    )
    parser.add_argument(
        "--compare-out-dir",
        default=None,
        help="Optional output directory for IC-vs-reference comparison plots.",
    )
    parser.add_argument(
        "--require-soft-probabilities",
        action="store_true",
        help="Fail if probabilities are unavailable or appear one-hot for all ICs.",
    )
    parser.add_argument(
        "--require-soft-probabilities",
        action="store_true",
        help="Fail if probabilities are unavailable or appear one-hot for all ICs.",
    )
    return parser


def _exc_info(exc: Exception) -> dict:
    return {"type": exc.__class__.__name__, "message": str(exc)}


def _write_report(report_path: str, report: dict) -> None:
    parent = op.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    file_base = _file_base(args.filename, args.outbasename)
    results_subdir = op.join(args.results_dir, file_base)
    os.makedirs(results_subdir, exist_ok=True)
    report_path = args.report_file or op.join(results_subdir, "megnet_summary.json")

    report: dict = {
        "file_base": file_base,
        "input_filename": args.filename,
        "status": "running",
        "stages": {},
    }
    _write_report(report_path, report)

    if not op.exists(args.filename):
        report["status"] = "failed"
        report["stages"]["input"] = {
            "status": "failed",
            "error": {"type": "FileNotFoundError", "message": f"Input dataset not found: {args.filename}"},
        }
        _write_report(report_path, report)
        print(f"Input dataset not found: {args.filename}")
        print(f"Summary report: {report_path}")
        return 1

    if not args.skip_init:
        try:
            megnet_init()
            report["stages"]["init"] = {"status": "ok"}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["init"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"Initialization failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["init"] = {"status": "skipped"}
    _write_report(report_path, report)

    bad_channels = [x.strip() for x in args.bad_channels.split(",") if x.strip()]

    if not args.classify_only:
        try:
            run_ica_pipeline(
                args.filename,
                outbasename=args.outbasename,
                mains_freq=args.line_freq,
                save_preproc=True,
                save_ica=True,
                seedval=0,
                results_dir=args.results_dir,
                filename_raw=args.filename_raw,
                do_assess_bads=bool(args.filename_raw),
                bad_channels=bad_channels,
            )
            report["stages"]["pipeline"] = {"status": "ok"}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["pipeline"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"Pipeline failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["pipeline"] = {"status": "skipped"}
    _write_report(report_path, report)

    try:
        class_output = classify_ica(
            results_dir=args.results_dir,
            outbasename=args.outbasename,
            filename=args.filename,
        )
        report["stages"]["classify"] = {"status": "ok"}
    except Exception as exc:
        hint = None
        if "list index out of range" in str(exc):
            hint = (
                "Classification may fail when usable ICA time-series chunks cannot be generated "
                "(e.g., very short or malformed inputs)."
            )
        report["status"] = "failed"
        out = {"status": "failed", "error": _exc_info(exc)}
        if hint:
            out["hint"] = hint
        report["stages"]["classify"] = out
        _write_report(report_path, report)
        print(f"Classification failed: {exc}")
        if hint:
            print(hint)
        print(f"Summary report: {report_path}")
        return 1
    classes = _to_int_list(class_output["classes"])
    bads_idx = sorted(set(_to_int_list(class_output["bads_idx"])))
    used_probability_fallback = class_output.get("class_probabilities") is None
    class_probabilities = get_ic_probabilities(class_output)
    probability_quality = summarize_probability_quality(class_probabilities)
    artifact_probabilities = get_artifact_probabilities(class_probabilities)
    top_eog = rank_ic_candidates(artifact_probabilities, artifact_key="eog_probability", top_k=2)
    top_ecg = rank_ic_candidates(artifact_probabilities, artifact_key="ecg_probability", top_k=2)

    if used_probability_fallback:
        report.setdefault("warnings", []).append(
            "class_probabilities missing from classifier output; falling back to one-hot probabilities from class labels."
        )
    if probability_quality.get("all_one_hot_like"):
        report.setdefault("warnings", []).append(
            "All IC probabilities are one-hot-like (0/1). Output may be hard labels rather than soft confidences."
        )
    if args.require_soft_probabilities and (used_probability_fallback or probability_quality.get("all_one_hot_like")):
        report["status"] = "failed"
        report["stages"]["classify"] = {
            "status": "failed",
            "error": {
                "type": "ProbabilityQualityError",
                "message": "Soft class probabilities were required, but classifier output was one-hot/fallback.",
            },
        }
        report["class_probability_diagnostics"] = probability_quality
        _write_report(report_path, report)
        print("Soft probabilities were requested but unavailable/one-hot-like.")
        print(f"Summary report: {report_path}")
        return 1

    report.update(_build_report(file_base=file_base, classes=classes, bads_idx=bads_idx))
    report["class_probabilities"] = class_probabilities
    report["class_probability_diagnostics"] = probability_quality
    report["artifact_probabilities"] = artifact_probabilities
    report["top_artifact_candidates"] = {"eog": top_eog, "ecg": top_ecg}
    _write_report(report_path, report)

    preproc_fif = op.join(results_subdir, f"{file_base}_250srate_meg.fif")
    ica_file = op.join(results_subdir, f"{file_base}_0-ica.fif")
    ica_applied_file = ica_file.replace("_0-ica.fif", "_0-ica_applied.fif")

    out_clean = None
    out_ica_applied = None
    if not args.skip_apply:
        try:
            if not op.exists(ica_file):
                raise FileNotFoundError(
                    f"Cannot apply ICA because this file is missing: {ica_file}. "
                    "Run without --classify-only, or generate ICA first."
                )
            out_clean = op.join(results_subdir, "ica_clean.fif")
            out_ica_applied = ica_applied_file
            _apply_ica_cleanup(
                raw_dataset=args.filename,
                ica_path=ica_file,
                bads_idx=bads_idx,
                out_clean_path=out_clean,
                out_ica_applied_path=out_ica_applied,
            )
            report["stages"]["apply"] = {"status": "ok", "ica_clean_file": out_clean, "ica_applied_file": out_ica_applied}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["apply"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"ICA apply failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["apply"] = {"status": "skipped"}
    _write_report(report_path, report)

    if args.run_qc:
        try:
            if not op.exists(preproc_fif):
                raise FileNotFoundError(f"QC plotting requires this file: {preproc_fif}")
            qc_ica_file = out_ica_applied or (ica_applied_file if op.exists(ica_applied_file) else ica_file)
            if not op.exists(qc_ica_file):
                raise FileNotFoundError(f"QC plotting requires an ICA file: {qc_ica_file}")
            _run_qc_plotting(
                ica_file=qc_ica_file,
                data_file=preproc_fif,
                apply_filter=args.qc_apply_filter,
                block=args.qc_block,
            )
            report["qc"] = {"status": "ok", "ica_file": qc_ica_file, "data_file": preproc_fif}
            report["stages"]["qc"] = {"status": "ok"}
        except Exception as exc:
            report.setdefault("warnings", []).append(f"QC failed: {exc}")
            report["stages"]["qc"] = {"status": "failed", "error": _exc_info(exc)}
            print(f"QC failed: {exc}")
    else:
        report["stages"]["qc"] = {"status": "skipped"}
    _write_report(report_path, report)

    if args.run_ref_compare:
        try:
            compare_out_dir = args.compare_out_dir or op.join(results_subdir, "IC_ref_comparisons")
            compare_info = _run_reference_comparisons(
                raw_fif_path=preproc_fif,
                ica_path=ica_file if op.exists(ica_file) else ica_applied_file,
                classes=classes,
                output_dir=compare_out_dir,
                max_seconds=float(args.compare_max_seconds),
            )
            report["reference_comparison"] = compare_info
            report["stages"]["reference_comparison"] = {"status": "ok"}
        except Exception as exc:
            report.setdefault("warnings", []).append(f"Reference comparison failed: {exc}")
            report["stages"]["reference_comparison"] = {"status": "failed", "error": _exc_info(exc)}
            print(f"Reference comparison failed: {exc}")
    else:
        report["stages"]["reference_comparison"] = {"status": "skipped"}

    if report.get("status") == "running":
        report["status"] = "ok_with_warnings" if report.get("warnings") else "ok"

    _write_report(report_path, report)

    print(f"Results directory: {results_subdir}")
    print(f"Predicted class IDs for components 1..20: {classes}")
    print(f"Predicted removable components (0-based): {bads_idx}")
    print(f"Predicted removable components (1-based): {[idx + 1 for idx in bads_idx]}")
    print("Per-IC class probabilities:")
    for ic_idx, prob_map in enumerate(class_probabilities, start=1):
        formatted = ", ".join(f"{label}={prob:.4f}" for label, prob in prob_map.items())
        print(f"  IC{ic_idx:02d}: {formatted}")

    print(
        "Probability diagnostics: "
        f"row_sum_min={probability_quality['row_sum_min']:.6f}, "
        f"row_sum_max={probability_quality['row_sum_max']:.6f}, "
        f"one_hot_like={probability_quality['one_hot_like_count']}/{probability_quality['n_components']}"
    )

    print("Top EOG-probability IC candidates:")
    for row in top_eog:
        print(f"  IC{int(row['component_number_1based']):02d}: p(EOG)={row['eog_probability']:.4f}")

    print("Top ECG-probability IC candidates:")
    for row in top_ecg:
        print(f"  IC{int(row['component_number_1based']):02d}: p(ECG)={row['ecg_probability']:.4f}")
    if args.run_ref_compare and "reference_comparison" in report:
        print(f"Reference comparison status: {report['reference_comparison'].get('status')}")
    print(f"Summary report: {report_path}")
    if out_clean is not None and out_ica_applied is not None:
        print(f"ICA-cleaned raw file: {out_clean}")
        print(f"ICA file with exclude list: {out_ica_applied}")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"Warning: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
