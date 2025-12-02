import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import partitura as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import miditoolkit
from src.utils.midi import midi_to_ids
from src.model.pianoformer import PianoT5GemmaConfig

def get_evaluate_list(gt_path, pred_path, not_in = []):
    out = []
    gt_path = Path(gt_path)
    pred_path = Path(pred_path)
    gt_files = sorted(list(gt_path.glob("*.mid")))
    for gt_file in gt_files:
        prefix = str(gt_file).split("-")[-2].split("/")[-1]
        number = str(gt_file).split("-")[-1].split(".")[0]
        if prefix in not_in:
            continue
        out.append({
            "gt": os.path.join(gt_path, f"{prefix}-{number}.mid"), 
            "pred": os.path.join(pred_path, f"{prefix}.mid")
        })
    return out

def plot_velocity_distribution_and_calculate_metrics_pt(evaluate_list, method):
    method_name = method.split("/")[-1]
    gt_velocities = []
    pred_velocities = []

    print(f"[{method_name}] Extracting velocity data for distribution analysis...")
    for item in tqdm(evaluate_list, desc=f"Analyzing Vel Dist for {method_name}"):
        try:
            gt_perf = pt.load_performance_midi(item["gt"])
            gt_note_array = gt_perf.note_array()
            if gt_note_array.size > 0:
                gt_velocities.extend(gt_note_array['velocity'])
            
            pred_perf = pt.load_performance_midi(item["pred"])
            pred_note_array = pred_perf.note_array()
            if pred_note_array.size > 0:
                pred_velocities.extend(pred_note_array['velocity'])
        except Exception as e:
            print(f"Warning: Could not process file pair for plot. GT: {item['gt']}. Error: {e}")
            continue

    if not gt_velocities or not pred_velocities:
        print(f"Error: No valid velocity data found for {method_name}. Skipping plot and metrics.")
        return None, None, None

    bins = np.arange(0, 129)
    
    gt_hist, _ = np.histogram(gt_velocities, bins=bins)
    pred_hist, _ = np.histogram(pred_velocities, bins=bins)

    epsilon = 1e-10
    gt_prob = (gt_hist / np.sum(gt_hist)) + epsilon
    pred_prob = (pred_hist / np.sum(pred_hist)) + epsilon
    
    gt_prob /= np.sum(gt_prob)
    pred_prob /= np.sum(pred_prob)

    js_divergence = jensenshannon(gt_prob, pred_prob, base=2)

    
    intersection = np.sum(np.minimum(gt_prob, pred_prob))

    print(f"--- Distribution Metrics for {method_name} ---")
    print(f"Jensen-Shannon Divergence: {js_divergence:.4f} (↓ lower is better)")
    print(f"Histogram Intersection: {intersection:.4f} (↑ higher is better)")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(gt_velocities, color="skyblue", label="Ground Truth (Human)", kde=False, bins=bins, alpha=0.7)
    sns.histplot(pred_velocities, color="red", label=f"Prediction ({method_name})", kde=False, bins=bins, alpha=0.5)
    
    plt.title(f"Velocity Distribution Comparison: {method_name}", fontsize=16)
    plt.xlabel("MIDI Velocity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.xlim(0, 128)
    
    metrics_text = (f"JS Divergence: {js_divergence:.4f}\n"
                    f"Intersection: {intersection:.4f}")
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_dir = Path("results/imgs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"vel_dist_{method_name}.png")
    plt.close()
    
    return js_divergence, intersection

def plot_duration_distribution_and_metrics_pt(evaluate_list, method, duration_range=(0, 500), num_bins=250):
    method_name = method.split("/")[-1]
    gt_durations = []
    pred_durations = []

    print(f"[{method_name}] Extracting duration data for distribution analysis...")
    for item in tqdm(evaluate_list, desc=f"Analyzing Dur Dist for {method_name}"):
        try:
            gt_perf = pt.load_performance_midi(item["gt"])
            gt_note_array = gt_perf.note_array()
            if gt_note_array.size > 0:
                gt_durs_tick = gt_note_array['duration_tick']
                gt_durations.extend(gt_durs_tick)

            pred_perf = pt.load_performance_midi(item["pred"])
            pred_note_array = pred_perf.note_array()
            if pred_note_array.size > 0:
                pred_durs_tick = pred_note_array['duration_tick']
                pred_durations.extend(pred_durs_tick)
        except Exception as e:
            print(f"Warning: Could not process file pair. GT: {item['gt']}. Error: {e}")
            continue
    
    gt_durations = np.array(gt_durations)
    pred_durations = np.array(pred_durations)
    
    gt_durations = gt_durations[(gt_durations >= duration_range[0]) & (gt_durations < duration_range[1])]
    pred_durations = pred_durations[(pred_durations >= duration_range[0]) & (pred_durations < duration_range[1])]

    if gt_durations.size == 0 or pred_durations.size == 0:
        print(f"Error: No valid duration data found in range for {method_name}. Skipping.")
        return None, None, None

    bins = np.linspace(duration_range[0], duration_range[1], num_bins + 1)
    gt_hist, _ = np.histogram(gt_durations, bins=bins)
    pred_hist, _ = np.histogram(pred_durations, bins=bins)

    epsilon = 1e-10
    gt_prob = (gt_hist / np.sum(gt_hist)) + epsilon
    pred_prob = (pred_hist / np.sum(pred_hist)) + epsilon
    gt_prob /= np.sum(gt_prob)
    pred_prob /= np.sum(pred_prob)

    js_divergence = jensenshannon(gt_prob, pred_prob, base=2)
    intersection = np.sum(np.minimum(gt_prob, pred_prob))

    print(f"--- Duration Distribution Metrics for {method_name} (Range: {duration_range}) ---")
    print(f"Jensen-Shannon Divergence: {js_divergence:.4f} (↓ lower is better)")
    print(f"Histogram Intersection: {intersection:.4f} (↑ higher is better)")

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(gt_durations, color="skyblue", label="Ground Truth (Human)", kde=False, bins=bins, alpha=0.7)
    sns.histplot(pred_durations, color="red", label=f"Prediction ({method_name})", kde=False, bins=bins, alpha=0.5)
    plt.title(f"Note Duration Distribution Comparison: {method_name}", fontsize=16)
    plt.xlabel("Duration (ticks)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.xlim(duration_range)

    metrics_text = (f"JS Divergence: {js_divergence:.4f}\n"
                    f"Intersection: {intersection:.4f}")
    plt.text(0.65, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_dir = Path("results/imgs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"dur_dist_{method_name}.png")
    plt.close()

    return js_divergence, intersection

def plot_ioi_distribution_and_metrics_pt(evaluate_list, method, ioi_range=(0, 200), num_bins=200):
    method_name = method.split("/")[-1]
    gt_iois = []
    pred_iois = []

    print(f"[{method_name}] Extracting IOI data for distribution analysis...")
    for item in tqdm(evaluate_list, desc=f"Analyzing IOI Dist for {method_name}"):
        try:
            gt_perf = pt.load_performance_midi(item["gt"])
            gt_note_array = gt_perf.note_array()
            if gt_note_array.size > 1:
                gt_ioi_vals = np.diff(gt_note_array['onset_tick'])
                gt_iois.extend(gt_ioi_vals)

            pred_perf = pt.load_performance_midi(item["pred"])
            pred_note_array = pred_perf.note_array()
            if pred_note_array.size > 1:
                pred_ioi_vals = np.diff(pred_note_array['onset_tick'])
                pred_iois.extend(pred_ioi_vals)
        except Exception as e:
            print(f"Warning: Could not process file pair. GT: {item['gt']}. Error: {e}")
            continue

    gt_iois = np.array(gt_iois)
    pred_iois = np.array(pred_iois)
    
    gt_iois = gt_iois[(gt_iois >= ioi_range[0]) & (gt_iois < ioi_range[1])]
    pred_iois = pred_iois[(pred_iois >= ioi_range[0]) & (pred_iois < ioi_range[1])]

    if gt_iois.size == 0 or pred_iois.size == 0:
        print(f"Error: No valid IOI data found in range for {method_name}. Skipping.")
        return None, None, None

    bins = np.linspace(ioi_range[0], ioi_range[1], num_bins + 1)
    gt_hist, _ = np.histogram(gt_iois, bins=bins)
    pred_hist, _ = np.histogram(pred_iois, bins=bins)

    epsilon = 1e-10
    gt_prob = (gt_hist / np.sum(gt_hist)) + epsilon
    pred_prob = (pred_hist / np.sum(pred_hist)) + epsilon
    gt_prob /= np.sum(gt_prob)
    pred_prob /= np.sum(pred_prob)
    
    js_divergence = jensenshannon(gt_prob, pred_prob, base=2)
    intersection = np.sum(np.minimum(gt_prob, pred_prob))

    print(f"--- IOI Distribution Metrics for {method_name} (Range: {ioi_range}) ---")
    print(f"Jensen-Shannon Divergence: {js_divergence:.4f} (↓ lower is better)")
    print(f"Histogram Intersection: {intersection:.4f} (↑ higher is better)")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(gt_iois, color="skyblue", label="Ground Truth (Human)", kde=False, bins=bins, alpha=0.7)
    sns.histplot(pred_iois, color="red", label=f"Prediction ({method_name})", kde=False, bins=bins, alpha=0.5)
    plt.title(f"Inter-Onset Interval (IOI) Distribution: {method_name}", fontsize=16)
    plt.xlabel("IOI (ticks)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.xlim(ioi_range)
    
    metrics_text = (f"JS Divergence: {js_divergence:.4f}\n"
                    f"Intersection: {intersection:.4f}")
    plt.text(0.65, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_dir = Path("results/imgs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"ioi_dist_{method_name}.png")
    plt.close()

    return js_divergence, intersection

def plot_pedal_pattern_distribution(
    evaluate_list: list[dict], 
    method: str, 
    pedal_token_base: int = 5261, 
    pedal_binarize_threshold: int = 64
):
    gt_patterns = []
    pred_patterns = []
    method_name = Path(method).stem
    config = PianoT5GemmaConfig()
    print(f"[{method_name}] Analyzing pedal patterns...")

    for item in tqdm(evaluate_list, desc=f"Analyzing Pedals for {method_name}"):
        try:
            gt_midi = miditoolkit.MidiFile(item["gt"])
            gt_tokens = midi_to_ids(config, gt_midi)
            pred_midi = miditoolkit.MidiFile(item["pred"])
            pred_tokens = midi_to_ids(config, pred_midi)

            def extract_patterns(tokens):
                patterns = []
                for i in range(0, len(tokens), 8):
                    note_chunk = tokens[i : i + 8]
                    if len(note_chunk) < 8:
                        continue

                    pedal_tokens = note_chunk[4:]
                    binary_values = []
                    for token in pedal_tokens:
                        if pedal_token_base <= token < pedal_token_base + 128:
                            pedal_value = token - pedal_token_base
                            binary_value = 1 if pedal_value >= pedal_binarize_threshold else 0
                            binary_values.append(binary_value)
                    
                    if len(binary_values) == 4:
                        pattern_decimal = (
                            binary_values[0] * 8 +
                            binary_values[1] * 4 +
                            binary_values[2] * 2 +
                            binary_values[3] * 1
                        )
                        patterns.append(pattern_decimal)
                return patterns

            gt_patterns.extend(extract_patterns(gt_tokens))
            pred_patterns.extend(extract_patterns(pred_tokens))

        except Exception as e:
            print(f"Warning: Could not process file pair. GT: {item['gt']}. Error: {e}")
            continue

    if not gt_patterns or not pred_patterns:
        print(f"Error: No valid pedal data found for {method_name}. Skipping analysis.")
        return None, None, None

    bins = np.arange(17)
    gt_hist, _ = np.histogram(gt_patterns, bins=bins)
    pred_hist, _ = np.histogram(pred_patterns, bins=bins)

    epsilon = 1e-10
    gt_prob = (gt_hist / np.sum(gt_hist)) + epsilon
    pred_prob = (pred_hist / np.sum(pred_hist)) + epsilon
    
    js_divergence = jensenshannon(gt_prob, pred_prob, base=2)
    intersection = np.sum(np.minimum(gt_prob, pred_prob))

    print(f"--- Pedal Pattern Distribution Metrics for {method_name} ---")
    print(f"Jensen-Shannon Divergence: {js_divergence:.4f} (↓ lower is better)")
    print(f"Histogram Intersection: {intersection:.4f} (↑ higher is better)")

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))
    
    x = np.arange(16)
    width = 0.35

    plt.bar(x - width/2, gt_hist, width, label="Ground Truth (Human)", color="skyblue", alpha=0.8)
    plt.bar(x + width/2, pred_hist, width, label=f"Prediction ({method_name})", color="red", alpha=0.6)

    plt.title(f"Pedal Pattern Distribution Comparison: {method_name}", fontsize=16)
    plt.xlabel("Pedal Pattern ID (0-15)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(x, [f'{i}\n({i:04b})' for i in x])
    plt.legend()
    
    metrics_text = (f"JS Divergence: {js_divergence:.4f}\n"
                    f"Intersection: {intersection:.4f}")
    plt.text(0.75, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_dir = Path("results/imgs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"pedal_pattern_dist_{method_name}.png")
    plt.close()
    
    return js_divergence, intersection

if __name__ == "__main__":
    BASE_PATH = "data/midis/testset-norm" 
    MODEL_NAME = "score" 
    
    gt_path = os.path.join(BASE_PATH, "human")
    pred_path = os.path.join(BASE_PATH, MODEL_NAME)
    
    evaluate_list = get_evaluate_list(gt_path, pred_path, not_in=[str(i) for i in range(1, 23)])
    
    plot_velocity_distribution_and_calculate_metrics_pt(evaluate_list, pred_path)
    plot_duration_distribution_and_metrics_pt(evaluate_list, pred_path)
    plot_ioi_distribution_and_metrics_pt(evaluate_list, pred_path)
    plot_pedal_pattern_distribution(evaluate_list, pred_path)
