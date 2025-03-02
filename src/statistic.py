import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

def plot_feature_values_over_time(all_features_data, output_dir, custom_labels=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_name, time_dict in all_features_data.items():
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
        if custom_labels and len(custom_labels) == len(time_indices_sorted):
            x_labels = custom_labels
        else:
            x_labels = time_indices_sorted
            print("Warnung: Benutzerdefinierte Labels nicht gefunden oder Anzahl stimmt nicht überein.")
        
        group_names = sorted(list({g for t in time_dict.values() for g in t.keys()}))
        if len(group_names) < 2:
            continue

        data_for_boxplot = []
        for t in time_indices_sorted:
            group_list = [time_dict[t].get(g, []) for g in group_names]
            data_for_boxplot.append(group_list)

        p_values = {}
        test_times = time_indices_sorted[1:]
        n_comparisons = len(test_times)
        adjusted_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

        for t in test_times:
            bdnf_data = time_dict[t].get('BDNF', [])
            sham_data = time_dict[t].get('SHAM', [])
            if bdnf_data and sham_data:
                _, p = stats.mannwhitneyu(bdnf_data, sham_data, alternative='two-sided')
                p_values[t] = p
            else:
                p_values[t] = None

        def add_significance(ax, data_for_boxplot, time_indices_sorted):
            for time_idx, t in enumerate(time_indices_sorted):
                if t in p_values and p_values[t] is not None and p_values[t] < adjusted_alpha:
                    y_max = max([val for group_data in data_for_boxplot[time_idx] for val in group_data]) if any(data_for_boxplot[time_idx]) else 0
                    ax.text(time_idx, y_max * 1.05, '*', ha='center', va='bottom', color='black', fontsize=14)

        width = 0.6 / len(group_names)
        colors = plt.cm.Set1.colors

        fig, ax = plt.subplots(figsize=(10, 6))
        for time_idx, group_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_list):
                position = time_idx + group_idx * width - (width * (len(group_names) - 1) / 2)
                bp = ax.boxplot(group_values, positions=[position], widths=width, patch_artist=True, 
                                manage_ticks=False, showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])
        add_significance(ax, data_for_boxplot, time_indices_sorted)
        ax.set_title(feature_name, fontsize=14)
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.legend(handles=[
            plt.Line2D([0], [0], color=colors[i], label=g, marker='s', linestyle='', markersize=10) 
            for i, g in enumerate(group_names)
        ], title="Gruppen", loc='lower left', bbox_to_anchor=(0.5, -0.15), ncol=len(group_names), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_name.replace(" ", "_")}_boxplot.png'), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        for time_idx, group_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_list):
                position = time_idx + group_idx * width - (width * (len(group_names) - 1) / 2)
                bp = ax.boxplot(group_values, positions=[position], widths=width, patch_artist=True, 
                                manage_ticks=False, showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])
        add_significance(ax, data_for_boxplot, time_indices_sorted)
        ax.set_yscale('log')
        ax.set_title(feature_name + " (Log-Scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_name.replace(" ", "_")}_boxplot_log.png'), dpi=300)
        plt.close()

        ref_means = [np.mean(data_for_boxplot[0][i]) if data_for_boxplot[0][i] else 1 for i in range(len(group_names))]
        data_rel = [[[(val/ref_means[i]-1) if ref_means[i] else 0 for val in group] 
                   for i, group in enumerate(time_data)] for time_data in data_for_boxplot]

        fig, ax = plt.subplots(figsize=(10, 6))
        for time_idx, group_list in enumerate(data_rel):
            for group_idx, group_values in enumerate(group_list):
                position = time_idx + group_idx * width - (width * (len(group_names) - 1) / 2)
                bp = ax.boxplot(group_values, positions=[position], widths=width, patch_artist=True, 
                                manage_ticks=False, showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])
        add_significance(ax, data_rel, time_indices_sorted)
        ax.set_ylabel("Relative Änderung")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_name.replace(" ", "_")}_boxplot_rel.png'), dpi=300)
        plt.close()

def main():
    parent_dir = "data"
    custom_labels = None
    if os.path.exists(os.path.join(parent_dir, "labels.txt")):
        with open(os.path.join(parent_dir, "labels.txt"), "r") as f:
            custom_labels = [l.strip() for l in f.read().split(";")]
    
    all_features_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for group in os.listdir(parent_dir):
        group_path = os.path.join(parent_dir, group)
        if os.path.isdir(group_path):
            for chip in os.listdir(group_path):
                chip_path = os.path.join(group_path, chip)
                if os.path.isdir(chip_path):
                    for idx, npz in enumerate(sorted([f for f in os.listdir(chip_path) if f.endswith('.npz')]), 1):
                        data = np.load(os.path.join(chip_path, npz), allow_pickle=True)
                        if 'features' in data:
                            for feat, val in data['features'].item().items():
                                all_features_data[feat][str(idx)][group].append(val)
    
    plot_feature_values_over_time(all_features_data, os.path.join(parent_dir, 'Statistik'), custom_labels)

if __name__ == "__main__":
    main()