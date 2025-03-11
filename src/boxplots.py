import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch

def process_feature_value(value):
    """
    Process feature values to ensure we have scalar values:
    - For Spike Contrast (synchrony_with_trace), extract the synchrony value
    - For array-like data, use the max value
    - Otherwise, return the value as is
    """
    if isinstance(value, dict) and 'synchrony' in value:
        return value['synchrony']
    elif isinstance(value, (list, np.ndarray)) and hasattr(value, 'size') and value.size > 1:
        return np.max(value)
    else:
        return value

def plot_feature_values_over_time(all_features_data, output_dir, custom_labels=None):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_name, time_dict in all_features_data.items():
        # Skip Spike Contrast Trace features that we don't need
        if feature_name.startswith('Spike Contrast Trace'):
            continue
            
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
        if custom_labels and len(custom_labels) == len(time_indices_sorted):
            x_labels = custom_labels
        else:
            x_labels = time_indices_sorted
            print("Warning: Custom labels not found or count mismatch. Falling back to ordinal sequence.")
        group_names = set()
        for t in time_indices_sorted:
            group_names.update(time_dict[t].keys())
        group_names = sorted(list(group_names))

        # Collect data for boxplots
        data_for_boxplot = []
        for t in time_indices_sorted:
            group_list_for_this_time = []
            for g in group_names:
                group_values = time_dict[t].get(g, [])
                
                # Process values to ensure scalar values
                processed_values = [process_feature_value(val) for val in group_values]
                
                group_list_for_this_time.append(processed_values)
            data_for_boxplot.append(group_list_for_this_time)

        # Plot 1: Standard
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        n_times = len(time_indices_sorted)
        n_groups = len(group_names)
        width = 0.6 / n_groups
        colors = plt.cm.Set1.colors

        for time_idx, group_values_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                                positions=[position],
                                widths=width,
                                patch_artist=True,
                                manage_ticks=False,
                                showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])

        ax.set_title(feature_name, fontsize=14)
        ax.set_ylabel(feature_name, fontsize=12)
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Time index of file (ordinal order per ID-chip)", fontsize=12)

        # Use patches for legend like in statistic.py
        legend_patches = [
            Patch(color=colors[idx % len(colors)], label=group)
            for idx, group in enumerate(group_names)
        ]
        ax.legend(handles=legend_patches, loc='upper left', 
                  bbox_to_anchor=(1, 1), fontsize=10)

        plt.tight_layout()
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Log-Scale
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        for time_idx, group_values_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                                positions=[position],
                                widths=width,
                                patch_artist=True,
                                manage_ticks=False,
                                showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])

        ax.set_title(feature_name + " (Log-Scale)", fontsize=14)
        ax.set_ylabel(feature_name, fontsize=12)
        ax.set_yscale('log')
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Time index of file (ordinal order per ID-chip)", fontsize=12)
        
        ax.legend(handles=legend_patches, loc='upper left', 
                  bbox_to_anchor=(1, 1), fontsize=10)
                  
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot_log.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Relative change (Control = 0)
        data_for_boxplot_relative = []
        ref_means_per_group = []
        first_time_idx_data = data_for_boxplot[0]

        for group_idx, group_values in enumerate(first_time_idx_data):
            if len(group_values) > 0:
                ref_means_per_group.append(np.median(group_values))
            else:
                ref_means_per_group.append(1.0)

        for time_idx, group_values_list in enumerate(data_for_boxplot):
            group_list_for_this_time_rel = []
            for group_idx, group_values in enumerate(group_values_list):
                if ref_means_per_group[group_idx] != 0:
                    # Values are normalized and then -1 is calculated, so the control measurement becomes 0
                    norm_values = [(val / ref_means_per_group[group_idx]) - 1 for val in group_values]
                else:
                    norm_values = [0 for _ in group_values]
                group_list_for_this_time_rel.append(norm_values)
            data_for_boxplot_relative.append(group_list_for_this_time_rel)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        for time_idx, group_values_list in enumerate(data_for_boxplot_relative):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                                positions=[position],
                                widths=width,
                                patch_artist=True,
                                manage_ticks=False,
                                showfliers=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])

        ax.set_title(f"Relative Change in {feature_name} to Reference Measurement", fontsize=14)
        ax.set_ylabel(f"Relative Change (normalized to reference)", fontsize=12)
        
        # Define linthresh for symlog scale
        linthresh = 0.5
        
        # Use symlog scale like in boxplots.py
        ax.set_yscale('symlog', linthresh=linthresh)
        
        # Add reference line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add transition indicators for symlog scale
        ax.axhline(y=linthresh, color='red', linestyle=':', alpha=0.7)
        ax.axhline(y=-linthresh, color='red', linestyle=':', alpha=0.7)
        ax.text(n_times-0.5, linthresh*1.1, f"Linear → Log transition at ±{linthresh}", 
                ha='right', va='bottom', color='red', fontsize=9, alpha=0.7)
        
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Time index of file (ordinal order per ID-chip)", fontsize=12)
        
        ax.legend(handles=legend_patches, loc='upper left', 
                  bbox_to_anchor=(1, 1), fontsize=10)

        plt.tight_layout()
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_rel.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parent_dir = "data"

    label_file_path = os.path.join(parent_dir, "labels.txt")
    if os.path.isfile(label_file_path):
        with open(label_file_path, "r", encoding="utf-8") as f:
            labels_str = f.read().strip()
            custom_labels = [e.strip() for e in labels_str.split(";")] if labels_str else None
            print("Custom labels found.")
    else:
        custom_labels = None
        print("No custom labels found.")

    groups = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    all_features_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for group in groups:
        group_path = os.path.join(parent_dir, group)
        chip_dirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        for chip_dir in chip_dirs:
            chip_path = os.path.join(group_path, chip_dir)
            npz_files = sorted([f for f in os.listdir(chip_path) if f.endswith('.npz')])

            for idx, npz_file in enumerate(npz_files, start=1):
                npz_path = os.path.join(chip_path, npz_file)
                print(f"Reading features from: Group={group}, Chip={chip_dir}, File={npz_file}, Time-Index={idx}")
                with np.load(npz_path, allow_pickle=True) as data:
                    if 'features' in data:
                        features = data['features'].item()
                        for feat_key, feat_val in features.items():
                            # Skip Spike Contrast Trace features
                            if feat_key.startswith('Spike Contrast Trace'):
                                continue
                                
                            # Add feature value to our data structure
                            all_features_data[feat_key][str(idx)][group].append(feat_val)

    inter_group_output_dir = os.path.join(parent_dir, 'Inter_Group_Boxplots')
    plot_feature_values_over_time(all_features_data, inter_group_output_dir, custom_labels)

    print("Done. Boxplots have been created from the .npz files.")

if __name__ == "__main__":
    main()
