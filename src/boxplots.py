import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_feature_values_over_time(all_features_data, output_dir, custom_labels=None):
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_name, time_dict in all_features_data.items():
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
        if custom_labels and len(custom_labels) == len(time_indices_sorted):
            x_labels = custom_labels
        else:
            x_labels = time_indices_sorted
            print("Warnung: Benutzerdefinierte Labels nicht gefunden oder Anzahl stimmt nicht überein.")
        group_names = set()
        for t in time_indices_sorted:
            group_names.update(time_dict[t].keys())
        group_names = sorted(list(group_names))

        # Daten für Boxplots sammeln
        data_for_boxplot = []
        for t in time_indices_sorted:
            group_list_for_this_time = []
            for g in group_names:
                group_values = time_dict[t].get(g, [])
                group_list_for_this_time.append(group_values)
            data_for_boxplot.append(group_list_for_this_time)

        # Plot 1: Standard
        fig, ax = plt.subplots(figsize=(10, 6))
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
        ax.set_xlabel("Zeit-Index der Datei (pro ID-Chip ordinale Reihenfolge)", fontsize=12)

        legend_patches = [
            plt.Line2D([0], [0], color=colors[idx % len(colors)],
                       label=group, marker='s', linestyle='', markersize=10)
            for idx, group in enumerate(group_names)
        ]
        ax.legend(handles=legend_patches, title="Gruppen", loc='lower left',
                  bbox_to_anchor=(0.5, -0.15), ncol=len(group_names),
                  fontsize=10, title_fontsize=12)

        plt.tight_layout()
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot.png'), dpi=300)
        plt.close()

        # Plot 2: Log-Skala
        fig, ax = plt.subplots(figsize=(10, 6))
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

        ax.set_title(feature_name + "(Log-Scale)", fontsize=14)
        ax.set_ylabel(feature_name, fontsize=12)
        ax.set_yscale('log')
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Zeit-Index der Datei (pro ID-Chip ordinale Reihenfolge)", fontsize=12)
        ax.legend(handles=legend_patches, title="Gruppen", loc='lower left',
                  bbox_to_anchor=(0.5, -0.15), ncol=len(group_names),
                  fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot_log.png'), dpi=300)
        plt.close()

        # Plot 3: Relative Veränderung (Kontrolle = 0)
        data_for_boxplot_relative = []
        ref_means_per_group = []
        first_time_idx_data = data_for_boxplot[0]

        for group_idx, group_values in enumerate(first_time_idx_data):
            if len(group_values) > 0:
                ref_means_per_group.append(np.mean(group_values))
            else:
                ref_means_per_group.append(1.0)

        for time_idx, group_values_list in enumerate(data_for_boxplot):
            group_list_for_this_time_rel = []
            for group_idx, group_values in enumerate(group_values_list):
                if ref_means_per_group[group_idx] != 0:
                    # Werte werden normiert und dann -1 gerechnet, sodass die Kontrollmessung 0 wird.
                    norm_values = [(val / ref_means_per_group[group_idx]) - 1 for val in group_values]
                else:
                    norm_values = [0 for _ in group_values]
                group_list_for_this_time_rel.append(norm_values)
            data_for_boxplot_relative.append(group_list_for_this_time_rel)

        fig, ax = plt.subplots(figsize=(10, 6))
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

        ax.set_title("Relative Veränderung der " + feature_name + " zur Referenzmessung", fontsize=14)
        ax.set_ylabel("Relative Änderung", fontsize=12)
        ax.set_yscale('symlog', linthresh=0.5)
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Zeit-Index der Datei (pro ID-Chip ordinale Reihenfolge)", fontsize=12)
        ax.legend(handles=legend_patches, title="Gruppen", loc='lower left',
                  bbox_to_anchor=(0.5, -0.15), ncol=len(group_names),
                  fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot_rel.png'), dpi=300)
        plt.close()

def main():
    parent_dir = "data"

    label_file_path = os.path.join(parent_dir, "labels.txt")
    if os.path.isfile(label_file_path):
        with open(label_file_path, "r", encoding="utf-8") as f:
            labels_str = f.read().strip()
            custom_labels = [e.strip() for e in labels_str.split(";")] if labels_str else None
            print("Benutzerdefinierte Labels gefunden.")
    else:
        custom_labels = None
        print("Keine benutzerdefinierten Labels gefunden.")

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
                print(f"Lese Features aus: Gruppe={group}, Chip={chip_dir}, Datei={npz_file}, Zeit-Index={idx}")
                with np.load(npz_path, allow_pickle=True) as data:
                    if 'features' in data:
                        features = data['features'].item()
                        for feat_key, feat_val in features.items():
                            all_features_data[feat_key][str(idx)][group].append(feat_val)

    inter_group_output_dir = os.path.join(parent_dir, 'Inter_Group_Boxplots')
    plot_feature_values_over_time(all_features_data, inter_group_output_dir, custom_labels)

    print("Fertig. Boxplots wurden aus den .npz-Dateien erstellt.")

if __name__ == "__main__":
    main()
