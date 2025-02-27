#For custom labels create an labels.txt in your experiment root directory with the following content:
#Label1;Label2;Label3
#This will be used as the x-axis labels for the boxplots.
#The script will create a separate boxplot for each feature in the .npz files.
#The boxplots will be grouped by the group names and the time indices of the .npz files.
#The boxplots will be saved as .png files in a subdirectory called 'Inter_Group_Boxplots'.
#The script will also print a warning if the custom labels are not found or the number of labels does not match the number of time indices.

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

        data_for_boxplot = []
        for t in time_indices_sorted:
            group_list_for_this_time = []
            for g in group_names:
                group_values = time_dict[t].get(g, [])
                group_list_for_this_time.append(group_values)
            data_for_boxplot.append(group_list_for_this_time)

        fig, ax = plt.subplots(figsize=(10, 6))
        n_times = len(time_indices_sorted)
        n_groups = len(group_names)
        width = 0.6 / n_groups

        colors = plt.cm.Set1.colors  # Farbschema Set1

        # Boxplots pro Zeitindex und Gruppe
        # showfliers=False unterdrückt das Zeichnen der Ausreißer, wodurch der Plot weniger "gestaucht" wirkt.
        for time_idx, group_values_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                                positions=[position],
                                widths=width,
                                patch_artist=True,
                                manage_ticks=False,
                                showfliers=False  # <-- Ausreißer nicht anzeigen
                                )
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])

        ax.set_title(feature_name, fontsize=14)
        ax.set_ylabel(feature_name, fontsize=12)
        ax.set_xticks(range(len(time_indices_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
        ax.set_xlabel("Zeit-Index der Datei (pro ID-Chip ordinale Reihenfolge)", fontsize=12)
        # Setzen der Achsenbeschriftung auf logarithmisch, falls gewünscht
        #ax.set_yscale('log')

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
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_grouped_boxplot.png'), dpi=300)
        plt.close()

def main():
    parent_dir = input("Bitte geben Sie den Pfad zum übergeordneten Verzeichnis ein: ").strip()

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