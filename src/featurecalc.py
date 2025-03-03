# Projektverzeichnis/
# │
# ├──  Daten/
# │   ├──  BDNF/
# │   │   ├──  ID2024-01/
# │   │   │   ├── Datei1.bxr
# │   │   │   ├── Datei2.bxr
# │   │   │   ├── ...
# │   │   ├──  ID2024-03/
# │   │   │   ├── Datei1.bxr
# │   │   │   ├── Datei2.bxr
# │   │   │   ├── ...
# │   │   ├── ...
# │   │
# │   ├──  SHAM/
# │   │   ├──  ID2024-02/
# │   │   │   ├── Datei1.bxr
# │   │   │   ├── Datei2.bxr
# │   │   │   ├── ...
# │   │   ├──  ID2024-04/
# │   │   │   ├── Datei1.bxr
# │   │   │   ├── Datei2.bxr
# │   │   │   ├── ...
# │   │   ├── ...

# Für Input bitte Pfad zu Daten angeben: C:\Projektverzeichnis\Daten
# Zudem werden .npz-Dateien mit den berechneten Features erstellt und gespeichert. Diese können dann für weitere Analysen verwendet werden. (bspw. Connectivity-Graph)

import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import quantities as pq
from neo import SpikeTrain
from elephant.spike_train_synchrony import spike_contrast
from viziphant.spike_train_synchrony import plot_spike_contrast
from elephant.functional_connectivity import total_spiking_probability_edges
from elephant.conversion import BinnedSpikeTrain
from collections import defaultdict

# GPU-Beschleunigung
use_gpu = 'n' #input("Möchten Sie GPU-Beschleunigung nutzen? (y/n): ").strip().lower()

if use_gpu == 'y':
    import cupy as cp
    xp = cp
    asnumpy = cp.asnumpy
    print("GPU-Beschleunigung wurde aktiviert (NVIDIA GPU erforderlich).")
else:
    xp = np
    def asnumpy(x):
        return x
    print("Keine GPU-Beschleunigung. Das Skript läuft ausschließlich auf der CPU.")

# Berechnung von Features pro Datei
def process_file(file, well_id='Well_A1'):
    features = {}
    if well_id + '/SpikeTimes' in file:
        spike_times = np.array(file[well_id + '/SpikeTimes'])
        spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

        if 'SamplingRate' in file.attrs:
            sampling_rate = file.attrs['SamplingRate']
        else:
            raise ValueError("Sampling rate not found in file attributes.")

        spike_times_sec = spike_times / sampling_rate

        # Spike-Bursts
        if well_id + '/SpikeBurstTimes' in file:
            burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
            burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            burst_times_sec = burst_times / sampling_rate
        else:
            burst_times = np.array([])
            burst_channels = np.array([])
            burst_times_sec = np.array([])

        # Network-Bursts
        if well_id + '/SpikeNetworkBurstTimes' in file:
            network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
            network_burst_times_sec = network_burst_times / sampling_rate
        else:
            network_burst_times = np.array([])
            network_burst_times_sec = np.array([])

        # Gesamtdauer
        if spike_times_sec.size > 0:
            total_duration_sec = spike_times_sec[-1] - spike_times_sec[0]
        else:
            total_duration_sec = 0

        # Anzahl Spikes, Spike-Rate
        num_spikes = len(spike_times_sec)
        spike_rate = num_spikes / total_duration_sec if total_duration_sec > 0 else 0

        # ISI - Inter-Spike-Interval
        if num_spikes > 1:
            spike_times_sec_xp = xp.asarray(spike_times_sec)
            isi_xp = xp.diff(spike_times_sec_xp) * 1000
            isi_mean = asnumpy(xp.mean(isi_xp))
            isi_std = asnumpy(xp.std(isi_xp))
        else:
            isi_mean = 0
            isi_std = 0

        # Entropie der Spike-Verteilung
        if num_spikes > 0 and total_duration_sec > 0:
            bin_size = 0.1
            num_bins = int(np.ceil(total_duration_sec / bin_size))
            bin_edges = np.linspace(spike_times_sec[0], spike_times_sec[0] + num_bins * bin_size, num_bins + 1)
            spike_counts, _ = np.histogram(spike_times_sec, bins=bin_edges)
            spike_probabilities = spike_counts / np.sum(spike_counts)
            spike_probabilities = spike_probabilities[spike_probabilities > 0]

            spike_probabilities_xp = xp.asarray(spike_probabilities)
            spike_entropy_xp = -xp.sum(spike_probabilities_xp * xp.log2(spike_probabilities_xp))
            spike_entropy = asnumpy(spike_entropy_xp)
        else:
            spike_entropy = 0

        # Network-Bursts
        num_network_bursts = len(network_burst_times_sec)
        network_burst_rate = num_network_bursts / total_duration_sec if total_duration_sec > 0 else 0

        if (network_burst_times_sec.size > 0 and
                network_burst_times_sec.ndim == 2 and
                network_burst_times_sec.shape[1] >= 2):
            network_burst_durations = (network_burst_times_sec[:, 1] - network_burst_times_sec[:, 0]) * 1000
            network_burst_duration_mean = np.mean(network_burst_durations)
            network_burst_duration_std = np.std(network_burst_durations)
        else:
            network_burst_durations = np.array([])
            network_burst_duration_mean = 0
            network_burst_duration_std = 0

        # Spike-Bursts
        if (burst_times_sec.size > 0 and
                burst_times_sec.ndim == 2 and
                burst_times_sec.shape[1] >= 2):
            burst_durations = (burst_times_sec[:, 1] - burst_times_sec[:, 0]) * 1000
            burst_rate = len(burst_durations) / total_duration_sec if total_duration_sec > 0 else 0
            burst_duration_mean = np.mean(burst_durations)
            burst_duration_std = np.std(burst_durations)
        else:
            burst_durations = np.array([])
            burst_rate = 0
            burst_duration_mean = 0
            burst_duration_std = 0

        # Inter-Burst-Interval (IBI)
        if burst_times_sec.shape[0] > 1 and burst_times_sec.shape[1] >= 1:
            burst_start_times = burst_times_sec[:, 0]
            ibi = np.diff(burst_start_times) * 1000
            ibi_mean = np.mean(ibi)
        else:
            ibi_mean = 0

        # Inter-Network-Burst-Interval (INBI)
        if (network_burst_times_sec.shape[0] > 1 and
                network_burst_times_sec.ndim == 2 and
                network_burst_times_sec.shape[1] >= 1):
            network_burst_start_times = network_burst_times_sec[:, 0]
            inbi = np.diff(network_burst_start_times) * 1000
            inbi_mean = np.mean(inbi)
        else:
            inbi_mean = 0

        # Anzahl der Kanäle
        unique_channels = np.unique(spike_channels)
        num_channels = len(unique_channels)

        # Pearson-Korrelation
        if num_channels > 1:
            channel_spike_times = {ch: spike_times_sec[spike_channels == ch] for ch in unique_channels}
            bin_size_ms = 50
            bin_size_s = bin_size_ms / 1000.0
            num_bins = int(np.ceil(total_duration_sec / bin_size_s))

            # Spike-Matrix pro Kanal
            spike_trains_xp = xp.zeros((num_channels, num_bins), dtype=xp.float32)
            for idx, ch in enumerate(unique_channels):
                ch_spike_times = channel_spike_times[ch]
                if ch_spike_times.size > 0:
                    ch_spike_indices_xp = ((ch_spike_times - spike_times_sec[0]) / bin_size_s).astype(xp.int32)
                    ch_spike_indices_xp = ch_spike_indices_xp[ch_spike_indices_xp < num_bins]
                    spike_trains_xp[idx, ch_spike_indices_xp] = 1

            if num_bins > 0:
                pearson_corr_matrix_xp = xp.corrcoef(spike_trains_xp)
                pearson_corr_matrix = asnumpy(pearson_corr_matrix_xp)

                # Obere Dreiecksmatrix zum Mittelwert
                upper_triangle_indices = np.triu_indices(num_channels, k=1)
                mean_pearson_corr = np.mean(pearson_corr_matrix[upper_triangle_indices])

                # Schwellenwert und Anzahl Verbindungen
                connectivity_threshold = 0.20
                connectivity_matrix = (pearson_corr_matrix > connectivity_threshold).astype(int)
                np.fill_diagonal(connectivity_matrix, 0)
                num_connections = np.sum(connectivity_matrix) / 2
            else:
                pearson_corr_matrix = np.array([])
                connectivity_matrix = np.array([])
                mean_pearson_corr = 0
                num_connections = 0
        else:
            pearson_corr_matrix = np.array([])
            connectivity_matrix = np.array([])
            mean_pearson_corr = 0
            num_connections = 0

        # Spike Contrast (https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_synchrony/elephant.spike_train_synchrony.spike_contrast.html#elephant.spike_train_synchrony.spike_contrast)
        spike_trains = []
        if spike_times_sec.size > 0:
            t_start = spike_times_sec.min() * pq.s
            t_stop = spike_times_sec.max() * pq.s
        else:
            t_start = 0 * pq.s
            t_stop = 1 * pq.s

        for ch in unique_channels:
            ch_spike_times = spike_times_sec[spike_channels == ch]
            if ch_spike_times.size > 0:
                st = SpikeTrain(ch_spike_times * pq.s, t_start=t_start, t_stop=t_stop)
                spike_trains.append(st)

        try:
            spike_contrast_result = spike_contrast(
                spike_trains,
                t_start=t_start,
                t_stop=t_stop,
                bin_shrink_factor=0.9,
                return_trace=True
            )
            synchrony_with_trace = spike_contrast_result[0]
            spike_contrast_trace = spike_contrast_result[1]
        except Exception as e:
            print(f"Fehler bei der Berechnung von spike_contrast mit Trace: {e}")
            synchrony_with_trace = 0
            spike_contrast_trace = None

        # Total Spiking Probability Edges (TSPE) (https://elephant.readthedocs.io/en/latest/reference/_toctree/functional_connectivity/elephant.functional_connectivity.total_spiking_probability_edges.html)
        try:
            binned_ST = BinnedSpikeTrain(spike_trains, bin_size=1*pq.s, n_bins=None, t_start=None, 
                                         t_stop=None, tolerance=1e-08, sparse_format='csr')
            tsp_matrix, tsp_delay = total_spiking_probability_edges(
                spike_trains=binned_ST,
                surrounding_window_sizes=[3, 4, 5, 6, 7, 8],  # Angepasste Bin-Größe
                observed_window_sizes=[2, 3, 4, 5, 6],
                crossover_window_sizes=[0],
                max_delay=25,
                normalize=False
            )
            #print("TSPE-Matrix:", tsp_matrix)
            # tsp_matrix ist ein nxn numpy Array
            mean_tsp = np.mean(tsp_matrix)
            #print("Durchschnittliche TSPE:", mean_tsp)
        except Exception as e:
            print(f"Fehler bei der Berechnung von TSPE: {e}")
            tsp_matrix = np.array([])
            mean_tsp = np.nan

        # Features speichern
        features = {
            'Spike Rate (Hz)': spike_rate,
            'Number of Spikes': num_spikes,
            'ISI Mean (ms)': isi_mean,
            'ISI Std (ms)': isi_std,
            'Entropy': spike_entropy,
            'Burst Rate (Hz)': burst_rate,
            'Burst Duration Mean (ms)': burst_duration_mean,
            'Burst Duration Std (ms)': burst_duration_std,
            'Inter Burst Interval (IBI) (ms)': ibi_mean,
            'Network Burst Rate (Hz)': network_burst_rate,
            'Network Burst Duration Mean (ms)': network_burst_duration_mean,
            'Network Burst Duration Std (ms)': network_burst_duration_std,
            'Inter Network Burst Interval (INBI) (ms)': inbi_mean,
            'Synchrony (Mean Pearson-Correlation)': mean_pearson_corr,
            'Connectivity (Number of Connections)': num_connections,
            'Synchrony (Spike Contrast)': synchrony_with_trace,
            'Mean TSPE': mean_tsp
        }

        # Falls Spike Contrast Trace existiert, zusätzliche Ergebnisse
        if spike_contrast_trace is not None and len(spike_contrast_trace) == 4:
            contrast, active_spiketrains, synchrony_trace, bin_size = spike_contrast_trace
            features['Spike Contrast Trace - Contrast'] = contrast
            features['Spike Contrast Trace - Active Spiketrains'] = active_spiketrains
            features['Spike Contrast Trace - Synchrony'] = synchrony_trace
            features['Spike Contrast Trace - Bin Size'] = bin_size

        # Speichern der Ergebnisse
        output_feature_path = os.path.splitext(file.filename)[0] + '_features.npz'
        np.savez(
            output_feature_path,
            features=features,
            pearson_corr_matrix=pearson_corr_matrix,
            connectivity_matrix=connectivity_matrix,
            unique_channels=unique_channels,
            num_channels=num_channels,
            tsp_matrix=tsp_matrix,
            tsp_delay=tsp_delay
        )

        # Plot Spike Contrast Trace
        if spike_contrast_trace is not None and len(spike_contrast_trace) == 4:
            filename = os.path.splitext(os.path.basename(file.filename))[0]
            plot_spike_contrast_viziphant(spike_contrast_trace,
                                          #spike_trains, uncomment if rasterplot wanted
                                          os.path.dirname(output_feature_path),
                                          filename)
        return features
    else:
        print(f"Warnung: 'SpikeTimes' nicht in {well_id} vorhanden.")
        return None

# Plotten der Spike Contrast Trace
def plot_spike_contrast_viziphant(spike_contrast_trace, output_dir, filename):
#def plot_spike_contrast_viziphant(spike_contrast_trace,spike_trains, output_dir, filename):
    try:
        plot_spike_contrast(spike_contrast_trace, filename=filename)
        #plot_spike_contrast(spike_contrast_trace, spike_trains, filename)
        plot_path = os.path.join(output_dir, f"{filename}_spike_contrast_trace.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Fehler beim Plotten mit viziphant: {e}")

# Erstellt gruppierte Boxplots: X-Achse = Zeitindex (Dateireihenfolge), Gruppen = LSD/SHAM/...
def plot_feature_values_over_time(all_features_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_name, time_dict in all_features_data.items():
        time_indices_sorted = sorted(time_dict.keys(), key=lambda x: int(x))
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
        for time_idx, group_values_list in enumerate(data_for_boxplot):
            for group_idx, group_values in enumerate(group_values_list):
                position = time_idx + group_idx * width - (width * (n_groups - 1) / 2)
                bp = ax.boxplot(group_values,
                                positions=[position],
                                widths=width,
                                patch_artist=True,
                                manage_ticks=False)
                for box in bp['boxes']:
                    box.set(facecolor=colors[group_idx % len(colors)])

        ax.set_title(feature_name, fontsize=14)
        ax.set_ylabel(feature_name, fontsize=12)
        ax.set_xticks(range(n_times))
        ax.set_xticklabels(time_indices_sorted, rotation=45, fontsize=10)
        ax.set_xlabel("Zeit-Index der Datei (pro ID-Chip ordinale Reihenfolge)", fontsize=12)

        legend_patches = [
            plt.Line2D([0], [0], color=colors[idx % len(colors)],
                       label=group, marker='s', linestyle='', markersize=10)
            for idx, group in enumerate(group_names)
        ]
        ax.legend(handles=legend_patches, title="Gruppen", loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=len(group_names), fontsize=10, title_fontsize=12)

        plt.tight_layout()
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_grouped_boxplot.png'), dpi=300)
        plt.close()

# Hauptfunktion
def main():
    parent_dir = "data"  

    # Oberste Ebene: Gruppen (z.B. LSD, SHAM, ...)
    groups = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # all_features_data[feature][time_index][group_name] = [Werte...]
    all_features_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Verarbeitung jeder Gruppe
    for group in groups:
        group_path = os.path.join(parent_dir, group)

        # Innerhalb der Gruppe nach Unterordnern (Chips) suchen
        chip_dirs = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        for chip_dir in chip_dirs:
            chip_path = os.path.join(group_path, chip_dir)

            # .bxr-Dateien in aufsteigender Reihenfolge
            bxr_files = sorted([f for f in os.listdir(chip_path) if f.endswith('_NB.bxr')])

            # Jede Datei repräsentiert einen Zeit-Index
            for idx, bxr_file in enumerate(bxr_files, start=1):
                bxr_path = os.path.join(chip_path, bxr_file)
                print(f"Verarbeite: Gruppe={group}, Chip={chip_dir}, Datei={bxr_file}, Zeit-Index={idx}")

                with h5py.File(bxr_path, 'r') as file:
                    features = process_file(file)
                    if features is not None:
                        for feat_key, feat_val in features.items():
                            all_features_data[feat_key][str(idx)][group].append(feat_val)

    # Boxplots über Gruppen und Zeitverläufe
    #inter_group_output_dir = os.path.join(parent_dir, 'Inter_Group_Boxplots')
    #plot_feature_values_over_time(all_features_data, inter_group_output_dir)

    print("Fertig. Alle Features wurden berechnet.")

if __name__ == "__main__":
    main()
