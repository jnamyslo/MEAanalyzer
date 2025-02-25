# NOT YET IN USE!!

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import cupy as cp  # Cupy importieren
import quantities as pq
from neo import SpikeTrain
from elephant.spike_train_synchrony import spike_contrast
from elephant.functional_connectivity import total_spiking_probability_edges
from elephant.conversion import BinnedSpikeTrain
from viziphant.spike_train_synchrony import plot_spike_contrast

# Funktion zum Plotten des Rasterplots mit Spikes, Bursts und Netzwerk-Bursts
def raster_plots_main(filepath, save_as='svg', vector_output=False, dpi=600, output_dir='Rasterplots'):
    with h5py.File(filepath, 'r') as file:
        well_id = 'Well_A1'
        if well_id + '/SpikeTimes' in file:
            # Lesen der Spike-Daten
            spike_times = np.array(file[well_id + '/SpikeTimes'])
            spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

            # Sampling-Rate aus den Dateiattributen lesen
            if 'SamplingRate' in file.attrs:
                sampling_rate = file.attrs['SamplingRate']
            else:
                raise ValueError("Sampling rate not found in file attributes.")

            spike_times_sec = spike_times / sampling_rate

            # Lesen der Burst-Daten
            if well_id + '/SpikeBurstTimes' in file:
                burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
                burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            else:
                burst_times = np.array([])
                burst_channels = np.array([])

            # Lesen der Netzwerk-Burst-Daten
            if well_id + '/SpikeNetworkBurstTimes' in file:
                network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
                network_burst_times = network_burst_times / sampling_rate  # In Sekunden umwandeln
            else:
                network_burst_times = np.array([])

            # Einzigartige Kanäle und Mapping
            unique_channels = np.unique(spike_channels)
            channel_map = {ch: idx for idx, ch in enumerate(unique_channels)}

            # Vorbereiten der Rasterdaten
            spike_raster_data = [[] for _ in range(len(unique_channels))]
            for time, channel in zip(spike_times_sec, spike_channels):
                spike_raster_data[channel_map[channel]].append(time)

            burst_raster_data = [[] for _ in range(len(unique_channels))]
            if burst_times.size > 0:
                for burst_time, channel in zip(burst_times, burst_channels):
                    burst_raster_data[channel_map[channel]].append(burst_time[0] / sampling_rate)

            network_burst_raster_data = [[] for _ in range(len(unique_channels))]
            if network_burst_times.size > 0:
                for net_burst_time in network_burst_times:
                    for channel_bursts in network_burst_raster_data:
                        channel_bursts.append(net_burst_time[0])

            # Kombinierter Rasterplot
            create_raster_plot(spike_raster_data, burst_raster_data, network_burst_raster_data, filepath, output_dir, vector_output, dpi)

            # Rasterplot mit Spikes, Bursts und NetworkBursts untereinander
            create_separate_raster_plot(spike_raster_data, burst_raster_data, network_burst_raster_data, filepath, output_dir, vector_output, dpi)
        else:
            print(f"Warnung: 'SpikeTimes' nicht in {filepath} vorhanden.")

# Funktion zum Erstellen des kombinierten Rasterplots
def create_raster_plot(spike_raster_data, burst_raster_data, network_burst_raster_data, filepath, output_dir, vector_output, dpi):
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_raster_data, colors='black', linelengths=0.9)
    if any(burst_raster_data):
        plt.eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)
    if any(network_burst_raster_data):
        plt.eventplot(network_burst_raster_data, colors=[[1, 0, 0, 0.5]], linelengths=1.0)

    filename = os.path.basename(filepath)
    plt.title(f'Raster Plot: {filename}')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Kanalindex')
    plt.tight_layout()

    # Legende hinzufügen
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Spikes'),
        Line2D([0], [0], color='lightgreen', lw=2, label='Spike-Bursts'),
        Line2D([0], [0], color=[1, 0, 0, 0.5], lw=2, label='Networkbursts')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_basename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_raster_plot')
    plt.savefig(f'{output_basename}.png', format='png', dpi=dpi, bbox_inches='tight')
    if vector_output:
        plt.savefig(f'{output_basename}.svg', format='svg', bbox_inches='tight')
    plt.close()

# Funktion zum Erstellen des separaten Rasterplots
def create_separate_raster_plot(spike_raster_data, burst_raster_data, network_burst_raster_data, filepath, output_dir, vector_output, dpi):
    num_channels = len(spike_raster_data)
    filename = os.path.basename(filepath)
    output_basename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_separate_raster_plot')

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Spikes
    axes[0].eventplot(spike_raster_data, colors='black', linelengths=0.9)
    axes[0].set_ylabel('Kanalindex')
    axes[0].set_title('Spikes')
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(-0.5, num_channels - 0.5)

    # Bursts
    if any(burst_raster_data):
        axes[1].eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)
    axes[1].set_ylabel('Kanalindex')
    axes[1].set_title('Spike-Bursts')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(-0.5, num_channels - 0.5)

    # Network Bursts
    if any(network_burst_raster_data):
        axes[2].eventplot(network_burst_raster_data, colors=[[1, 0, 0, 0.5]], linelengths=1.0)
    axes[2].set_ylabel('Kanalindex')
    axes[2].set_title('Networkbursts')
    axes[2].set_xlabel('Zeit (s)')
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(-0.5, num_channels - 0.5)

    plt.tight_layout()

    plt.savefig(f'{output_basename}.png', format='png', dpi=dpi, bbox_inches='tight')
    if vector_output:
        plt.savefig(f'{output_basename}.svg', format='svg', bbox_inches='tight')
    plt.close()

# Funktion zum Berechnen von Spike-Features und weiteren Features
def process_file(file, well_id='Well_A1'):
    features = {}
    if well_id + '/SpikeTimes' in file:
        # Lesen der Spike-Daten
        spike_times = np.array(file[well_id + '/SpikeTimes'])
        spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

        if 'SamplingRate' in file.attrs:
            sampling_rate = file.attrs['SamplingRate']
        else:
            raise ValueError("Sampling rate not found in file attributes.")

        spike_times_sec = spike_times / sampling_rate

        # Lesen der Burst-Daten
        if well_id + '/SpikeBurstTimes' in file:
            burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
            burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            burst_times_sec = burst_times / sampling_rate  # In Sekunden umwandeln
        else:
            burst_times = np.array([])
            burst_channels = np.array([])
            burst_times_sec = np.array([])

        # Lesen der Netzwerk-Burst-Daten
        if well_id + '/SpikeNetworkBurstTimes' in file:
            network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
            network_burst_times_sec = network_burst_times / sampling_rate  # In Sekunden umwandeln
        else:
            network_burst_times = np.array([])
            network_burst_times_sec = np.array([])

        # Gesamtdauer der Aufnahme in Sekunden
        if spike_times_sec.size > 0:
            total_duration_sec = spike_times_sec[-1] - spike_times_sec[0]
        else:
            total_duration_sec = 0

        # Anzahl der Spikes und Spike-Rate
        num_spikes = len(spike_times_sec)
        spike_rate = num_spikes / total_duration_sec if total_duration_sec > 0 else 0

        # InterSpikeInterval (ISI)
        if num_spikes > 1:
            spike_times_sec_gpu = cp.asarray(spike_times_sec)
            isi_gpu = cp.diff(spike_times_sec_gpu) * 1000  # Umwandlung in Millisekunden
            isi_mean = cp.asnumpy(cp.mean(isi_gpu))
            isi_std = cp.asnumpy(cp.std(isi_gpu))
        else:
            isi_mean = 0
            isi_std = 0

        # Entropie der Spike-Zeiten
        if num_spikes > 0 and total_duration_sec > 0:
            bin_size = 0.1  # Sekunden
            num_bins = int(np.ceil(total_duration_sec / bin_size))
            bin_edges = np.linspace(spike_times_sec[0], spike_times_sec[0] + num_bins * bin_size, num_bins + 1)
            spike_counts, _ = np.histogram(spike_times_sec, bins=bin_edges)
            spike_probabilities = spike_counts / np.sum(spike_counts)
            spike_probabilities = spike_probabilities[spike_probabilities > 0]

            spike_probabilities_gpu = cp.asarray(spike_probabilities)
            spike_entropy_gpu = -cp.sum(spike_probabilities_gpu * cp.log2(spike_probabilities_gpu))
            spike_entropy = cp.asnumpy(spike_entropy_gpu)
        else:
            spike_entropy = 0

        # Netzwerk-Burst-Rate
        num_network_bursts = len(network_burst_times_sec)
        network_burst_rate = num_network_bursts / total_duration_sec if total_duration_sec > 0 else 0

        # Netzwerk-Burst-Dauer
        if network_burst_times_sec.size > 0 and network_burst_times_sec.ndim == 2 and network_burst_times_sec.shape[1] >= 2:
            network_burst_durations = (network_burst_times_sec[:, 1] - network_burst_times_sec[:, 0]) * 1000  # ms
            network_burst_duration_mean = np.mean(network_burst_durations)
            network_burst_duration_std = np.std(network_burst_durations)
        else:
            network_burst_durations = np.array([])
            network_burst_duration_mean = 0
            network_burst_duration_std = 0

        # Burst-Rate, Burst-Dauer Mean und Std
        if burst_times_sec.size > 0 and burst_times_sec.ndim == 2 and burst_times_sec.shape[1] >= 2:
            burst_durations = (burst_times_sec[:, 1] - burst_times_sec[:, 0]) * 1000  # ms
            burst_rate = len(burst_durations) / total_duration_sec if total_duration_sec > 0 else 0
            burst_duration_mean = np.mean(burst_durations)
            burst_duration_std = np.std(burst_durations)
        else:
            burst_durations = np.array([])
            burst_rate = 0
            burst_duration_mean = 0
            burst_duration_std = 0

        # Inter Burst Interval (IBI) berechnen
        if burst_times_sec.shape[0] > 1 and burst_times_sec.shape[1] >= 1:
            burst_start_times = burst_times_sec[:, 0]
            ibi = np.diff(burst_start_times) * 1000  # ms
            ibi_mean = np.mean(ibi)
        else:
            ibi_mean = 0

        # Inter Network Burst Interval (INBI) berechnen
        if network_burst_times_sec.shape[0] > 1 and network_burst_times_sec.ndim == 2 and network_burst_times_sec.shape[1] >= 1:
            network_burst_start_times = network_burst_times_sec[:, 0]
            inbi = np.diff(network_burst_start_times) * 1000  # ms
            inbi_mean = np.mean(inbi)
        else:
            inbi_mean = 0

        # Synchronität (Pearson-Korrelation)
        unique_channels = np.unique(spike_channels)
        num_channels = len(unique_channels)
        if num_channels > 1:
            channel_spike_times = {ch: spike_times_sec[spike_channels == ch] for ch in unique_channels}
            bin_size_ms = 50  # ms
            bin_size_s = bin_size_ms / 1000.0
            num_bins = int(np.ceil(total_duration_sec / bin_size_s))

            spike_trains_gpu = cp.zeros((num_channels, num_bins), dtype=cp.float32)
            for idx, ch in enumerate(unique_channels):
                ch_spike_times = channel_spike_times[ch]
                if ch_spike_times.size > 0:
                    ch_spike_indices_gpu = ((ch_spike_times - spike_times_sec[0]) / bin_size_s).astype(cp.int32)
                    ch_spike_indices_gpu = ch_spike_indices_gpu[ch_spike_indices_gpu < num_bins]
                    spike_trains_gpu[idx, ch_spike_indices_gpu] = 1

            if num_bins > 0:
                pearson_corr_matrix_gpu = cp.corrcoef(spike_trains_gpu)
                pearson_corr_matrix = cp.asnumpy(pearson_corr_matrix_gpu)
                upper_triangle_indices = np.triu_indices(num_channels, k=1)
                mean_pearson_corr = np.mean(pearson_corr_matrix[upper_triangle_indices])

                connectivity_threshold = 0.21  # angepasst-JN
                connectivity_matrix = (pearson_corr_matrix > connectivity_threshold).astype(int)
                np.fill_diagonal(connectivity_matrix, 0)
                num_connections = np.sum(connectivity_matrix) / 2
            else:
                mean_pearson_corr = 0
                num_connections = 0
        else:
            mean_pearson_corr = 0
            num_connections = 0

        # Synchronität (Spike-Contrast)
        # Vorbereitung der SpikeTrains für spike_contrast und TSPE => neo.SpikeTrains Format
        spike_trains = []
        if spike_times_sec.size > 0:
            t_start = spike_times_sec.min() * pq.s
            t_stop = spike_times_sec.max() * pq.s
        else:
            t_start = 0 * pq.s
            t_stop = 1 * pq.s  # Mindestdauer von 1s, falls keine Spikes

        for ch in unique_channels:
            ch_spike_times = spike_times_sec[spike_channels == ch]
            if ch_spike_times.size > 0:
                st = SpikeTrain(ch_spike_times * pq.s, t_start=t_start, t_stop=t_stop)
                spike_trains.append(st)

        # Berechnung der Spike Contrast Synchrony mit Trace
        try:
            spike_contrast_result = spike_contrast(
                spike_trains,
                t_start=t_start,
                t_stop=t_stop,
                #min_bin=10 * pq.ms,
                bin_shrink_factor=0.9,
                return_trace=True
            )
            synchrony_with_trace = spike_contrast_result[0]  # Einzelner Synchrony Wert
            #print(synchrony_with_trace)
            spike_contrast_trace = spike_contrast_result[1]  # Rückgabe der vier Rückgabewerte von spike_contrast
            #print(spike_contrast_trace)
        except Exception as e:
            print(f"Fehler bei der Berechnung von spike_contrast mit Trace: {e}")
            spike_contrast_trace = None

        # Berechnung der Total Spiking Probability Edges (TSPE) Konnektivitätsmatrix
        try:
            binned_ST = BinnedSpikeTrain(spike_trains, bin_size=1*pq.s, n_bins=None, t_start=None, 
                                         t_stop=None, tolerance=1e-08, sparse_format='csr')
            tsp_matrix = total_spiking_probability_edges(
                spike_trains=binned_ST,
                surrounding_window_sizes=[3, 4, 5, 6, 7, 8],  # Angepasste Bin-Größe
                observed_window_sizes=[2, 3, 4, 5, 6],
                crossover_window_sizes=[0],
                max_delay=25,
                normalize=False
            )
            # tsp_matrix ist ein nxn numpy Array
            mean_tsp = np.mean(tsp_matrix)
            print("Durchschnittliche TSPE:", mean_tsp)
        except Exception as e:
            print(f"Fehler bei der Berechnung von TSPE: {e}")
            tsp_matrix = np.array([])
            mean_tsp = np.nan

        # ----------------------- Ende der Integration -----------------------

        # Zusammenstellen der Features
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
            'Synchrony (Spike Contrast)': synchrony_with_trace,  # Hinzufügen des spike_contrast Wertes mit Trace
            'Mean TSPE': mean_tsp  # Hinzugefügt
        }

        # Hinzufügen der Spike Contrast Trace in Features nach Plausibilitätscheck
        if spike_contrast_trace is not None:
            if len(spike_contrast_trace) == 4:
                contrast, active_spiketrains, synchrony_trace, bin_size = spike_contrast_trace
                features['Spike Contrast Trace - Contrast'] = contrast
                features['Spike Contrast Trace - Active Spiketrains'] = active_spiketrains
                features['Spike Contrast Trace - Synchrony'] = synchrony_trace
                features['Spike Contrast Trace - Bin Size'] = bin_size
            else:
                print("Warnung: Spike Contrast Trace hat unerwartete Länge.")

        # Speichern der Features und Matrizen
        output_feature_path = os.path.splitext(file.filename)[0] + '_features.npz'
        np.savez(
            output_feature_path,
            features=features,
            pearson_corr_matrix=pearson_corr_matrix if num_channels > 1 else np.array([]),
            connectivity_matrix=connectivity_matrix if num_channels > 1 else np.array([]),
            tsp_matrix=tsp_matrix if tsp_matrix.size > 0 else np.array([]),  # Hinzugefügt
            unique_channels=unique_channels,
            num_channels=num_channels
        )

        # Plotten der Spike Contrast Trace mittels Viziphant
        if spike_contrast_trace is not None and len(spike_contrast_trace) == 4:
            plot_spike_contrast_viziphant(spike_contrast_trace, os.path.dirname(output_feature_path), os.path.splitext(os.path.basename(file.filename))[0])
        return features
    else:
        print(f"Warnung: 'SpikeTimes' nicht in {well_id} vorhanden.")
        return None

# Funktion zum Plotten der Features als Boxplots, um verschiedene Dateien zu vergleichen
def plot_feature_values(feature_data, labels, output_dir, title_suffix=''):
    for feature_name, data in feature_data.items():
        plt.figure(figsize=(10, 6))

        # Boxplot erstellen und alle Dateien/Chips in einem Diagramm darstellen
        plt.boxplot(data, patch_artist=True, tick_labels=labels)

        plt.title(f'{feature_name} {title_suffix}')
        plt.ylabel(feature_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Boxplot speichern
        filename_safe = feature_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'{filename_safe}_boxplot{title_suffix.replace(" ", "_")}.png'))
        plt.close()

# Funktion zum Plotten der Spike Contrast Trace mittels Viziphant
def plot_spike_contrast_viziphant(spike_contrast_trace, output_dir, filename):
    try:
        # Plot erstellen
        plot_spike_contrast(spike_contrast_trace)

        # Plot speichern
        plot_path = os.path.join(output_dir, f"{filename}_spike_contrast_trace_viziphant.png")
        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        print(f"Fehler beim Plotten mit viziphant: {e}")

# Hauptfunktion
def main():
    parent_dir = input("Bitte geben Sie den Pfad zum übergeordneten Verzeichnis mit den Chip-Ordnern ein: ")
    chip_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith('ID2024')]
    if not chip_dirs:
        print("Es wurden keine Chip-Ordner gefunden.")
        return

    print("Verfügbare Chips:")
    print("0. Alle Chips")
    for i, chip_dir in enumerate(chip_dirs):
        print(f"{i + 1}. {chip_dir}")

    chip_auswahl = input("Bitte wählen Sie die Chips aus, die verarbeitet werden sollen (z.B. '1' oder '1,2,3' oder '0' für alle): ")
    if chip_auswahl == '0':
        chip_indices = list(range(len(chip_dirs)))
    else:
        try:
            chip_indices = [int(x.strip()) - 1 for x in chip_auswahl.split(',')]
            # Validierung der Indizes
            chip_indices = [idx for idx in chip_indices if 0 <= idx < len(chip_dirs)]
            if not chip_indices:
                print("Keine gültigen Chip-Indizes ausgewählt.")
                return
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie Zahlen wie '1,2,3' oder '0' ein.")
            return

    # Gesamtdaten für inter-Chip-Vergleich
    chip_feature_data = {}
    chip_labels = []

    for idx in chip_indices:
        chip_dir = chip_dirs[idx]
        chip_path = os.path.join(parent_dir, chip_dir)
        print(f"\nVerarbeite Chip: {chip_dir}...")

        dateien = [f for f in os.listdir(chip_path) if f.endswith('.bxr')]
        if not dateien:
            print(f"Keine .bxr-Dateien in {chip_dir} gefunden.")
            continue

        feature_data = {}
        file_labels = []

        output_dir = os.path.join(chip_path, 'FeaturePlots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rasterplots_output_dir = os.path.join(chip_path, 'Rasterplots')
        if not os.path.exists(rasterplots_output_dir):
            os.makedirs(rasterplots_output_dir)

        for datei_name in dateien:
            dateipfad = os.path.join(chip_path, datei_name)
            print(f"Verarbeite Datei: {datei_name}...")

            with h5py.File(dateipfad, 'r') as file:
                features = process_file(file)
                if features is not None:
                    # Features sammeln
                    for key, value in features.items():
                        if key not in feature_data:
                            feature_data[key] = []
                        feature_data[key].append(value)
                    file_labels.append(f'{datei_name}')

                    # Rasterplot erzeugen
                    raster_plots_main(dateipfad, vector_output=False, dpi=600, output_dir=rasterplots_output_dir)
                    print(f"Fertig mit {datei_name}.")
                else:
                    print(f"Warnung: 'SpikeTimes' nicht in {datei_name} vorhanden.")

        # Intra-Chip-Variabilität plotten
        if feature_data:
            # Daten in richtigem Format für Boxplot
            feature_data_for_plot = {k: [v for v in v_list] for k, v_list in feature_data.items()}
            plot_feature_values(feature_data_for_plot, file_labels, output_dir, title_suffix=f'in {chip_dir}')

            # Gesamtdaten für inter-Chip-Vergleich sammeln (alle Werte)
            for key, values in feature_data.items():
                if key not in chip_feature_data:
                    chip_feature_data[key] = []
                chip_feature_data[key].append(values)
            chip_labels.append(chip_dir)

            print(f"Fertig mit Chip {chip_dir}.\n")
        else:
            print(f"Keine Daten für Chip {chip_dir} gesammelt.")

    # Inter-Chip-Vergleich plotten
    if chip_feature_data:
        inter_chip_output_dir = os.path.join(parent_dir, 'Inter_Chip_Plots')
        if not os.path.exists(inter_chip_output_dir):
            os.makedirs(inter_chip_output_dir)

        # Daten für Boxplots vorbereiten
        feature_data_for_inter_chip_plot = chip_feature_data

        plot_feature_values(feature_data_for_inter_chip_plot, chip_labels, inter_chip_output_dir, title_suffix='zwischen Chips')

        print("Inter-Chip-Vergleich abgeschlossen.")

if __name__ == "__main__":
    main()