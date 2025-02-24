#This code will create raster plots from .bxr files in a given directory.
#The raster plots will be saved as .png and .svg files in a subdirectory called 'Rasterplots'.
#The script will print a warning if the 'SpikeTimes' dataset is not found in the .bxr file.
#The script will also print a warning if the 'SamplingRate' attribute is not found in the .bxr file.
#The script will also create a separate raster plot for spikes, spike bursts, and network bursts.

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Funktion zum Erstellen des Rasterplots
def raster_plots_main(filepath, save_as='svg', vector_output=False, dpi=600, output_dir='Rasterplots'):
    with h5py.File(filepath, 'r') as file:
        well_id = 'Well_A1'
        if well_id + '/SpikeTimes' in file:
            spike_times = np.array(file[well_id + '/SpikeTimes'])
            spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

            # Sampling Rate aus Attribut lesen
            if 'SamplingRate' in file.attrs:
                sampling_rate = file.attrs['SamplingRate']
            else:
                raise ValueError("Sampling rate not found in file attributes.")

            # Zeit in Sekunden
            spike_times_sec = spike_times / sampling_rate

            # Falls SpikeBurstTimes vorhanden
            if well_id + '/SpikeBurstTimes' in file:
                burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
                burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            else:
                burst_times = np.array([])
                burst_channels = np.array([])

            # Falls SpikeNetworkBurstTimes vorhanden
            if well_id + '/SpikeNetworkBurstTimes' in file:
                network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
                network_burst_times = network_burst_times / sampling_rate
            else:
                network_burst_times = np.array([])

            # Zuweisen der Channel-Indizes
            unique_channels = np.unique(spike_channels)
            channel_map = {ch: idx for idx, ch in enumerate(unique_channels)}

            # Spike-Daten pro Kanal sammeln
            spike_raster_data = [[] for _ in range(len(unique_channels))]
            for time, channel in zip(spike_times_sec, spike_channels):
                spike_raster_data[channel_map[channel]].append(time)

            # Burst-Daten pro Kanal sammeln
            burst_raster_data = [[] for _ in range(len(unique_channels))]
            if burst_times.size > 0:
                for burst_time, channel in zip(burst_times, burst_channels):
                    burst_raster_data[channel_map[channel]].append(burst_time[0] / sampling_rate)

            # Netzwerk-Burst-Daten pro Kanal sammeln
            network_burst_raster_data = [[] for _ in range(len(unique_channels))]
            if network_burst_times.size > 0:
                for net_burst_time in network_burst_times:
                    for channel_bursts in network_burst_raster_data:
                        channel_bursts.append(net_burst_time[0])

            # Plot erzeugen
            create_raster_plot(spike_raster_data,
                               burst_raster_data,
                               network_burst_raster_data,
                               filepath,
                               output_dir,
                               vector_output,
                               dpi)

            # Separaten Plot erzeugen
            create_separate_raster_plot(spike_raster_data,
                                        burst_raster_data,
                                        network_burst_raster_data,
                                        filepath,
                                        output_dir,
                                        vector_output,
                                        dpi)
        else:
            print(f"Warnung: 'SpikeTimes' nicht in {filepath} vorhanden.")

# Kombinierter Rasterplot
def create_raster_plot(spike_raster_data,
                       burst_raster_data,
                       network_burst_raster_data,
                       filepath,
                       output_dir,
                       vector_output,
                       dpi):
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

# Getrennter Rasterplot
def create_separate_raster_plot(spike_raster_data,
                                burst_raster_data,
                                network_burst_raster_data,
                                filepath,
                                output_dir,
                                vector_output,
                                dpi):
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

    # Spike-Bursts
    if any(burst_raster_data):
        axes[1].eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)
    axes[1].set_ylabel('Kanalindex')
    axes[1].set_title('Spike-Bursts')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(-0.5, num_channels - 0.5)

    # Networkbursts
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

# Funktion zum rekursiven Durchsuchen des Verzeichnisses und Erstellen der Rasterplots
def generate_raster_plots_for_all_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.bxr'):
                filepath = os.path.join(dirpath, filename)
                print(f"Erstelle Rasterplot für Datei: {filepath}")
                output_dir = os.path.join(dirpath, 'Rasterplots')
                try:
                    raster_plots_main(filepath, output_dir=output_dir)
                    print(f"Rasterplot gespeichert in: {output_dir}")
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {filepath}: {e}")

# Hauptprogramm
if __name__ == "__main__":
    root_dir = "data/NB_Spiketrains"
    generate_raster_plots_for_all_files(root_dir)
    print("Alle Rasterplots wurden erstellt.")
