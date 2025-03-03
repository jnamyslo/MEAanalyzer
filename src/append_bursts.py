#Takes long to compute. Chiappalone Ansatz!

import os
import h5py
import numpy as np
import time

def detect_bursts(spike_times, isi_threshold_factor=0.25, min_spikes_in_burst=3):
    """
    Erkennung von Bursts nach Baker et al.
    Parameters:
    - spike_times: Liste der Spike-Zeitpunkte.
    - isi_threshold_factor: Faktor zur Bestimmung des ISI-Schwellenwerts.
    - min_spikes_in_burst: Minimale Anzahl von Spikes in einem Burst.
    
    Returns:
    - burst_times: Liste von Tupeln (Burst-Start, Burst-Ende).
    """
    if len(spike_times) < min_spikes_in_burst:
        return []
    
    spike_times = np.sort(spike_times)
    isis = np.diff(spike_times)
    median_isi = np.median(isis) if len(isis) > 0 else 0
    isi_threshold = isi_threshold_factor * median_isi

    burst_start_indices = []
    burst_end_indices = []
    in_burst = False
    burst_start = 0

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                in_burst = False
                burst_end = i + 1
                if (burst_end - burst_start) >= min_spikes_in_burst:
                    burst_start_indices.append(burst_start)
                    burst_end_indices.append(burst_end)
    
    if in_burst:
        burst_end = len(spike_times)
        if (burst_end - burst_start) >= min_spikes_in_burst:
            burst_start_indices.append(burst_start)
            burst_end_indices.append(burst_end)

    burst_times = [
        (spike_times[start], spike_times[end - 1])
        for start, end in zip(burst_start_indices, burst_end_indices)
    ]
    return burst_times

def detect_network_bursts(spikes_per_channel, bin_size=0.025, threshold=9, min_duration=0.1):
    """
    Erkennt Network-Bursts basierend auf der Synchronisation von Aktivität über mehrere Kanäle.
    Stellt sicher, dass überlappende Network-Bursts zu einem Ereignis zusammengefasst werden.

    Erkennung von Netzwerk-Bursts nach dem Ansatz von Chiappalone:
    
    Parameters:
    - spikes_per_channel: Dictionary {Kanal: [SpikeTimes, ...]} mit Spike-Zeiten pro Kanal
    - bin_size: Zeitfenstergröße in Sekunden für die Analyse der synchronen Aktivität
    - threshold: Schwellenwert für das Produkt aus aktiven Kanälen und Spike-Anzahl
    - min_duration: Minimale Dauer eines Network-Bursts in Sekunden
    
    Returns:
    - network_bursts: Liste von Tupeln (NetworkBurst-Start, NetworkBurst-Ende)
    """
    if not spikes_per_channel:
        return []

    all_spikes = np.concatenate(list(spikes_per_channel.values())) if spikes_per_channel else np.array([])
    if all_spikes.size == 0:
        return []

    min_time, max_time = np.min(all_spikes), np.max(all_spikes)
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    bin_count = len(bins) - 1

    active_channels_per_bin = np.zeros(bin_count, dtype=int)
    total_spikes_per_bin = np.zeros(bin_count, dtype=int)

    for spike_times in spikes_per_channel.values():
        if len(spike_times) == 0:
            continue
        hist, _ = np.histogram(spike_times, bins=bins)
        active_channels_per_bin += (hist > 0).astype(int)
        total_spikes_per_bin += hist

    product = active_channels_per_bin * total_spikes_per_bin
    network_burst_bins = product >= threshold

    network_bursts = []
    in_burst = False
    burst_start = None

    # Durch die Bins iterieren und zusammenhängende True-Bereiche als Network-Burst bestimmen
    for i, is_burst in enumerate(network_burst_bins):
        if is_burst:
            if not in_burst:
                burst_start = bins[i]
                in_burst = True
        else:
            if in_burst:
                burst_end = bins[i]
                if (burst_end - burst_start) >= min_duration:
                    network_bursts.append((burst_start, burst_end))
                in_burst = False

    if in_burst:
        burst_end = bins[-1]
        if (burst_end - burst_start) >= min_duration:
            network_bursts.append((burst_start, burst_end))

    # Überlappende Network-Bursts zusammenfassen
    merged_network_bursts = []
    for burst in network_bursts:
        if not merged_network_bursts:
            merged_network_bursts.append(burst)
        else:
            last_start, last_end = merged_network_bursts[-1]
            curr_start, curr_end = burst
            # Falls sich Zeitintervalle überlappen oder direkt anschließen, zusammenfassen
            if curr_start <= last_end:
                merged_network_bursts[-1] = (last_start, max(last_end, curr_end))
            else:
                merged_network_bursts.append((curr_start, curr_end))

    return merged_network_bursts

def load_spike_data(input_file):
    """
    Lädt Spike-Daten aus einer bestehenden .bxr-Datei.
    """
    with h5py.File(input_file, 'r') as f:
        spike_times = f['Well_A1/SpikeTimes'][:]
        spike_channels = f['Well_A1/SpikeChIdxs'][:]
        sampling_rate = f.attrs['SamplingRate']
    return spike_times, spike_channels, sampling_rate

def save_burst_data(output_file,
                    spike_times,
                    spike_channels,
                    SpikeBurstTimes,
                    SpikeBurstChIdxs,
                    SpikeNetworkBurstTimes,
                    sampling_rate):
    """
    Speichert die Burst-Ergebnisse in einer neuen .bxr-Datei.
    """
    with h5py.File(output_file, 'w') as f:
        f.attrs['Version'] = 301
        f.attrs['SamplingRate'] = sampling_rate
        f.attrs['MinAnalogValue'] = -32768
        f.attrs['MaxAnalogValue'] = 32767

        spike_grp = f.create_group('Well_A1')
        spike_grp.create_dataset('SpikeTimes', data=spike_times, dtype=np.int64)
        spike_grp.create_dataset('SpikeChIdxs', data=spike_channels, dtype=np.int32)

        spike_grp.create_dataset('SpikeBurstTimes',
                                 data=np.array(SpikeBurstTimes, dtype=np.int64))
        spike_grp.create_dataset('SpikeBurstChIdxs',
                                 data=np.array(SpikeBurstChIdxs, dtype=np.int32))
        spike_grp.create_dataset('SpikeNetworkBurstTimes',
                                 data=np.array(SpikeNetworkBurstTimes, dtype=np.int64))

        print(f"BXR-Datei '{output_file}' erfolgreich gespeichert.")

def process_bxr_file(input_file):
    """
    Verarbeitet eine bestehende .bxr-Datei, erkennt Bursts und NetworkBursts
    und speichert die Ergebnisse in einer neuen Datei.
    """
    start_time = time.time()
    
    try:
        spike_times, spike_channels, sampling_rate = load_spike_data(input_file)
        print(f"Verarbeite Datei: {input_file}")
        print(f"Geladene Spike-Daten: {len(spike_times)} Spikes über {len(np.unique(spike_channels))} Kanäle.")
        
        # Convert spike times from frames to seconds
        spike_times_sec = spike_times / sampling_rate
        
        spikes_per_channel = {}
        for s_time, ch_idx in zip(spike_times_sec, spike_channels):
            spikes_per_channel.setdefault(ch_idx, []).append(s_time)
        
        bursts_per_channel = {}
        for ch_idx, times in spikes_per_channel.items():
            bursts_per_channel[ch_idx] = detect_bursts(times)
        
        SpikeBurstTimes = []
        SpikeBurstChIdxs = []
        for ch_idx, bursts in bursts_per_channel.items():
            for (burst_start, burst_end) in bursts:
                # Convert back to frames for storage
                SpikeBurstTimes.append((int(burst_start * sampling_rate), int(burst_end * sampling_rate)))
                SpikeBurstChIdxs.append(ch_idx)
        
        # Process network bursts with seconds
        network_bursts_sec = detect_network_bursts(
            spikes_per_channel,
            bin_size=0.025,
            threshold=12,
            min_duration=0.1
        )
        
        # Convert network bursts back to frames for storage
        SpikeNetworkBurstTimes = [(int(start * sampling_rate), int(end * sampling_rate)) 
                                for start, end in network_bursts_sec]
        
        print(f"Erkannte NetworkBursts: {len(SpikeNetworkBurstTimes)}")

        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_NB{ext}"
        
        save_burst_data(
            output_file,
            spike_times,
            spike_channels,
            SpikeBurstTimes,
            SpikeBurstChIdxs,
            SpikeNetworkBurstTimes,
            sampling_rate
        )
        
        end_time = time.time()
        print(f"Verarbeitung abgeschlossen. Datei gespeichert unter: {output_file}")
        print(f"Dauer: {end_time - start_time:.2f} Sekunden\n")
        
    except Exception as e:
        print(f"Fehler bei der Verarbeitung der Datei '{input_file}': {e}\n")

def find_bxr_files(folder_path):
    """
    Findet alle .bxr-Dateien rekursiv im angegebenen Ordner.
    """
    bxr_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.bxr'):
                bxr_files.append(os.path.join(root, file))
    return bxr_files

def process_all_bxr_files(folder_path):
    """
    Findet und verarbeitet alle .bxr-Dateien im angegebenen Ordner rekursiv.
    """
    start_time = time.time()
    bxr_files = find_bxr_files(folder_path)
    print(f"Gefundene .bxr-Dateien: {len(bxr_files)}\n")
    
    if not bxr_files:
        print("Keine .bxr-Dateien im angegebenen Ordner gefunden.")
        return
    
    for idx, bxr_file in enumerate(bxr_files, 1):
        print(f"Verarbeite Datei {idx} von {len(bxr_files)}:")
        process_bxr_file(bxr_file)
    
    end_time = time.time()
    print(f"Fertig. Gesamtzeit: {end_time - start_time:.2f} Sekunden.")

def main():
    folder_path = "data"
    
    if not os.path.isdir(folder_path):
        print(f"Der Ordner '{folder_path}' existiert nicht oder ist kein Verzeichnis.")
        return
    
    process_all_bxr_files(folder_path)

if __name__ == "__main__":
    main()
