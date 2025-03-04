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

def detect_network_bursts_from_spikes(spikes_per_channel, active_channels_factor=0.1, 
                                     min_duration=0.1, time_window=0.05):
    """
    Erkennung von Netzwerk-Bursts direkt aus Spike-Daten.
    Ein Network-Burst beginnt, sobald mindestens ein bestimmter Anteil (factor)
    der Kanäle innerhalb eines Zeitfensters aktiv sind.
    
    Parameters:
    - spikes_per_channel: Dictionary {Kanal: [Spike-Zeitpunkte in Sekunden]}
    - active_channels_factor: Faktor (0.0-1.0) der Kanäle, die gleichzeitig aktiv sein müssen
    - min_duration: Minimale Dauer eines Network-Bursts in Sekunden
    - time_window: Zeitfenster in Sekunden, in dem Spikes als gleichzeitig betrachtet werden
    
    Returns:
    - network_bursts: Liste von Tupeln (NetworkBurst-Start, NetworkBurst-Ende)
    """
    # Erzeuge ein Event-Array mit (Zeit, +1/-1, Kanal)
    # wobei +1 für "Aktivität beginnt" und -1 für "Aktivität endet" steht
    spike_events = []
    
    total_channels = len(spikes_per_channel)
    min_active_channels = max(1, int(total_channels * active_channels_factor))
    
    print(f"Total channels with spikes: {total_channels}")
    print(f"Min active channels for network burst: {min_active_channels} (factor: {active_channels_factor})")
    
    # Für jeden Kanal und jeden Spike erzeugen wir ein "Aktivitäts"-Fenster
    for channel_idx, spike_times in spikes_per_channel.items():
        for spike_time in spike_times:
            # Jeder Spike aktiviert den Kanal für die Dauer des Zeitfensters
            spike_events.append((spike_time, +1, channel_idx))
            spike_events.append((spike_time + time_window, -1, channel_idx))
    
    # Sortiere Events zeitlich 
    # (Aktivierungen vor Deaktivierungen bei gleicher Zeit)
    spike_events.sort(key=lambda x: (x[0], -x[1]))
    
    active_channels = set()
    network_burst_start = None
    network_bursts = []
    
    for time_event, delta, ch_idx in spike_events:
        if delta == +1:
            active_channels.add(ch_idx)
        else:
            active_channels.discard(ch_idx)
        
        # Prüfen, ob ein Network-Burst beginnt
        if len(active_channels) >= min_active_channels and network_burst_start is None:
            network_burst_start = time_event
        
        # Prüfen, ob ein laufender Network-Burst endet
        elif len(active_channels) < min_active_channels and network_burst_start is not None:
            network_burst_end = time_event
            
            # Mindestdauer prüfen
            if (network_burst_end - network_burst_start) >= min_duration:
                network_bursts.append((network_burst_start, network_burst_end))
            
            network_burst_start = None
    
    # Falls am Ende noch ein Network-Burst "offen" ist
    if network_burst_start is not None:
        network_burst_end = spike_events[-1][0]
        if (network_burst_end - network_burst_start) >= min_duration:
            network_bursts.append((network_burst_start, network_burst_end))
    
    # Zusammenführen von überlappenden oder nahe beieinander liegenden Network-Bursts
    if network_bursts:
        merged_bursts = [network_bursts[0]]
        for current_start, current_end in network_bursts[1:]:
            prev_start, prev_end = merged_bursts[-1]
            
            # Wenn der aktuelle Burst innerhalb eines kurzen Zeitfensters nach dem vorherigen beginnt,
            # werden sie zusammengeführt
            if current_start - prev_end < time_window:
                merged_bursts[-1] = (prev_start, current_end)
            else:
                merged_bursts.append((current_start, current_end))
    network_bursts = merged_bursts
    return network_bursts

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
            bursts_per_channel[ch_idx] = detect_bursts(
                times,
                isi_threshold_factor=0.6,
                min_spikes_in_burst=3)
        
        SpikeBurstTimes = []
        SpikeBurstChIdxs = []
        for ch_idx, bursts in bursts_per_channel.items():
            for (burst_start, burst_end) in bursts:
                # Convert back to frames for storage
                SpikeBurstTimes.append((int(burst_start * sampling_rate), int(burst_end * sampling_rate)))
                SpikeBurstChIdxs.append(ch_idx)
        
        # Process network bursts with seconds
        network_bursts_sec = detect_network_bursts_from_spikes(
            spikes_per_channel,
            active_channels_factor=0.01,
            min_duration=0.1,
            time_window=0.001  # 50ms window to consider spikes as co-active
        )
        
        # Convert network bursts back to frames for storage
        SpikeNetworkBurstTimes = [(int(start * sampling_rate), int(end * sampling_rate)) 
                                for start, end in network_bursts_sec]
        
        print(f"Erkannte Bursts: {len(SpikeBurstTimes)}")
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
