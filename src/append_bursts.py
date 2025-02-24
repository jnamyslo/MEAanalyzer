# Calculating SpikeBursts and NetworkBursts based on existing Spikes of previous Spikedetection such as 3Brain BrainWave 5 or own Code
import os
import h5py
import numpy as np
import time

def detect_bursts(spike_times, isi_threshold_factor=0.8, min_spikes_in_burst=3):
    """
    Erkennung von Bursts nach Chiappalone.
    
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
    median_isi = np.median(isis)
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

    burst_times = [(spike_times[start], spike_times[end - 1]) for start, end in zip(burst_start_indices, burst_end_indices)]
    return burst_times

def detect_network_bursts(bursts_per_channel, min_active_bursts=3, min_duration=0.1):
    """
    Erkennung von Netzwerk-Bursts nach dem Ansatz von Chiappalone.
    
    Parameters:
    - bursts_per_channel: Dictionary mit Bursts pro Kanal.
    - min_active_bursts: Minimal erforderliche Anzahl an gleichzeitig aktiven Einzel-Bursts.
    - min_duration: Minimale Dauer eines Network-Bursts in Sekunden.
    
    Returns:
    - network_bursts: Liste von Tupeln (NetworkBurst-Start, NetworkBurst-Ende).
    """
    burst_events = []
    
    # Sammle alle Start- und End-Zeiten der Bursts als Events
    for channel_idx, bursts in bursts_per_channel.items():
        for burst_start, burst_end in bursts:
            burst_events.append((burst_start, +1))  # +1 für Burst-Start
            burst_events.append((burst_end, -1))    # -1 für Burst-Ende

    # Sortiere die Events nach Zeit (bei Gleichstand Start-Events vor End-Events)
    burst_events.sort(key=lambda x: (x[0], -x[1]))

    # Initialisierung der Netzwerk-Burst-Logik
    active_bursts = 0
    network_burst_start = None
    network_bursts = []

    # Verarbeitung der Events
    for time_event, delta in burst_events:
        active_bursts += delta

        if active_bursts >= min_active_bursts and network_burst_start is None:
            # Neuer Network-Burst beginnt
            network_burst_start = time_event
        elif active_bursts < min_active_bursts and network_burst_start is not None:
            # Network-Burst endet
            network_burst_end = time_event
            duration = network_burst_end - network_burst_start
            if duration >= min_duration:
                network_bursts.append((network_burst_start, network_burst_end))
            network_burst_start = None

    # Falls ein Network-Burst am Ende nicht geschlossen wurde
    if network_burst_start is not None:
        network_burst_end = burst_events[-1][0]
        duration = network_burst_end - network_burst_start
        if duration >= min_duration:
            network_bursts.append((network_burst_start, network_burst_end))

    return network_bursts

def load_spike_data(input_file):
    """
    Lädt Spike-Daten aus einer bestehenden .bxr-Datei.
    
    Parameters:
    - input_file: Pfad zur bestehenden .bxr-Datei.
    
    Returns:
    - spike_times: Liste der Spike-Zeitpunkte.
    - spike_channels: Liste der zugehörigen Kanalindizes.
    - sampling_rate: Abtastrate des Signals.
    """
    with h5py.File(input_file, 'r') as f:
        spike_times = f['Well_A1/SpikeTimes'][:]
        spike_channels = f['Well_A1/SpikeChIdxs'][:]
        sampling_rate = f.attrs['SamplingRate']
    return spike_times, spike_channels, sampling_rate

def save_burst_data(output_file, spike_times, spike_channels, SpikeBurstTimes, SpikeBurstChIdxs, SpikeNetworkBurstTimes, sampling_rate):
    """
    Speichert die Burst-Ergebnisse in einer neuen .bxr-Datei.
    
    Parameters:
    - output_file: Pfad zur Ausgabedatei.
    - spike_times: Liste der Spike-Zeitpunkte.
    - spike_channels: Liste der zugehörigen Kanalindizes.
    - SpikeBurstTimes: Liste der Burst-Zeitintervalle.
    - SpikeBurstChIdxs: Liste der zugehörigen Kanalindizes für Bursts.
    - SpikeNetworkBurstTimes: Liste der Network-Burst-Zeitintervalle.
    - sampling_rate: Abtastrate des Signals.
    """
    with h5py.File(output_file, 'w') as f:
        f.attrs['Version'] = 301
        f.attrs['SamplingRate'] = sampling_rate
        f.attrs['MinAnalogValue'] = -32768
        f.attrs['MaxAnalogValue'] = 32767

        # Original Spike-Daten kopieren
        spike_grp = f.create_group('Well_A1')
        spike_grp.create_dataset('SpikeTimes', data=spike_times, dtype=np.int64)
        spike_grp.create_dataset('SpikeChIdxs', data=spike_channels, dtype=np.int32)

        # Burst-Daten hinzufügen
        spike_grp.create_dataset('SpikeBurstTimes', data=np.array(SpikeBurstTimes, dtype=np.int64))
        spike_grp.create_dataset('SpikeBurstChIdxs', data=np.array(SpikeBurstChIdxs, dtype=np.int32))
        spike_grp.create_dataset('SpikeNetworkBurstTimes', data=np.array(SpikeNetworkBurstTimes, dtype=np.int64))

        print(f"BXR-Datei '{output_file}' erfolgreich gespeichert.")

def process_bxr_file(input_file):
    """
    Verarbeitet eine bestehende .bxr-Datei, erkennt Bursts und NetworkBursts und speichert die Ergebnisse.
    
    Parameters:
    - input_file: Pfad zur bestehenden .bxr-Datei.
    """
    start_time = time.time()
    
    try:
        # Lade die Spike-Daten
        spike_times, spike_channels, sampling_rate = load_spike_data(input_file)
        print(f"Verarbeite Datei: {input_file}")
        print(f"Geladene Spike-Daten: {len(spike_times)} Spikes über {len(np.unique(spike_channels))} Kanäle.")
        
        # Organisiere Spikes pro Kanal
        spikes_per_channel = {}
        for spike_time, channel_idx in zip(spike_times, spike_channels):
            if channel_idx not in spikes_per_channel:
                spikes_per_channel[channel_idx] = []
            spikes_per_channel[channel_idx].append(spike_time)
        
        # Erkennung von Bursts pro Kanal
        bursts_per_channel = {}
        total_bursts = 0
        for channel_idx, times in spikes_per_channel.items():
            bursts = detect_bursts(times)
            bursts_per_channel[channel_idx] = bursts
            total_bursts += len(bursts)
            print(f"Kanal {channel_idx}: Erkannt {len(bursts)} Bursts.")
        
        # Vorbereitung der Burst-Daten zum Speichern
        SpikeBurstTimes = []
        SpikeBurstChIdxs = []
        
        for channel_idx, bursts in bursts_per_channel.items():
            for burst_start, burst_end in bursts:
                SpikeBurstTimes.append((burst_start, burst_end))
                SpikeBurstChIdxs.append(channel_idx)
        
        # NetworkBurstdetection
        SpikeNetworkBurstTimes = detect_network_bursts(
            bursts_per_channel,
            min_active_bursts=3,  # Schwellenwert für aktive Einzel-Bursts
            min_duration=0.1       # Minimale Dauer in Sekunden
        )
        
        print(f"Gesamt erkannt {len(SpikeNetworkBurstTimes)} NetworkBursts.")
        
        # Erstellen des Ausgabepfads
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_BT_NBT{ext}"
        
        # Speichern der Daten in eine neue .bxr-Datei
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
        print(f"Verarbeitung abgeschlossen. Ergebnisse wurden gespeichert in: {output_file}")
        print(f"Ausführungszeit für diese Datei: {end_time - start_time:.2f} Sekunden\n")
        
    except Exception as e:
        print(f"Fehler bei der Verarbeitung der Datei '{input_file}': {e}\n")

def find_bxr_files(folder_path):
    """
    Findet alle .bxr-Dateien rekursiv im angegebenen Ordner.
    
    Parameters:
    - folder_path: Pfad zum Hauptordner.
    
    Returns:
    - List of .bxr file paths.
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
    
    Parameters:
    - folder_path: Pfad zum Hauptordner.
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
    print(f"Es wurde bei allen Dateien die Bursts und Networkbursts angehängt. Gesamtzeit: {end_time - start_time:.2f} Sekunden.")

def main():
    folder_path = "data"
    
    if not os.path.isdir(folder_path):
        print(f"Der Ordner '{folder_path}' existiert nicht oder ist kein Verzeichnis.")
        return
    
    process_all_bxr_files(folder_path)

if __name__ == "__main__":
    main()
