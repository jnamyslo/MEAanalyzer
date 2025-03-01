#NOT WORKING YET!

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
    
    # Spike-Zeiten sortieren
    spike_times = np.sort(spike_times)
    # ISIs berechnen (Abstände zwischen aufeinanderfolgenden Spikes)
    isis = np.diff(spike_times)
    median_isi = np.median(isis) if len(isis) > 0 else 0
    
    # Schwellenwert: faktor * median_isi
    isi_threshold = isi_threshold_factor * median_isi
    
    burst_start_indices = []
    burst_end_indices = []
    
    in_burst = False
    burst_start = 0

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if not in_burst:
                # Neuer Burst beginnt
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                # Burst endet
                in_burst = False
                burst_end = i + 1  # weil i sich auf isis bezieht
                if (burst_end - burst_start) >= min_spikes_in_burst:
                    burst_start_indices.append(burst_start)
                    burst_end_indices.append(burst_end)
    
    # Falls letzter Burst bis zum Ende reicht
    if in_burst:
        burst_end = len(spike_times)
        if (burst_end - burst_start) >= min_spikes_in_burst:
            burst_start_indices.append(burst_start)
            burst_end_indices.append(burst_end)

    # Indizes in tatsächliche Zeitangaben umwandeln
    burst_times = [
        (spike_times[start], spike_times[end - 1])
        for start, end in zip(burst_start_indices, burst_end_indices)
    ]
    return burst_times

def detect_network_bursts(spikes_per_channel, bin_size=0.025, threshold=9, min_duration=0.1):
    """
    Erkennt Network-Bursts basierend auf der Synchronisation von Aktivität über mehrere Kanäle.

    Parameters:
    - spikes_per_channel: Dictionary mit Kanalindizes als Keys und Listen von Spike-Zeiten als Values.
    - bin_size: Größe der Zeitfenster in Sekunden (Standard: 0.025 s = 25 ms).
    - threshold: Schwellenwert für das Produkt aus aktiven Kanälen und Gesamtspikes.
    - min_duration: Mindestdauer eines Network-Bursts in Sekunden.

    Returns:
    - network_bursts: Liste von Tupeln (Startzeit, Endzeit) der Network-Bursts.
    """
    if not spikes_per_channel:
        return []

    # Alle Spiketimes abrufen und prüfen, ob Daten vorhanden sind
    all_spikes = np.concatenate(list(spikes_per_channel.values())) if spikes_per_channel else np.array([])
    if all_spikes.size == 0:
        return []

    min_time, max_time = np.min(all_spikes), np.max(all_spikes)
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    bin_count = len(bins) - 1  # Anzahl der Bins

    # Optimierte Speicherung der Werte
    active_channels_per_bin = np.zeros(bin_count, dtype=int)
    total_spikes_per_bin = np.zeros(bin_count, dtype=int)

    # Binweise Berechnung der aktiven Kanäle und Gesamtspikes
    for spike_times in spikes_per_channel.values():
        if len(spike_times) == 0:
            continue  # Überspringe leere Kanäle
        hist, _ = np.histogram(spike_times, bins=bins)
        active_channels_per_bin += (hist > 0).astype(int)  # Kanal zählt als aktiv, wenn mind. 1 Spike vorhanden ist
        total_spikes_per_bin += hist  # Gesamtzahl der Spikes pro Bin

    # Berechnung des Schwellenwertprodukts
    product = active_channels_per_bin * total_spikes_per_bin
    network_burst_bins = product >= threshold  # Boolean-Array: True für Bins, die als Network-Burst gelten

    # Identifikation zusammenhängender Bursts
    network_bursts = []
    in_burst = False
    burst_start = None

    for i, is_burst in enumerate(network_burst_bins):
        if is_burst:
            if not in_burst:
                burst_start = bins[i]
                in_burst = True
        else:
            if in_burst:
                burst_end = bins[i]  # Burst endet hier
                if (burst_end - burst_start) >= min_duration:
                    network_bursts.append((burst_start, burst_end))
                in_burst = False

    # Falls der letzte Burst bis zum Ende reicht
    if in_burst:
        burst_end = bins[-1]
        if (burst_end - burst_start) >= min_duration:
            network_bursts.append((burst_start, burst_end))

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

def save_burst_data(output_file,
                    spike_times,
                    spike_channels,
                    SpikeBurstTimes,
                    SpikeBurstChIdxs,
                    SpikeNetworkBurstTimes,
                    sampling_rate):
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
        for s_time, ch_idx in zip(spike_times, spike_channels):
            spikes_per_channel.setdefault(ch_idx, []).append(s_time)
        
        # Bursts pro Kanal erkennen
        bursts_per_channel = {}
        for ch_idx, times in spikes_per_channel.items():
            bursts_per_channel[ch_idx] = detect_bursts(times)
        
        # Für die Ausgabe sammeln wir die einzelnen Bursts
        SpikeBurstTimes = []
        SpikeBurstChIdxs = []
        for ch_idx, bursts in bursts_per_channel.items():
            for (burst_start, burst_end) in bursts:
                SpikeBurstTimes.append((burst_start, burst_end))
                SpikeBurstChIdxs.append(ch_idx)
        
        # Network-Bursts nach Chiappalone:
        SpikeNetworkBurstTimes = detect_network_bursts(
            spikes_per_channel,  # Dictionary mit Spike-Zeiten pro Kanal
            bin_size=0.5,      # 25 ms
            threshold=9,         # Schwellenwert aus dem Paper
            min_duration=0.1     # Mindestdauer 100 ms
        )
        
        print(f"Erkannte NetworkBursts: {len(SpikeNetworkBurstTimes)}")

        # Ausgabedateiname festlegen
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_BT_NBT{ext}"
        
        # Ergebnisse speichern
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
    
    Parameters:
    - folder_path: Pfad zum Hauptordner.
    
    Returns:
    - Liste aller gefundenen .bxr-Dateipfade.
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
    print(f"Fertig. Gesamtzeit: {end_time - start_time:.2f} Sekunden.")

def main():
    folder_path = "data"
    
    if not os.path.isdir(folder_path):
        print(f"Der Ordner '{folder_path}' existiert nicht oder ist kein Verzeichnis.")
        return
    
    process_all_bxr_files(folder_path)

if __name__ == "__main__":
    main()
