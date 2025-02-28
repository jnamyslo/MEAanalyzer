import os
import h5py
import numpy as np
import time

def perform_spike_sorting(spike_times, spike_channels, sampling_rate, 
                          min_refractory_period, min_median_isi):
    # 1) Spikes filtern, die kürzer als die Refraktärzeit aufeinander folgen
    sorted_spike_times = []
    sorted_spike_channels = []
    channel_indices = np.unique(spike_channels)
    refractory_samples = int(min_refractory_period * sampling_rate)

    for ch in channel_indices:
        ch_mask = spike_channels == ch
        ch_spike_times = np.sort(spike_times[ch_mask])
        filtered_times = []
        last_spike = None
        for t in ch_spike_times:
            if last_spike is None or (t - last_spike) >= refractory_samples:
                filtered_times.append(t)
                last_spike = t
        sorted_spike_times.extend(filtered_times)
        sorted_spike_channels.extend([ch] * len(filtered_times))

    sorted_spike_times = np.array(sorted_spike_times, dtype=np.int64)
    sorted_spike_channels = np.array(sorted_spike_channels, dtype=np.int32)

    # 2) Kanäle entfernen, deren Median-ISI unter einem physiologisch sinnvollen Wert liegt
    final_spike_times = []
    final_spike_channels = []
    channel_indices = np.unique(sorted_spike_channels)

    isi_limit_samples = int(min_median_isi * sampling_rate)

    for ch in channel_indices:
        ch_mask = (sorted_spike_channels == ch)
        ch_spike_times = np.sort(sorted_spike_times[ch_mask])
        if len(ch_spike_times) < 2:
            continue
        isis = np.diff(ch_spike_times)
        median_isi = np.median(isis)
        if median_isi >= isi_limit_samples:
            final_spike_times.extend(ch_spike_times)
            final_spike_channels.extend([ch] * len(ch_spike_times))

    return np.array(final_spike_times, dtype=np.int64), np.array(final_spike_channels, dtype=np.int32)

def detect_bursts(spike_times, isi_threshold_factor=0.8, min_spikes_in_burst=3):
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

def detect_network_bursts(bursts_per_channel, min_active_channels=3, min_duration=0.1):
    burst_events = []
    for channel_idx, bursts in bursts_per_channel.items():
        for (burst_start, burst_end) in bursts:
            burst_events.append((burst_start, +1, channel_idx))
            burst_events.append((burst_end, -1, channel_idx))
    burst_events.sort(key=lambda x: (x[0], -x[1]))
    active_channels = set()
    network_burst_start = None
    network_bursts = []
    for time_event, delta, ch_idx in burst_events:
        if delta == +1:
            active_channels.add(ch_idx)
        else:
            active_channels.discard(ch_idx)
        if len(active_channels) >= min_active_channels and network_burst_start is None:
            network_burst_start = time_event
        elif len(active_channels) < min_active_channels and network_burst_start is not None:
            network_burst_end = time_event
            if (network_burst_end - network_burst_start) >= min_duration:
                network_bursts.append((network_burst_start, network_burst_end))
            network_burst_start = None
    if network_burst_start is not None:
        network_burst_end = burst_events[-1][0]
        if (network_burst_end - network_burst_start) >= min_duration:
            network_bursts.append((network_burst_start, network_burst_end))
    return network_bursts

def load_spike_data(input_file):
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
    with h5py.File(output_file, 'w') as f:
        f.attrs['Version'] = 301
        f.attrs['SamplingRate'] = sampling_rate
        f.attrs['MinAnalogValue'] = -32768
        f.attrs['MaxAnalogValue'] = 32767
        spike_grp = f.create_group('Well_A1')
        spike_grp.create_dataset('SpikeTimes', data=spike_times, dtype=np.int64)
        spike_grp.create_dataset('SpikeChIdxs', data=spike_channels, dtype=np.int32)
        spike_grp.create_dataset('SpikeBurstTimes', data=np.array(SpikeBurstTimes, dtype=np.int64))
        spike_grp.create_dataset('SpikeBurstChIdxs', data=np.array(SpikeBurstChIdxs, dtype=np.int32))
        spike_grp.create_dataset('SpikeNetworkBurstTimes', data=np.array(SpikeNetworkBurstTimes, dtype=np.int64))
        print(f"BXR-Datei '{output_file}' erfolgreich gespeichert.")

def process_bxr_file(input_file):
    start_time = time.time()
    try:
        spike_times, spike_channels, sampling_rate = load_spike_data(input_file)
        print(f"Verarbeite Datei: {input_file}")
        print(f"Geladene Spike-Daten: {len(spike_times)} Spikes über {len(np.unique(spike_channels))} Kanäle.")

        spike_times, spike_channels = perform_spike_sorting(
            spike_times, spike_channels, sampling_rate,
            min_refractory_period=0.005,  # z.B. 1 ms
            min_median_isi=2.0         # z.B. 100 ms
        )

        spikes_per_channel = {}
        for s_time, ch_idx in zip(spike_times, spike_channels):
            spikes_per_channel.setdefault(ch_idx, []).append(s_time)

        bursts_per_channel = {}
        for ch_idx, times in spikes_per_channel.items():
            bursts_per_channel[ch_idx] = detect_bursts(times)

        SpikeBurstTimes = []
        SpikeBurstChIdxs = []
        for ch_idx, bursts in bursts_per_channel.items():
            for (burst_start, burst_end) in bursts:
                SpikeBurstTimes.append((burst_start, burst_end))
                SpikeBurstChIdxs.append(ch_idx)

        SpikeNetworkBurstTimes = detect_network_bursts(
            bursts_per_channel,
            min_active_channels=3,
            min_duration=0.1
        )
        print(f"Erkannte NetworkBursts: {len(SpikeNetworkBurstTimes)}")

        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_BT_NBT{ext}"

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
    bxr_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.bxr'):
                bxr_files.append(os.path.join(root, file))
    return bxr_files

def process_all_bxr_files(folder_path):
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
