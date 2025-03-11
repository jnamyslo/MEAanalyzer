import os
import h5py
import numpy as np
import time

def detect_bursts(spike_times, isi_threshold_factor=0.25, min_spikes_in_burst=3):
    """
    Burst detection according to Baker et al.
    Parameters:
    - spike_times: List of spike timestamps.
    - isi_threshold_factor: Factor for determining the ISI threshold.
    - min_spikes_in_burst: Minimum number of spikes in a burst.
    
    Returns:
    - burst_times: List of tuples (burst start, burst end).
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
    Detects network bursts based on the synchronization of activity across multiple channels.
    Ensures that overlapping network bursts are merged into one event.

    Network burst detection using the Chiappalone approach:
    
    Parameters:
    - spikes_per_channel: Dictionary {channel: [SpikeTimes, ...]} with spike times per channel
    - bin_size: Time window size in seconds for analyzing synchronous activity
    - threshold: Threshold for the product of active channels and spike count
    - min_duration: Minimum duration of a network burst in seconds
    
    Returns:
    - network_bursts: List of tuples (NetworkBurst start, NetworkBurst end)
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

    # Iterate through bins and identify connected True regions as network bursts
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

    # Merge overlapping network bursts
    merged_network_bursts = []
    for burst in network_bursts:
        if not merged_network_bursts:
            merged_network_bursts.append(burst)
        else:
            last_start, last_end = merged_network_bursts[-1]
            curr_start, curr_end = burst
            # If time intervals overlap or connect directly, merge them
            if curr_start <= last_end:
                merged_network_bursts[-1] = (last_start, max(last_end, curr_end))
            else:
                merged_network_bursts.append((curr_start, curr_end))

    return merged_network_bursts
    #return network_bursts

def load_spike_data(input_file):
    """
    Loads spike data from an existing .bxr file.
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
    Saves burst results in a new .bxr file.
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

        print(f"BXR file '{output_file}' successfully saved.")

def process_bxr_file(input_file):
    """
    Processes an existing .bxr file, detects bursts and network bursts,
    and saves the results in a new file.
    """
    start_time = time.time()
    
    try:
        spike_times, spike_channels, sampling_rate = load_spike_data(input_file)
        print(f"Processing file: {input_file}")
        print(f"Loaded spike data: {len(spike_times)} spikes across {len(np.unique(spike_channels))} channels.")
        
        # Convert spike times from frames to seconds
        spike_times_sec = spike_times / sampling_rate
        
        spikes_per_channel = {}
        for s_time, ch_idx in zip(spike_times_sec, spike_channels):
            spikes_per_channel.setdefault(ch_idx, []).append(s_time)
        
        bursts_per_channel = {}
        for ch_idx, times in spikes_per_channel.items():
            bursts_per_channel[ch_idx] = detect_bursts(
                times,
                isi_threshold_factor=0.7,
                min_spikes_in_burst=3)
        
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
            threshold=90,
            min_duration=0.1
        )
        
        # Convert network bursts back to frames for storage
        SpikeNetworkBurstTimes = [(int(start * sampling_rate), int(end * sampling_rate)) 
                                for start, end in network_bursts_sec]
        
        print(f"Detected bursts: {len(SpikeBurstTimes)}")
        print(f"Detected network bursts: {len(SpikeNetworkBurstTimes)}")

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
        print(f"Processing complete. File saved as: {output_file}")
        print(f"Duration: {end_time - start_time:.2f} seconds\n")
        
    except Exception as e:
        print(f"Error processing file '{input_file}': {e}\n")

def find_bxr_files(folder_path):
    """
    Recursively finds all .bxr files in the specified folder.
    """
    bxr_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.bxr'):
                bxr_files.append(os.path.join(root, file))
    return bxr_files

def process_all_bxr_files(folder_path):
    """
    Finds and processes all .bxr files in the specified folder recursively.
    """
    start_time = time.time()
    bxr_files = find_bxr_files(folder_path)
    print(f"Found .bxr files: {len(bxr_files)}\n")
    
    if not bxr_files:
        print("No .bxr files found in the specified folder.")
        return
    
    for idx, bxr_file in enumerate(bxr_files, 1):
        print(f"Processing file {idx} of {len(bxr_files)}:")
        process_bxr_file(bxr_file)
    
    end_time = time.time()
    print(f"Done. Total time: {end_time - start_time:.2f} seconds.")

def main():
    folder_path = "data"
    
    if not os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' does not exist or is not a directory.")
        return
    
    process_all_bxr_files(folder_path)

if __name__ == "__main__":
    main()
