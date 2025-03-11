import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Main function for creating raster plots
def raster_plots_main(filepath, 
                      save_as='svg', 
                      vector_output=False, 
                      dpi=600, 
                      output_dir='Rasterplots'):
    with h5py.File(filepath, 'r') as file:
        well_id = 'Well_A1'
        
        # Check if SpikeTimes exist
        if well_id + '/SpikeTimes' in file:
            spike_times = np.array(file[well_id + '/SpikeTimes'])
            spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

            # Read sampling rate
            if 'SamplingRate' in file.attrs:
                sampling_rate = file.attrs['SamplingRate']
            else:
                raise ValueError("Sampling rate not found in file attributes.")

            spike_times_sec = spike_times / sampling_rate

            # Burst information (optional)
            if well_id + '/SpikeBurstTimes' in file:
                burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
                burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            else:
                burst_times = np.array([])
                burst_channels = np.array([])

            # Network-Burst information (optional)
            if well_id + '/SpikeNetworkBurstTimes' in file:
                network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
                network_burst_times = network_burst_times / sampling_rate
            else:
                network_burst_times = np.array([])

            # Map channels
            unique_channels = np.unique(spike_channels)
            channel_map = {ch: idx for idx, ch in enumerate(unique_channels)}

            # Spikes per channel
            spike_raster_data = [[] for _ in range(len(unique_channels))]
            for time, channel in zip(spike_times_sec, spike_channels):
                spike_raster_data[channel_map[channel]].append(time)

            # Burst starts per channel
            burst_raster_data = [[] for _ in range(len(unique_channels))]
            if burst_times.size > 0:
                for (start, end), channel in zip(burst_times, burst_channels):
                    burst_raster_data[channel_map[channel]].append(start / sampling_rate)

            # Create combined plot
            create_raster_plot(
                spike_raster_data,
                burst_raster_data,
                network_burst_times,
                filepath,
                output_dir,
                vector_output,
                dpi
            )

            # Create separate plot
            create_separate_raster_plot(
                spike_raster_data,
                burst_raster_data,
                network_burst_times,
                filepath,
                output_dir,
                vector_output,
                dpi
            )
        else:
            print(f"Warning: 'SpikeTimes' not found in {filepath}.")

# Combined raster plot
def create_raster_plot(spike_raster_data,
                       burst_raster_data,
                       network_burst_times,
                       filepath,
                       output_dir,
                       vector_output,
                       dpi):
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_raster_data, colors='black', linelengths=0.9)
    if any(burst_raster_data):
        plt.eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)

    # Network-Bursts as color bars (start to end) across the entire height
    if network_burst_times.size > 0:
        y_min, y_max = plt.ylim()
        for (start, end) in network_burst_times:
            plt.axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.3)

    filename = os.path.basename(filepath)
    plt.title(f'Raster Plot: {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Channel Index')
    plt.tight_layout()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Spikes'),
        Line2D([0], [0], color='lightgreen', lw=2, label='Spike-Bursts'),
        Line2D([0], [0], color='red', lw=2, label='Network Bursts')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_basename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_raster_plot')
    plt.savefig(f'{output_basename}.png', format='png', dpi=dpi, bbox_inches='tight')
    if vector_output:
        plt.savefig(f'{output_basename}.svg', format='svg', bbox_inches='tight')
    plt.close()

# Separate raster plot
def create_separate_raster_plot(spike_raster_data,
                                burst_raster_data,
                                network_burst_times,
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
    axes[0].set_ylabel('Channel Index')
    axes[0].set_title('Spikes')
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(-0.5, num_channels - 0.5)

    # Spike-Bursts
    if any(burst_raster_data):
        axes[1].eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)
    axes[1].set_ylabel('Channel Index')
    axes[1].set_title('Spike-Bursts')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(-0.5, num_channels - 0.5)

    # Network bursts as color bars
    axes[2].eventplot([], colors='red', linelengths=1.0) # Dummy plot for legend
    axes[2].set_ylabel('Channel Index')
    axes[2].set_title('Network Bursts')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(-0.5, num_channels - 0.5)
    if network_burst_times.size > 0:
        y_min, y_max = axes[2].get_ylim()
        for (start, end) in network_burst_times:
            axes[2].axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.3)

    plt.tight_layout()

    plt.savefig(f'{output_basename}.png', format='png', dpi=dpi, bbox_inches='tight')
    if vector_output:
        plt.savefig(f'{output_basename}.svg', format='svg', bbox_inches='tight')
    plt.close()

# Function for recursively searching and creating raster plots
def generate_raster_plots_for_all_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_NB.bxr'):
                filepath = os.path.join(dirpath, filename)
                print(f"Creating raster plot for file: {filepath}")
                output_dir = os.path.join(dirpath, 'Rasterplots')
                try:
                    raster_plots_main(filepath, output_dir=output_dir)
                    print(f"Raster plot saved in: {output_dir}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    root_dir = "data"
    generate_raster_plots_for_all_files(root_dir)
    print("All raster plots have been created.")
