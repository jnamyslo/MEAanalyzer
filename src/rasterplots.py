import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Fixed total number of electrodes at 4096
TOTAL_CHANNELS = 4096

def raster_plots_main(filepath, 
                      save_as='svg', 
                      vector_output=False, 
                      dpi=600, 
                      output_dir='Rasterplots'):
    filename_base = os.path.splitext(os.path.basename(filepath))[0]
    rasterplot_path = os.path.join(output_dir, f'{filename_base}_raster_plot.png')
    separate_rasterplot_path = os.path.join(output_dir, f'{filename_base}_separate_raster_plot.png')
    
    # Check if raster plots already exist
    if os.path.exists(rasterplot_path) and os.path.exists(separate_rasterplot_path):
        print(f"Skipping {filepath}, raster plots already exist.")
        return
    
    with h5py.File(filepath, 'r') as file:
        well_id = 'Well_A1'
        
        if well_id + '/SpikeTimes' in file:
            spike_times = np.array(file[well_id + '/SpikeTimes'])
            spike_channels = np.array(file[well_id + '/SpikeChIdxs'])

            if 'SamplingRate' in file.attrs:
                sampling_rate = file.attrs['SamplingRate']
            else:
                raise ValueError("Sampling rate not found in file attributes.")

            spike_times_sec = spike_times / sampling_rate

            if well_id + '/SpikeBurstTimes' in file:
                burst_times = np.array(file[well_id + '/SpikeBurstTimes'])
                burst_channels = np.array(file[well_id + '/SpikeBurstChIdxs'])
            else:
                burst_times = np.array([])
                burst_channels = np.array([])

            if well_id + '/SpikeNetworkBurstTimes' in file:
                network_burst_times = np.array(file[well_id + '/SpikeNetworkBurstTimes'])
                network_burst_times = network_burst_times / sampling_rate
            else:
                network_burst_times = np.array([])

            spike_raster_data = [[] for _ in range(TOTAL_CHANNELS)]
            for time, channel in zip(spike_times_sec, spike_channels):
                spike_raster_data[channel].append(time)

            burst_raster_data = [[] for _ in range(TOTAL_CHANNELS)]
            if burst_times.size > 0:
                for (start, end), channel in zip(burst_times, burst_channels):
                    burst_raster_data[channel].append(start / sampling_rate)

            create_raster_plot(
                spike_raster_data,
                burst_raster_data,
                network_burst_times,
                filepath,
                output_dir,
                vector_output,
                dpi
            )

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

    if network_burst_times.size > 0:
        for (start, end) in network_burst_times:
            plt.axvspan(start, end, ymin=0.95, ymax=1.0, color='red', alpha=0.3)

    filename = os.path.basename(filepath)
    plt.title(f'Raster Plot: {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Channel Index')
    
    # Fixed axis limits for 4096 channels
    plt.ylim(-0.5, TOTAL_CHANNELS - 0.5)

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
    filename = os.path.basename(filepath)
    output_basename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_separate_raster_plot')

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].eventplot(spike_raster_data, colors='black', linelengths=0.9)
    axes[0].set_ylabel('Channel Index')
    axes[0].set_title('Spikes')
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(-0.5, TOTAL_CHANNELS - 0.5)

    if any(burst_raster_data):
        axes[1].eventplot(burst_raster_data, colors='lightgreen', linelengths=0.7)
    axes[1].set_ylabel('Channel Index')
    axes[1].set_title('Spike-Bursts')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(-0.5, TOTAL_CHANNELS - 0.5)

    axes[2].eventplot([], colors='red', linelengths=1.0)
    axes[2].set_ylabel('Channel Index')
    axes[2].set_title('Network Bursts')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(-0.5, TOTAL_CHANNELS - 0.5)

    if network_burst_times.size > 0:
        for (start, end) in network_burst_times:
            axes[2].axvspan(start, end, ymin=0, ymax=1, color='red', alpha=0.3)

    plt.tight_layout()

    plt.savefig(f'{output_basename}.png', format='png', dpi=dpi, bbox_inches='tight')
    if vector_output:
        plt.savefig(f'{output_basename}.svg', format='svg', bbox_inches='tight')
    plt.close()

def generate_raster_plots_for_all_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_NB.bxr'):
                filepath = os.path.join(dirpath, filename)
                output_dir = os.path.join(dirpath, 'Rasterplots')
                print(f"Plotting file: {filepath}")
                try:
                    raster_plots_main(filepath, output_dir=output_dir)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    root_dir = "data"
    generate_raster_plots_for_all_files(root_dir)
    print("All raster plots have been created.")
