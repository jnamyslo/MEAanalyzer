# MEAanalyzer

![Connectivity Graph Example](sample_images/ConnGraph_ID2024-01_20240702_1.0M_L_after30min.png)

A containerized pipeline for analyzing Multi-Electrode Array (MEA) recordings with automated burst detection, feature calculation, visualization, and statistical analysis.

## Overview

MEAanalyzer is a comprehensive tool for neuronal activity analysis from MEA recordings. It performs:

- Burst detection (single-electrode and network-level)
- Feature calculation (spike rates, ISI, synchrony measures)
- Visualization (raster plots, boxplots, connectivity graphs)
- Statistical analysis between experimental groups

### Spike Detection Options

Before running MEAanalyzer, you need spike-detected MEA recordings. You have two options:

1. **Open-source method**: Use our companion spike detection repository [MEAexplorer](https://github.com/tivenide/MEAexplorer) to process raw MEA recordings (.brw) and generate .bxr files with detected spikes. This Repo was also developed in the BioMEMS Lab of TH-AB.

2. **Commercial software**: Alternatively, use BrainWave 5 from 3Brain, which offers comprehensive spike detection capabilities. If your files already contain spike detection through BrainWave 5, you can skip the `append_bursts_computeIntense.py` or `append_bursts.py` modules in the pipeline, as burst detection will be handled directly from the pre-processed files.

Choose the option that best fits your workflow and data processing needs.

## Getting Started

**IMPORTANT NOTE** 

Make sure to name your recording files properly. Your .bxr files need to have the following structure: 

ID<number><4 digit number>>-<<chip-ID>Chip-ID>_whateveryouwant.bxr

Which means for example: ID2024-01_20250306_Reference.bxr

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- MEA recording data in .bxr format
- Make sure to have your Chip ID in the file names! E.g. *ID2024-0X* or *ID2024-01_yourFilenamewhatever.bxr*

### Clone the Repository

```bash
git clone https://github.com/jnamyslo/MEAanalyzer.git
cd MEAanalyzer
```

### Important adaptations!
1. Make sure to edit your UID in the Dockerfile therefor:
```bash
echo $UID
```
The number which you get e.g. 1009 need to be placed in the Dockerfile under:


```bash
RUN adduser -u <YOURUIDHERE> --disabled-password --gecos "" appuser && chown -R appuser /app`
```

2. Make sure to adapt the folder_structure.sh script according to your experiment. Only change your experimental groups and chips:

```bash
# <EDIT START HERE>

# Define your experimental groups and their associated IDs
# Modify these arrays according to your actual grouping

group_ids["SHAM"]="02 04 06 08"
group_ids["BDNF"]="01 03 05 07"
# group_ids["GROUP"]="XX XX XX"

# < EDIT END HERE >
```

### Build the Docker Container

```bash
docker build -t meaanalyzer:latest .
```

To rebuild the container after changes:

```bash
docker rmi -f meaanalyzer:latest && docker build -t meaanalyzer:latest .
```

### Input Data Structure

The pipeline requires a specific folder structure for proper analysis. **This folderstructure is created automatically!** Nevertheless here for reference:

```
data/
├── labels.txt (optional)
├── GROUP1/              # Experimental group (e.g., "SHAM", "BDNF")
│   ├── ID2024-01/       # Individual sample identifier
│   │   ├── file1.bxr    # MEA recording files
│   │   └── file2.bxr
│   └── ID2024-03/
│       ├── file1.bxr
│       └── file2.bxr
└── GROUP2/
    ├── ID2024-02/
    │   ├── file1.bxr
    │   └── file2.bxr
    └── ID2024-04/
        ├── file1.bxr
        └── file2.bxr
```

**Important Notes:**
- Group folders represent experimental conditions
- ID folders represent individual samples/chips
- Files are processed in alphabetical order and treated as sequential time points
- You can optionally include a `labels.txt` file in the data directory with semicolon-separated labels for timepoints. Make sure to have an equal amount of files in each Chip-Folder. Make also sure to provide the same amount of labels in your `labels.txt`, otherwise it will not be used.

The Folder structure is created automatically. You just need to make sure you mount the data path with all your `.bxr`files in the root folder. (E.g. data)

### Custom Labels

To use custom labels for time points, add a file named `labels.txt` in your data directory:

```
Baseline;Treatment 1h;Treatment 24h;Recovery
```

### Run the Analysis

Mount your data directory and run the container:

```bash
docker run -d --rm -v /path/to/your/data:/app/data meaanalyzer:latest
```

To follow the logs:

```bash
docker logs -f <CONTAINER-ID>
```

## Customizing the Pipeline

You can customize which modules run by editing the `start.sh` file:

```bash
#!/bin/sh

python src/append_bursts_computeIntense.py  # Burst detection
python src/rasterplots.py                   # Generate raster plots
python src/featurecalc_tspe.py              # Calculate features
python src/boxplots.py                      # Generate box plots
python src/statistic.py                     # Statistical analysis
python src/connectivitygraphs.py            # Generate connectivity graphs
```

### Module Dependencies

- **append_bursts_computeIntense.py**: Entry point that processes raw .bxr files and adds burst detection
  - Alternative: append_bursts.py (faster but less precise burst detection)
- **rasterplots.py**: Creates visualizations of spike activity (depends on burst detection)
  ![Raster Plot Example](sample_images/ID2024-01_1_20240702_control_separate_raster_plot.png)
- **featurecalc_tspe.py**: Calculates neural activity features (depends on burst detection)
  - Alternative: featurecalc.py (without TSPE connectivity metrics) 
- **boxplots.py**: Generates boxplots for comparing groups (depends on feature calculation)
  ![Box Plot Example](sample_images/Connectivity_Number_of_Connections_rel_stat_boxplot.png)
- **statistic.py**: Performs statistical analysis between groups (depends on feature calculation)
- **connectivitygraphs.py**: Generates network visualization (depends on feature calculation)
  ![Connectivity Graph Example](sample_images/ConnGraph_ID2024-01_20240917_77DAT_20241025_09.png)

## Output

The analysis generates multiple outputs in your data directory:

- Processed .bxr files with burst detection
- Raster plots of neural activity
- Feature calculation files (.npz)
- Statistical comparison results
- Box plots of features across time points
- Connectivity network visualizations

## Contact

Jannick Namyslo (s180537@th-ab.de)
