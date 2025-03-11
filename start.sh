#!/bin/sh


./folder_structure.sh

python src/append_bursts.py 
python src/rasterplots.py
python src/featurecalc.py
python src/boxplots.py
python src/statistic.py
python src/connectivitygraphs.py


# If Spike Detection was performed using 3Brain BrainWave 5 and burst detection / network burst detection was performed, use the code below:

# python src/3Brain-Detected/rasterplots_3Brain.py
# python src/3Brain-Detected/featurecalc_3Brain.py
# python src/boxplots.py
# python src/statistic.py
# python src/3Brain-Detected/connectivitygraphs_3Brain.py

