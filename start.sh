#!/bin/sh

python src/append_bursts.py 
#python src/append_bursts_legacy.py #Use this for faster computing. It will use a faster but more error prune network burst detection algorithm.
python src/rasterplots.py
python src/featurecalc.py
python src/boxplots.py
python src/statistic.py
python src/connectivitygraphs.py
