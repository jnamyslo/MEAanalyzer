#!/bin/sh

#python src/append_bursts.py #Use this for faster computing. It will use a faster but more error prune network burst detection algorithm.
python src/append_bursts_computeIntense.py
python src/rasterplots.py
#python src/featurecalc.py
python src/featurecalc_tspe.py
python src/boxplots.py
python src/statistic.py
python src/connectivitygraphs.py
