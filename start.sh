#!/bin/sh

python src/append_bursts.py
#python src/rasterplots.py
python src/featurecalc.py
python src/connectivitygraphs.py