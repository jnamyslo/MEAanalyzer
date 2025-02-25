# MEAexplorer
containerbased MEA analysis

## Build the container
```bash
docker build -t meaanalyzer:latest .
```
## Run the container
```bash
docker run -rm -v /path/to/data/:/app/data meaanalyzer:latest
```
To avoid rebuilding the container every time, you can mount your local directory: add `-v .:/app` into the run command, when starting in the current working directory.

For debugging you have to set the ports according to your debugpy config: add `-p 5678:5678` into the run command.

Example:
```bash
docker run --rm -v .:/app -v /path/to/data/:/app/data meaexplorer:latest .
```

## Default folder structure
```
data/GROUPX/CHIPX
```
Also see `src/featurecalc.py`