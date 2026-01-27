# torch_brain tutorials for Dandisets

This repository contains a set of tutorials demonstrating how to train deep learning models using [torch_brain](https://github.com/neuro-galaxy/torch_brain) and [DANDI](https://dandiarchive.org/).

Tutorial notebooks:
- `000409_data_conversion.ipynb`: How to convert [Dandiset 000409: IBL - Brain Wide Map](https://dandiarchive.org/dandiset/000409/draft) to Brainset, the data format used by torch_brain
- `000409_poyo.ipynb`: How to train a POYO model on Dandiset 000409 files

Auxiliary Python modules:
- `pipeline.py`: Brainset conversion pipeline
- `aux_functions.py`: Auxiliary functions for model training and visualization
- `requirements.txt`: Required Python packages