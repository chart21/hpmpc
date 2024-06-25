# PyGEON

PyGEON can train models in PyTorch and export them into .bin files compatible with [PIGEON](https://github.com/chart21/hpmpc/tree/NN).
To get started, specify the model and dataset you want to use in `main.py`. Afterward, the model will be trained and model and dataset will be saved in the associfated folders. PyGEON can also load existing models in `.pth` format using the provided `load_model` function.
