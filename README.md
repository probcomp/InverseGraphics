# InverseGraphics

## Setup 

1. Clone the InverseGraphics Julia package:
```shell
git clone git@github.com:probcomp/InverseGraphics.git
cd InverseGraphics
```
2. Setup python3 virtual environment
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

3. Register the ProbComp registry in the Julia package manager:
```
registry add git@github.com:probcomp/InverseGraphicsRegistry.git
```

4. In the Julia REPL package manager run `activate .` and then `instantiate`

5. Get the YCB-Video data. Download 
- [data](https://www.dropbox.com/s/dhbqmiu8i3mb3lx/ycbv-test.zip?dl=0) (or get [data_small](https://www.dropbox.com/s/ryyeh0jdkcmdpmu/0048.zip?dl=0) if you just want a small subset of the full dataset)
- [models](https://www.dropbox.com/s/i4p7hci3kw375wd/models_txts_densefusion_results.zip?dl=0)
Extract the contents and place them in the `data` such that the structure looks as follows:
```
data/
  0048
  ...
  0059
  densefusion/
  models/
  model_list.txt
  keyframe.txt
```

6. Run `notebooks/demo.jl` in a jupyter/jupytext notebook. If all cells runs successfully, then the setup is functioning properly.

# Other Issues

If you encounter issues with PyCall referencing the wrong python instance
```shell
PYTHON=$(which python) PYCALL_JL_RUNTIME_PYTHON=$(which python) julia --project -e 'import Pkg; Pkg.build("Conda"); Pkg.build("PyCall")'
```

