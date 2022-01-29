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

3. Get the unregistered dependencies:
```shell
julia --project -e 'import Pkg;
                    Pkg.pkg"dev --local git@github.com:probcomp/MiniGSG.jl.git git@github.com:probcomp/GenDirectionalStats.jl.git git@github.com:probcomp/MeshCatViz.git git@github.com:probcomp/GLRenderer.jl.git";
                    Pkg.instantiate()'
```

if that doesn't work,
```shell
mkdir dev
cd dev
git clone git@github.com:probcomp/GenDirectionalStats.jl.git GenDirectionalStats
git clone git@github.com:probcomp/MeshCatViz.git MeshCatViz
git clone git@github.com:probcomp/GLRenderer.jl.git GLRenderer
git clone git@github.com:probcomp/MiniGSG.jl.git MiniGSG
```
then in the Julia REPL package manager (press `]` in REPL):
```
dev dev/GenDirectionalStats dev/MeshCatViz dev/GLRenderer dev/MiniGSG
```

4. Next, instantiate the dependencies by running `instantiate` in the Julia REPL package manager.

5. Get the YCB-Video data. Download [data](https://www.dropbox.com/s/dhbqmiu8i3mb3lx/ycbv-test.zip?dl=0) (or get [data_small](https://www.dropbox.com/s/ryyeh0jdkcmdpmu/0048.zip?dl=0) if you just want a small subset of the full dataset) and [models](https://www.dropbox.com/s/i4p7hci3kw375wd/models_txts_densefusion_results.zip?dl=0). Extract the contents and place them in the `data` such that the structure looks as follows:
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

