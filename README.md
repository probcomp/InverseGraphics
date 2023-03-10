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

If you encounter issues with PyCall referencing the wrong python instance run: `sh scripts/fix_pycall.sh`

which runs the following command
```shell
PYTHON=$(which python) PYCALL_JL_RUNTIME_PYTHON=$(which python) julia --project -e 'import Pkg; Pkg.build("Conda"); Pkg.build("PyCall")'
```

Depending on how encryption keys are configured you might have to do below steps (e.g. on GCP's pain Ubuntu VM, you need to manually do the steps below):
- To enable Julia's package manager to be able to pull code from github repos, you need to activate the ssh-agent, and specifically add your github key:
```
eval "$(ssh-agent -s)"
ssh-add <your github key>

```
- To start an X server on your machine. Please note the monitor being output. It should look like :0 or :1. Another (maybe better way) to do it is to use `startx` but I haven't been able to get that step to work yet on a Google Cloud Project VM with attached display. 
```
vncserver -localhost
# example output:
# New 'X' desktop is machine:1
# Starting applications specified in /home/balgobin/.vnc/xstartup
# Log file is /home/balgobin/.vnc/machine:1.log
```

- To tell GLFW which display to use (based on output from previous step):
```
export DISPLAY=:1
```

- If you get the error below:
```
ERROR: failed to clone from git@github.com:probcomp/InverseGraphicsRegistry.git, error: GitError(Code:EEOF, Class:SSH, ERROR: You're using an RSA key with SHA-1, which is no longer allowed. Please use a newer client or a different key type.
Please see https://github.blog/2021-09-01-improving-git-protocol-security-github/ for more information.
```
This is because your ssh key is not supporting SHA-2. You need to run `ssh-keygen -t ed25519` and then add this new key using `ssh-add <your NEW github key>`

- Unclear how to solve this error on Ubuntu 20.04 GCP VM (it can be a bit involved https://forums.gentoo.org/viewtopic-t-1038878-start-0.html):

<img width="1012" alt="Screen Shot 2022-03-28 at 13 59 42" src="https://user-images.githubusercontent.com/1942909/160468427-c3ee09ba-68ad-4741-9643-b10659e4f53e.png">








