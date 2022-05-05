# msc_thesis_code
Conformal prediction with an underlying GraphSAGE applied to dynamic and static graphs.


## Install Pytorch Geometric on Oracle Cloud
```
ssh -i .ssh/id_ed25519 ubuntu@130.61.160.82

// sudo apt install gcc

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev

sudo apt install python3-pip

// install gpu drivers
https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#installation_scripts

// verify installation
sudo nvidia-smi

// conda install pytorch cudatoolkit=10.1 -c pytorch
pip3 install torch==1.8.0

python3 -c "import torch; print(torch.version.cuda)"

pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip3 install torch-geometric

pip3 install ogb

// might need to run `pip3 install torch==1.8.0` again, to override ogb's torch version

pip3 install matplotlib
sudo apt-get install -y xvfb

pip3 install notebook

add github public key

git clone

python3 -m notebook

ssh -L 8888:localhost:8888  -i .ssh/id_ed25519 ubuntu@130.61.160.82

go to localhost:8888

copy-paste token from jupyter notebook command
```
