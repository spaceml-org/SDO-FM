# mount dataset
sudo apt -y install nfs-common
sudo mkdir /mnt/sdoml
sudo mount 10.14.32.66:/sdoml_hdd /mnt/sdoml -o ro,hard,timeo=600,retrans=3,rsize=262144,wsize=1048576,resvport,async,nconnect=7,_netdev

# install base dependencies
sudo apt -y update
sudo apt -y install libomp5
sudo systemctl restart multipathd.service packagekit.service snapd.service
# pip3 install mkl mkl-include
# pip3 install tf-nightly tb-nightly tbp-nightly
sudo apt -y install libopenblas-dev
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 numpy -f https://storage.googleapis.com/libtpu-releases/index.html

# setup tpu and test
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/
# export PJRT_DEVICE=TPU_C_API
# export PT_XLA_DEBUG=0
# export USE_TORCH=ON
# unset LD_PRELOAD
# export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
export PJRT_DEVICE=TPU_C_API
export PT_XLA_DEBUG=0
export USE_TORCH=ON
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_USE_BF16=1
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
python3 -c "import torch; import torch_xla; import torch_xla.core.xla_model as xm; print(xm.xla_device()); dev = xm.xla_device(); t1 = torch.randn(3,3,device=dev); t2 = torch.randn(3,3,device=dev); print(t1 + t2)"

# set python path
echo 'export PATH="$PATH:/usr/local/bin/python:/home/walsh/.local/bin"' >> ~/.bashrc
export PATH="$PATH:/usr/local/bin/python:/home/walsh/.local/bin"

# install SDO-FM dependencies
pip install -r ~/SDO-FM/requirements.txt
pip install -e ~/SDO-FM