To install dependencies:

If `gcc --version` does not give a version above >= 5.0, install and use GCC >= 5.0 by following [these instructions](https://our.internmc.facebook.com/intern/wiki/PyTorch/PyTorchDev/Workflow/Open_source/#initial-setup-recent-com).
Note that when you log out and back in, you may need to run `source scl_source enable devtoolset-7` since the default gcc version resets itself.

Download and extract libtorch.
To download on a devserver, run:
```
# from <repo root>/cpp/gpuops
wget $(fwdproxy-config wget) https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
```
(More info on fwdproxy here: https://www.internalfb.com/intern/wiki/Development_Environment/Internet_Proxy/)

To build:
```
mkdir build && cd build
# Optionally include -DCMAKE_BUILD_TYPE=Debug
cmake ..
make -j
```
You may need to include `-DCMAKE_PREFIX_PATH=/absolute/path/to/unzipped/libtorch` if you didn't put it in cpp/gpuops.

The following instructions are from a while ago and may or may not be helpful:

If on a devserver, also install BLAS:
```
sudo yum install openblas-devel
```
