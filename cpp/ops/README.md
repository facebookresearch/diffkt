# Fast Ops for the DiffKt Tensor library

## Dev setup

### Local prereqs

Install onednn (formerly mkl-dnn): `brew install onednn`

Install openMP: `brew install libomp`

Install eigen (Optional): `brew install eigen`

Install MKL (Optional) on mac:
1. download Intel oneAPI Base Toolkit : https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html#base-kit
2. install Intel oneAPI Base Toolkit : select `Custom Installation` ->
   un-select all but `Intel oneAPI Math Kernel Library`. Then consent, install, finish.
3. download Intel oneAPI HPC Toolkit : https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html#base-kit
4. install Intel oneAPI HPC Toolkit : select `Custom Installation` ->
   un-select `Intel Fortan Compiler Classic`. Then continue, skip.
5. run `source /opt/intel/oneapi/setvars.sh` (note that `setvars.sh` only sets
   up the environment variables for the terminal it runs in. So we need to
   run this every time before using MKL in a new terminal.)

### Devserver prereqs

Install DNNL:
```
# from <repo root>/cpp/ops
wget $(fwdproxy-config wget) https://github.com/oneapi-src/oneDNN/releases/download/v2.1/dnnl_lnx_2.1.0_cpu_gomp.tgz
tar -zxvf dnnl_lnx_2.1.0_cpu_gomp.tgz
mv dnnl_lnx_2.1.0_cpu_gomp dnnl
```
Or you can grab a more recent release from [here](https://github.com/oneapi-src/oneDNN/releases).

### Setup

To build:

```
# One-time or if you update the cmake version or local prereq version
rm -rf build
mkdir build && cd build

cmake ..

# Every time
make
```
This should produce libsparseops_jni and libops_jni in {repo_root}/kotlin/api/src/main/resources.

To auto-format code:
```
# From <repo root>/cpp/ops:
./scripts/format.sh
```

### Upgrading DNNL

Update onednn: `brew upgrade onednn`

Download a specific version of onednn (formerly mkl-dnn):
1. `brew search mkl-dnn` will show PR links for older brew formulas to download mkl-dnn. If no PRs show up, you'll have to manually navigate to the correct github commit.
2. Navigate to desired PR in browser and navigate to the raw `mkl-dnn.rb` file.
3. Download the raw `mkl-dnn.rb` file. `echo $(curl -fsSL <url to mkl-dnn.rb>) > mkl-dnn.rb`. For example to download version 1.4: `echo -e "$(curl -fsSL https://raw.githubusercontent.com/chenrui333/homebrew-core/93fd9e0e95cbfd065ef088bf5500129d606a1b38/Formula/mkl-dnn.rb)" > mkl-dnn.rb`.
4. `brew install --build-from-source mkl-dnn.rb`

For newer versions (1.6.2+), replace mkl-dnn with onednn in the instructions above.

### Select the sparse computation implementation

#### Three options are:
- [MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.6c6bff) : a high performance parallel library.
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) : a easy-to-use open-source, but not fully parallelized, BLAS library.
- OMP : parallel implementations with OpenMP. It doesn't rely on any third
  party library.

See section `Local prereqs` for installation instructions for `Eigen` and
`MKL`.

#### Select with `cmake`:
`cmake` variable `SPARSE_LIB` defines which implementation to use.
- `cmake -DSPARSE_LIB=MKL ..`, choose MKL
- `cmake -DSPARSE_LIB=EIGEN ..`, choose Eigen
- `cmake -DSPARSE_LIB=OMP ..`, choose OMP
- `cmake ..`, then `SPARSE_LIB=NONE`. That is the user doesn't give
  preference on which implementation to use. Then predefined preference
  will be applied. Currently it's: `MKL > Eigen > OMP`.
  If a preferred third party library is not installed, then check on the next
  preferred choice. For example, if `MKL` is not installed, then choose
  `Eigen`. If `Eigen` is also not installed, then choose `OMP`.
