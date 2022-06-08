Installation of diffkt on Ubuntu 20.04

1. On Ubuntu install C++ tools
	sudo apt-get install build-essential libssl-dev
2. Install CMAKE
	sudo snap install cmake -classic
3. Install JDK 11. If not installed, download "Oracle Java 11 JDK"
4. Make sure that JAVA_HOME is set to the JDK 11 directory.
5. Make sure your alternatives for java and javac point to the right version 
6. Add $JAVA_HOME/bin to your PATH in .bashrc
	export PATH = ${PATH}:$JAVA_HOME/bin
7. Add the following environmental variables
	export JAVA_INCLUDE_PATH=$JAVA_HOME/include/
	export JAVA_INCLUDE_PATH2=$JAVA_HOME/include/linux/
	export JAVA_AWT_INCLUDE_PATH=$JAVA_HOME/include/
8. Update your environment
	source .bashrc
9. Install IntelliJ IDEA 
	a) install Python plugin
	b) install Kotlin plugin
	c) install Kotlin Notebook plugin
	d) configure Kotlin Notebook plugin	
10. Install CUDA 11.? from NVIDIA (should work with the most current version, if not 11.1)
	https://developer.nvidia.com/cuda-downloads
11. Install cuDNN from NVDIA
	https://developer.nvidia.com/rdp/cudnn-download
12. Install OpenBlas
	sudo apt-get install libopenblas-dev	
13. Create a github account if you do not have one.
14. Setup your github account to use ssh.
15. Setup your github account to use a token.
16. Install github tools.
	sudo apt-get install gh
17. Install the token into github tools.
18. Fork diffkt to you github account.
19. Clone the fork to your local computer.
	gh repo clone <github-id>/diffkt
20. Check your local diffkt project to see if the original facebookresearch/diffkt.git is upstream
	cd <your git projects>/diffkt
	git remote -v
	
	You should see
	
	origin	git@github.com:<gethub id>/diffkt.git (fetch)
	origin	git@github.com:<gethub id>/diffkt.git (push)
	upstream	git@github.com:facebookresearch/diffkt.git (fetch)
	upstream	git@github.com:facebookresearch/diffkt.git (push)
	
	If you are missing upstream, execute the following.	
	git remote add upstream git@github.com:facebookresearch/diffkt.git
21. Read the github docs for merging your local repository with upstream.
	https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
22. In general, pull from upstream and merge, but checkin to origin (your fork) and do a pull request to merge with upstream.
23. cd <you project>/diffkt/cpp/ops
24. Read the readme.md
25. Install OpenMP
	sudo apt install libomp-dev
	
	and/or
	
	sudo apt-get install libgomp1
26. Install Eigen
	sudo apt-get install libeigen3-dev
27. Install the Intel oneAPI Base Toolkit
28. Add the proper oneAPI libraries to link. diffkt uses version 2.1.0, which is different from the Base Toolkit.
    cd <your git projects>diffkt/cpp/ops
    wget $(fwdproxy-config wget) https://github.com/oneapi-src/oneDNN/releases/download/v2.1/dnnl_lnx_2.1.0_cpu_gomp.tgz
    tar -zxvf dnnl_lnx_2.1.0_cpu_gomp.tgz

    mv dnnl_lnx_2.1.0_cpu_gomp dnnl	
29. mkdir build
30. cd build
31. cmake ..
32. make VERBOSE=1
33. If make completed without error, then the ops directory should be built and installed as the following:  
	<your projects>/diffkt/kotlin/api/src/main/resources/libdnnlops_jni.so
	<your projects>/diffkt/kotlin/api/src/main/resources/libops_jni.so
	<your projects>/diffkt/kotlin/api/src/main/resources/libsparseops_jni.so

34. cd <your projects>/diffkt/cpp/gpuops
35. read README.md
36. source /opt/intel/oneapi/setvars.sh
	Note: You may have link issues with the Intel libraries. For a quick work around, create a 
	symbolic link in /usr/lib to the library that is not linking.
37. Install libtorch in the gpuops directory. .gitignore is set to ignore the directory
	
	wget $(fwdproxy-config wget) https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
	unzip libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
	rm libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcu111.zip
38. mkdir build
39. cd build
40. cmake ..
41. make VERBOSE=1 -j
42. If make completed without error, then the gpuops directory should be built and installed as the following:

	<your projects>/diffkt/kotlin/api/src/main/resources/libgnuops_jni.so

43. Make sure your default shell is bash and that sh links to bash
44. cd <your projects>/diffkt/kotlin
45. read README.md
46. At GitHub make sure you token has "read:packages" scope.
47. Create the file github.env with the following in it:
	GITHUB_ACTOR=<your username>
	GITHUB_TOKEN=<your access token with the read:packages permission>
48. Start intellij
49. Open <you projects>/diffkt/kotlin
50. Build project
