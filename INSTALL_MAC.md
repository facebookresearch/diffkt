### Installation of **DiffKt** on a Mac

1. Create a Github account, if you do not have one.
2. Setup your github account to use ssh. 
3. Setup your github account to use a token.
4. Install github tools,

    `brew install gh`

5. Install the token into github tools.
6. At GitHub make sure your token has "read:packages" scope.
7Fork facebookresearch/diffkt to you github account. 
8. Clone the fork to your local computer.

    `gh repo clone {github-id}/diffkt`

9. Check your local **DiffKt** project to see if the original facebookresearch/diffkt.git is upstream,

    `cd {your git projects}/diffkt`

    `git remote -v`

    You should see

    `origin	git@github.com:{gethub id}/diffkt.git (fetch)`

    `origin	git@github.com:{gethub id}/diffkt.git (push)`

    `upstream	git@github.com:facebookresearch/diffkt.git (fetch)`

    `upstream	git@github.com:facebookresearch/diffkt.git (push)`

    If you are missing upstream, execute the following.

    `git remote add upstream git@github.com:facebookresearch/diffkt.git`

10. Read the github docs for merging your local repository with upstream,

    `https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork`

11. Create the file `diffkt\kotlin\github.env` with the following information in it,

    `GITHUB_ACTOR={your username}`

    `GITHUB_TOKEN={your access token with the read:packages permission}`

12. TBD, Instructions on C++ build enviroment.
13. In general, pull from upstream and merge, but checkin to origin (your fork) and do a pull request to merge with upstream,
14. Install Oracle JDK 11
15. Set `JAVA_HOME` to point to Oracle JDK 11
16. Add `$JAVA_HOME/bin` to your `PATH` in your shell initialization file,

    `export PATH = ${PATH}:$JAVA_HOME/bin`

17. Add the following environmental variables in your shell initialization file:

   `export JAVA_INCLUDE_PATH=$JAVA_HOME/include/`

   `export JAVA_INCLUDE_PATH2=$JAVA_HOME/include/linux/`

   `export JAVA_AWT_INCLUDE_PATH=$JAVA_HOME/include/`

18. Reinitialize your shell.
19. Install Brew if it is not installed,

    `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

20. Install the following packages using Brew,

    `brew install cmake`

    `brew install onednn`

    `brew install libomp`

    `brew install eigen`

21. Build the cpp/ops directory,

    `pushd cpp/ops`

    `mkdir -p build && cd build`

    `cmake -DCMAKE_PREFIX_PATH=$DNNL_PATH ..`

    `make -j && CTEST_OUTPUT_ON_FAILURE=1 make test`

    `popd`

22. Build the cpp/gpuops directory,

    TBD

23. Build the Kotlin system

    `pushd kotlin`

    `./gradlew clean`

    `./gradlew build`

    `popd`