# DiffKt Tensor Library

## Dev setup
One time:
1) Follow setup instructions [here](../cpp/ops/README.md) to build cpp dependencies
2) Create the following file at `<repo root>/kotlin/github.env`.
   The access token can be created [here](https://github.com/settings/tokens). It should have the `read:packages` scope. Make sure you have access to the [optimizer-plugins](https://github.com/facebookresearch/optimizer-plugins) repo.
    ```
    GITHUB_ACTOR=<your username>
    GITHUB_TOKEN=<your access token with the read:packages permission>
    ```
   