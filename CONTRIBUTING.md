# Contributing to DiffKt
We welcome and greatly value all kinds of contributions to DiffKt. You can contribute in several ways:

* Opening issues
* Contributing to the library code base
* Contributing examples
* Adding documentation

For contributions to ShapeTyping, please visit the ShapeTyping repository here. (TODO: link repository)

TODO: Decide on contribution guidelines


## Operating System Support 

Currently DiffKt build is supported on macOS as well as Ubuntu. Help is needed to support building on Windows. 

[Install on Mac](https://github.com/facebookresearch/diffkt/blob/main/INSTALL_MAC.md)

[Install on Ubuntu](https://github.com/facebookresearch/diffkt/blob/main/INSTALL_UBUNTU.md)

## Issues

If you encounter any issues, feel free to report it using GitHub issues. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## Pull Requests

TODO: Decide on PR best practices. 

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Deployment

To deploy DiffKt to the Central Repository: 

1. Create a Sontatype JIRA account at `issues.sontatype.org` 
2. File an issue to get rights to push to `com.facebook`
3. Install [GnuPG](https://gnupg.org) and generate GPG keys 

     1. On MacOS it is recommended to use Homebrew: `brew install gnupg` 
     2. Generate a key pair, provide a passphrase when prompted: `gpg --gen-key` 
     3. List the key pairs and note the last 8 characters of the 40-character key ID string: `gpg --list-keys`
     4. Move into the  _.gnugpg_ directory: `cd ./.gnugpg`
     5. Export the secret key: `gpg --keyring secring.gpg --export-secret-keys > ~/.gnupg/`
     6. Send the public key to the Ubuntu server, replace the `XXXXXXXXX` placeholder with the key ID or last 8 characters of the key ID: `gpg --send-keys --keyserver keyserver.ubuntu.com XXXXXXXXX`     

4. In your personal _.gradle_ directory outside the project (often in directory like _/Users/thomasnield/.gradle/_) create a _gradle.properties_ file and add the following contents to it. Be sure to provide the following information and change the _signing.secretKeyRingFile_ file to your _securing.gpg_ file path. 

```
repositoryUsername=[your Sonatype JIRA username]
repositoryPassword=[your Sonatype JIRA password]
signing.keyId=[last 8 characters of the key ID]
signing.password=[your passphrase for the key]
signing.secretKeyRingFile=/Users/thomasnield/.gnupg/secring.gpg
```

5. In the _diffkt/kotlin/api/build.gradle.kts_ script, change the release version. 

```
pom { 
...
                version = "X.X.X"
...
}
```

6. In the _diffkt/kotlin/api_ directory run the publish command: ` ./gradlew clean publish`
7. Upon success, log into [Staged Repositories](https://oss.sonatype.org/#stagingRepositories) with your Sonatype JIRA account, select the artifact you just uploaded, close it, wait a few minutes for it to process, refresh, and then click "Release."
8. Within an hour the artifact release should be visible in [the repository](https://repo1.maven.org/maven2/com/facebook/diffkt/). 

### Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of Meta’s open source projects.

Complete your CLA here: https://code.facebook.com/cla

## License

By contributing to DiffKt, you agree that your contributions will be licensed as described in the LICENSE file in the root directory of this source tree.
