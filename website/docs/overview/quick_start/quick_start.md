---
id: 'quick_start'
title: 'Quick Start'
slug: '/overview/quick_start'
---
## On a Mac

### Maven

A precompiled **DiffKt** jar for a Mac is available in Maven. It has jni libraries included, so it only works on a Mac.

The current version is `0.0.1-DEV2`

[Maven Description](https://search.maven.org/artifact/com.facebook.diffkt/diffkt/0.0.1-DEV2/jar)

### Gradle/JVM

To use DiffKt, use the following dependency to your `build.gradle.kts` file with the `x.y.z` version number.

```
dependencies {
    implementation("com.facebook.diffkt:x.y.z")
}
```

## On Ubuntu

Currently, you need to checkout the repo and build it.

[Ubuntu Installation](installation_ubuntu)