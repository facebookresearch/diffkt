# DiffKt - A Differentiable Programming Framework for Kotlin

## What is DiffKt?

DiffKt is a general-purpose, functional, differentiable programming framework for Kotlin. It can automatically differentiate through functions of tensors, scalars, and user-defined types. It supports forward-mode and reverse-mode differentiation including Jacobian-vector and vector-Jacobian products, which can be composed for higher-order differentiation.

DiffKt also includes an early integration of ShapeTyping, an extensible compiler plugin for ahead-of-time tensor shape verification and inspection. With the ShapeTyping IntelliJ IDE plugin, users can even inspect tensor shapes and see tensor shape errors while in active development. 


## Getting Started

### Dependency Installation

Install the following dependencies:

* TODO (Probably Eigen, DNNL, etc.)

### Gradle/JVM

Add the following dependency to your `build.gradle.kts` file:

```
dependencies {
    implementation("TODO")
    ...
}
```

#### ShapeTyping

To use ShapeTyping, apply the plugin to your `build.gradle.kts` file

```
plugins {
   id("TODO")
   ...
}
```

And add the following dependencies

```
dependencies {
    implementation("TODO")
    implementation("TODO")
    ...
}
```

TODO: Instructions on downloading the IntelliJ

For more detailed instructions, please visit the ShapeTyping repository here. (TODO: link repository)

### Building From Source

You may also build DiffKt from source.

TODO: MacOS and Linux instructions

#### Kotlin Examples

Navigate to the `kotlin` folder from the repository root.

To run an example from the `examples` folder, use the command `./gradlew :examples:run -Ppackage=<package-name>`. For example, `./gradlew :examples:run -Ppackage=vector2`.

#### Kotlin Tests

Navigate to the `kotlin` folder from the repository root.

All tests should be run with the command `./gradlew test`

To run specific tests, use the command `./gradlew :<subproject-name>:test --tests "<test-name>"`. For example, `./gradlew :api:test --tests "ReluTest"`.


## Tutorials

Here are some tutorials to help you get started.
TODO: Link several tutorials here.

## Contributing

We welcome and greatly value all kinds of contributions to DiffKt. If you would like to contribute, please see our Contributing Guidelines. // TODO Link to CONTRIBUTING.md or some other documentation on website.

## License

DiffKt is [MIT licensed](./LICENSE).


