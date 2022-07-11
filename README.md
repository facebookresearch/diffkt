# DiffKt - A Differentiable Programming Framework for Kotlin

## What is DiffKt?

DiffKt is a general-purpose, functional, differentiable programming framework for Kotlin. It can automatically differentiate through functions of tensors, scalars, and user-defined types. It supports forward-mode and reverse-mode differentiation including Jacobian-vector and vector-Jacobian products, which can be composed for higher-order differentiation. It also supports differentiating against user-defined types using interfaces. 

DiffKt also includes an early integration of ShapeTyping, an extensible compiler plugin for ahead-of-time tensor shape verification and inspection. With the ShapeTyping IntelliJ IDE plugin, users can even inspect tensor shapes and see tensor shape errors while in active development. 

## Getting Started


### Dependency Installation

Currently there are two implementations that are supported for DiffKt. Follow the links below to install dependencies. It is recommended to use [Homebrew](https://brew.sh) as a package manager for macOS. 

[Fast Ops](https://github.com/facebookresearch/diffkt/tree/main/cpp/ops)

[GPU Ops](https://github.com/facebookresearch/diffkt/blob/main/cpp/gpuops/README.md)


### Gradle/JVM

To use DiffKt, use the following dependency to your `build.gradle.kts` file with the `x.y.z` version number.

```
dependencies {
    implementation("com.facebook.diffkt:x.y.z")
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

## Building from the Source

Currently DiffKt building is supported on macOS as well as Ubuntu. Help is needed to support building on Windows. Build instructions can be found below: 

[Install on Mac](https://github.com/facebookresearch/diffkt/blob/main/INSTALL_MAC.md)

[Install on Ubuntu](https://github.com/facebookresearch/diffkt/blob/main/INSTALL_UBUNTU.md)

To build DiffKt from source files for UBUNTU, read INSTALL_UBUNTU.md

#### Kotlin Examples

Navigate to the `kotlin` folder from the repository root.

To run an example from the `examples` folder, use the command `./gradlew :examples:run -Ppackage=<package-name>`. For instance, `./gradlew :examples:run -Ppackage=vector2` will run the `vector2` example. 

#### Kotlin Tests

Navigate to the `kotlin` folder from the repository root.

All tests should be run with the command `./gradlew test`

To run specific tests, use the command `./gradlew :<subproject-name>:test --tests "<test-name>"`. For example, `./gradlew :api:test --tests "ReluTest"`.

## Tutorials

Here are some tutorials to help you get started.

### Overview

[Intro to Differentiable Programming](tutorials/intro_to_differentiable_programming.ipynb)

[Indexing](tutorials/indexing.ipynb)

[Broadcasting](tutorials/broadcasting.ipynb)


### Data Science and Machine Learning

[Simple Parabola Gradient Descent](tutorials/simple_parabola_gradient_descent.ipynb)

[Linear Regression w/ Gradient Descent](tutorials/linear_regression_gradient_descent.ipynb)

[3D Nonlinear Regression](tutorials/3d_nonlinear_regression.ipynb)

[Multivariable Linear Regression](tutorials/multivariable_linear_regression_gradient_descent.ipynb)

[Logistic Regression w/ Gradient Descent](tutorials/logistic_regression_gradient_descent.ipynb)

[Normal Distribution MLE](tutorials/normal_mle_gradient_descent.ipynb)

[Exponential Distribution MLE](tutorials/exponential_distribution_mle_gradient_descent.ipynb)

[Neural Network w/ Backpropagation](tutorials/neural_network_sgd.ipynb)

### Physics and Simulation

[Mass Spring](tutorials/mass_spring.ipynb)

### User-Defined Types

[User-Defined Types](tutorials/user_defined_types.ipynb)

[Linear Regression w/ User-Defined Types](tutorials/linear_regression_user_defined_types.ipynb)

[Neural Network w/ User-Defined Types](tutorials/neural_network_user_defined_types.ipynb)


## Contributing

We welcome and greatly value all kinds of contributions to DiffKt. If you would like to contribute, please see our Contributing Guidelines. Please refer to the [contributing document](CONTRIBUTING.md) or [DiffKt.org](https://diffkt.org) for more details.

## License

DiffKt is [MIT licensed](./LICENSE).


