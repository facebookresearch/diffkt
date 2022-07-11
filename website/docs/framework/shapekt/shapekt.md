---
id: 'shapekt'
title: 'ShapeKt'
slug: '/framework/shapekt'
---
ShapeKt is an extensible Kotlin compiler plugin for ahead-of-time tensor (multi-dimensional) arrays shape verification and inspection. Commonly used in machine learning, tensors are often fed through many different operations; each operation often has different shape requirements and produces a new tensor with a possibly different shape. ShapeKt provides a system to describe and enforce shape requirements and output shapes.

With the ShapeKt IntelliJ IDE plugin, users can inspect tensor shapes and see tensor shape errors while in active development.

ShapeKt is currently experimental. There is an early integration with DiffKt, a differentiable programming framework in Kotlin.

:::tip Open ShapeKt in Github
[ShapeKt](https://github.com/facebookresearch/shapekt)
:::