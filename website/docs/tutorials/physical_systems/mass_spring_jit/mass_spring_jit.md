---
id: 'mass_spring_jit'
title: 'Mass-Spring System with Just In Time Optimization'
slug: '/tutorials/physical_systems/mass_spring_jit'
---
This notebook demonstrates the just-in-time (jit) api. The jit api produces a 
optimized version of **DiffKt** code. This is useful for when you repeatedly call 
a function. On the first call to a jitted function, an optimized version is created. 
On subsequent calls, the optimized version is called, which should result in a speed up of the program.


:::tip Open tutorial in Github
[Mass-Spring System with Just In Time Optimization](https://github.com/facebookresearch/diffkt/blob/main/tutorials/mass_spring_jit.ipynb)
:::