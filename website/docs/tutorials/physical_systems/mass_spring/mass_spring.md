---
id: 'mass_spring'
title: 'Mass Spring System'
slug: '/tutorials/physical_systems/mass_spring'
---
The Mass Spring System notebook shows how to model coupled mass spring systems with **DiffKt**. Mass spring 
systems are used in computer graphics to model textures, fabrics, or clothes. The Mass Spring System 
notebook is a set of ordinary differential equations (ODEs) that model the forces of the spring interactions
as Hookian springs, and model the force of gravity pulling down on the masses. The ODEs are solved with a simple
forward Euler ODE solver. Energy equations are given for the different forces and **DiffKt** is used to 
automatically differentiate the energy equations in the solving of the ODEs.

:::tip Open tutorial in Github
[Mass Spring System](https://github.com/facebookresearch/diffkt/blob/main/tutorials/mass_spring.ipynb)