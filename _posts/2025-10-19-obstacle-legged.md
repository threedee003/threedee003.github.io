---
layout: post
title: "Hierarchical Safe Locomotion for Legged Robots Using Learnable Signed Distance Function and Nonlinear MPC"
author:
- Tribikram Dhar, Surekha Borra, Nilanjan Dey
---

***Abstract*** : We present a hierarchical motion planning framework for safe and efficient locomotion of legged robots. Our system integrates a learnable signed distance function (SDF) for environment perception, a nonlinear model predictive control (NMPC) module for trajectory optimization, and a neural network policy trained using Proximal Policy Optimization (PPO) to generate stable walking gaits. We embed the learnable SDF within the NMPC cost formulation, enabling the optimization process to explicitly reason about obstacle proximity and generate collision-free locomotion trajectories. To improve computational efficiency, we introduce an extended optimal action execution strategy that selectively reuses optimal action sequences, significantly reducing the computational burden of NMPC while preserving trajectory smoothness and control performance. Through simulation studies, we demonstrate that our framework enables reliable, obstacle-aware locomotion and achieves improved real-time performance compared to standard NMPC-based baselines

***Note*** : Code will be released upon publication.

![demo1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/3f7a5a29-a217-41f1-8aff-0389443963e8)
