---
layout: post
title: "Diffusion Policy Explained"
author:
- Tribikram Dhar
---










Diffusion Models [[1]] are a class of generative models that have been shown to generate high-quality images, videos, and other types of data. They are based on the idea of using a Markov chain to gradually transform a random noise vector into a sample from the desired distribution. 

Diffusion models have two stages : the forward process and the reverse process. In the forward process, the model gradually adds noise to the input data, while in the reverse process, the model gradually removes the noise to generate the final output. The model is trained to minimize the difference between the forward and reverse processes, which is typically done using maximum likelihood estimation. 

### The forward diffusion process

Let us consider a diffusion process that transforms a data point $x_0$ into a noisy data point $x_t$ at time step $t$. In the forward diffusion process at each time step $t$ we add a small amount of Gaussian noise to the sample producing a sequence of noisy samples $x_0, x_1, ..., x_T$. The step sizes are controlled by a variance schedule $\beta_1, \beta_2, ..., \beta_T$.


This blog would contain the explaination for the paper "Diffusion Policy : Visuomotor Policy Learning via Action Diffusion" \[2\]
which used diffusion process to generate actions for robot end effector to perform dexterous manipulation.





## References

\[1\] Ho et al. [Denoising diffusion probabilistic models , 2020](https://scholar.google.com/scholar_lookup?arxiv_id=2006.11239#:~:text=Denoising%20diffusion%20probabilistic%20models)
\[2\] Chi et al. [Diffusion Policy : Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137v4)






