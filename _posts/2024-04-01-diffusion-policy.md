---
layout: post
title: "Diffusion Policy Explained"
author:
- Tribikram Dhar
---


To get a good grasp of diffusion policy, we need to understand diffusion models first. Here it goes.


| ![image](/assets/ddpm.png) |
| :--: |
| *The forward Markov chain for adding noise to the data.* |

A denoising diffusion probabilistic model(DDPM) (1) uses two Markov chains, the forward diffusion process that converts data to noise and the reverse diffusion process that converts noise to data. The forward diffusion is usually handcrafted and the reverse diffusion is learned by a parameterized deep neural network.
Let us consider a data distribution $x_0 \sim q(x_0)$, the forward Markov chain generates a sequence of random variables $x_1, x_2, ..., x_T$ using a transition kernel $q(x_{t-1}|x_t)$. We can use the chain rule of probability and Markov property to write the joint distribution as $q(x_1, x_2, ..., x_T)$ conditioned on $x_0$.

The formulae for the transition kernel is given by:

$$ q(x_1, ..., x_T | x_0) = \prod_{t = 1}^T q(x_t | x_{t-1}) $$

In DDPM the kernel is hancrafted generally and we use a Gaussian distribution for the transition kernel.

$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I) $$

where $beta_t$ is a hyperparameter. $\beta_t \in (0, 1)$.

Now there is a long ass derivation to marginalise the joint distribution 



## References

(1) Ho et al. [Denoising diffusion probabilistic models , 2020](https://scholar.google.com/scholar_lookup?arxiv_id=2006.11239#:~:text=Denoising%20diffusion%20probabilistic%20models)\
(2) Chi et al. [Diffusion Policy : Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137v4)





