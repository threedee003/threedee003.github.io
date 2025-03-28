---
layout: post
title: "Diffusion Policy almost Explained"
author:
- Tribikram Dhar
---


To get a good grasp of diffusion policy, we need to understand diffusion models first. Here it goes.


| ![image](/assets/ddpm.png) |
| :--: |
| *The forward Markov chain for adding noise to the data.* |

A denoising diffusion probabilistic model(DDPM) (1) uses two Markov chains, the forward diffusion process that converts data to noise and the reverse diffusion process that converts noise to data. The forward diffusion is usually handcrafted and the reverse diffusion is learned by a parameterized deep neural network.\
Let us consider a data distribution $x_0 \sim q(x_0)$, the forward Markov chain generates a sequence of random variables $x_1, x_2, ..., x_T$ using a transition kernel $q(x_t | x_{t-1})$. We can use the chain rule of probability and Markov property to write the joint distribution as $q(x_1, x_2, ..., x_T)$ conditioned on $x_0$.

The formulae for the transition kernel is given by:

$$ q(x_1, ..., x_T | x_0) = \prod_{t = 1}^T q(x_t | x_{t-1}) \tag{1}$$

In DDPM the kernel is hancrafted generally and we use a Gaussian distribution for the transition kernel.

$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)  \tag{2}$$

where $beta_t$ is a hyperparameter. $\beta_t \in (0, 1)$.

Now there is a long ass derivation to marginalise the joint distribution. For that use the following equation:

$$ x_t = \sqrt{1-\beta} x_{t-1} + \sqrt{\beta} \mathcal{N} (0, \mathcal{I}) \tag{3}$$

In this equation replace $x_{t-1}$ with $\sqrt{1-\beta} x_{t-2} + \sqrt{\beta} \mathcal{N} (0, \mathcal{I})$ and then $x_{t-2}$ with $\sqrt{1-\beta} x_{t-3} + \sqrt{\beta} \mathcal{N} (0, \mathcal{I})$ and so on till we reach $x_0$. This will take time to derive and we will get the following:

$$ x_t = \sqrt{\alpha_t.\alpha_{t-1}...\alpha_1} x_0 + \sqrt{1-\alpha_t.\alpha_{t-1}...\alpha_1} \mathcal{N} (0, \mathcal{I}) \tag{4}$$

where $\alpha_t = 1 - \beta_t$. So equation 4 can be written as:

$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \mathcal{N} (0, \mathcal{I}) \tag{5}$$

where $\bar{\alpha_t} = \prod_{i=1}^t \alpha_i$.

Hence we can write the joint distribution in the marginaslised form as:

$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathcal{I}) \tag{6}$$

Now given a a data point $x_0$ we can easily obtain $x_t$ by sampling a noise vector $\epsilon \sim \mathcal (0, \mathcal{I})$ from a Gaussian distribution and apply the follwing transformation :
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \tag{7}$$

The forward process is shown in the figure above where using the transition kernel an image is slowly converted to a pure Gaussian noise. Now we will discuss about the reverse Markov diffusion process where noise is gradually removed to generate a data point.\
\
The reverse Markov chain is parameterised by a prior distribution $p(x_T) = \mathcal{N} (x_T; 0, \mathcal{I})$ and a learnable transition kernel $p_\theta(x_{t-1} | x_t)$ where $\theta$ is the learnable parameter of the deep neural network. The learnable transition kernel can expressed by the following equation:\
\
$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) \tag{8}$$\
In equation 8, $\mu_\theta(x_t, t)$ is the mean and $\Sigma_\theta(x_t, t)$ is the covariance of parameterised by the deep neural network. With the reverse Markov chain we can generate a data sample $x_0$ by first sampling a noise vector $x_T \sim p(x_T)$ and then iteratively sampling from the learnable transition kernel $x_{t-1} \sim p_\theta(x_{t-1} | x_t)$ till we reach $t = 1$. The idea is to train the Markov chain to match the actual reversal of the forward Markov chain. The deep neural network's parameters $\theta$ are adjusted in such a wat that the joint distribution of the reverse Markov chain closely approximates the forward Markov chain. To match the forward Markov chain we need to minimise the KL divergence between the forward and reverse Markov chain. KL divergence is a measure of how different two probability distributions are. In our case the KL divergence between the forward and reverse Markov chain is given by:

$$ \mathcal{L} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0, \mathcal{I})}[KL(q(x_t | x_0) || p_\theta(x_{t-1} | x_t))] \tag{9}$$

The equation 9 can be further simplified to:

$$ \mathcal{L} = -\mathbb{E}_{q(x_0,..x_T)} [log {p_{\theta} (x_0.. x_T)}] + constant \tag{10}$$

$$ \mathcal{L} = \mathbb{E}_{q(x_0,..x_T)} [ -log p(x_T) - \sum_{t = 1}^T log \frac{p_{\theta} (x_{t-1} | x_t)}{q(x_t | x_{t-1})}]+ constant \tag{11}$$

The entire term before constant in equation 11 is the negative log likelihood of the reverse Markov chain which is called the variantional lower bound (VLB). Our aim is to minimise the negetive VLB i.e maximise the VLB. We formulate the loss function as mean squared error between the predicted and actual noise. An important the constant term is a lot of mathematical jargon which is not parameterised by theta and not trained by the optimizer so I ignored them. You can read it in the original paper (1).\
The loss function is given by:
$$ \mathcal{L}_{mse} = \mathbb{E}_{t \sim \mathcal{U} [1,T],x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, \mathcal{I})}[ || \epsilon - \epsilon_\theta(x_t, t) ||^2] \tag{12}$$

$x_t$ can be computed from $x_0$ and  $\epsilon$ using equation 5. The deep learning network takes the noisy vector $x_t$ and the timestep $t$ as input and predicts the noise $\epsilon$ that was added to $x_0$ to get $x_t$. The timestep $t$ is sampled from a uniform distrbution. The optimizer trains the deep neural network $p_\theta$ to minimise the loss using the gradient descent algorithm. 

During inference we can sample a pure Gaussian noise and then iteratively sample from the reverse Markov chain till we reach $t = 1$ to generate a data point.

Now lets get to Diffusion Policy, what we were here for.






