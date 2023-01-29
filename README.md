# Efficient Gradient Flows in Sliced-Wasserstein Space

This repository contains the code and the experiments of the paper [Efficient Gradient Flows in Sliced-Wasserstein Space](https://arxiv.org/abs/2110.10972). In this paper, we propose to to minimize functionals in the space of probability measures endowed by the sliced-Wasserstein distance. This can be done by leveraging the so-called JKO scheme. As the sliced-Wasserstein distance is easy to approximate in closed-form and differentiable, we propose to approximate sliced-Wasserstein gradient flows with either a discrete grid, particles, or generative models, and by using the backpropagation to optimize each JKO step.

## Abstract

Minimizing functionals in the space of probability distributions can be done with Wasserstein gradient flows. To solve them numerically, a possible approach is to rely on the Jordan-Kinderlehrer-Otto (JKO) scheme which is analogous to the proximal scheme in Euclidean spaces. However, it requires solving a nested optimization problem at each iteration, and is known for its computational challenges, especially in high dimension. To alleviate it, very recent works propose to approximate the JKO scheme leveraging Brenier's theorem, and using gradients of Input Convex Neural Networks to parameterize the density (JKO-ICNN). However, this method comes with a high computational cost and stability issues. Instead, this work proposes to use gradient flows in the space of probability measures endowed with the sliced-Wasserstein (SW) distance. We argue that this method is more flexible than JKO-ICNN, since SW enjoys a closed-form differentiable approximation. Thus, the density at each step can be parameterized by any generative model which alleviates the computational burden and makes it tractable in higher dimensions.

## Citation

```
@article{bonet2022efficient,
    title={Efficient Gradient Flows in Sliced-Wasserstein Space},
    author={Clément Bonet and Nicolas Courty and François Septier and Lucas Drumetz},
    year={2022},
    journal={Transactions on Machine Learning Research}
}
```

## Experiments

- In the folder "Discretized Grid", you can find a notebook in which we applied the SW-JKO scheme on the Fokker-Planck equation for Gaussian examples, mixture of Gaussian and the Aggregation equation.
- In the folder "Generative Models", you can find first find notebooks on which we applied the SW-JKO scheme on the Fokker-Planck equation and on the Aggregation equation. In the folder SWF, we provide the results obtained with SW as functional on MNIST, FashionMNIST and CelebA (Section 4.3). In the folder Fokker_Planck_Gaussians, we provide the code to run the experiments of Figure 1. In the folder Bayesian_Logistic_Regression, we provide the code and the results of the experiment of Section 4.1.
- In the folder "Particles", you can find a notebook in which we applied the JKO-scheme on 2d examples (Aggregation equation, and Fokker-Planck), as well as a notebook with results obtained on the CelebA dataset by running the  SW-JKO scheme in the latent space. For the latter, the weights can be found at: https://mega.nz/folder/JdwhFQAL#n2qxvutASe7oy4bDlXBpDA

## Credits

- For the Bayesian Logistic Regression, we used some code of the [Large-Scale Wasserstein Gradient Flows](https://github.com/PetrMokrov/Large-Scale-Wasserstein-Gradient-Flows) repository.
- For the Sliced-Wasserstein flows experiments, we used autoencoders from the [swf](https://github.com/aliutkus/swf) repository.
- For the FID implementation, we used some code of the [SINF](https://github.com/biweidai/SINF) repository.
