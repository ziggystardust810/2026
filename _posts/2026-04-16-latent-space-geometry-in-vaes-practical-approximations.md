---
layout: distill
title: "Latent Space Geometry in VAEs: Practical Approximations for Curved Representations"
description: "Latent representations learned by generative models exhibit nontrivial geometry, yet are often analyzed with Euclidean distances. We study practical Riemannian alternatives, highlighting computational challenges and proposing scalable approximations."
date: 2026-04-16
authors:
  - name: Anonymous
toc: true
bibliography: 2026-04-16-latent-space-geometry-in-vaes-practical-approximations.bib
---

## Introduction

Latent-variable generative models are often analyzed through the geometry of their latent spaces. In practice, however, most analyses rely on Euclidean distances, implicitly assuming that latent representations are flat and homogeneous. In deep generative models such as Variational Autoencoders (VAEs), this assumption is rarely valid: nonlinear decoders induce strong distortions of the latent space, creating shortcuts through regions that are weakly constrained or unsupported by data {% cite arvanitidis2021latentspaceodditycurvature kingma2022autoencodingvariationalbayes %}. As a result, Euclidean proximity in latent space frequently fails to reflect meaningful similarity in data space.

![Which points are closer?](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/DistEuclABC.png)

A principled alternative is to endow the latent space with a geometry induced by the generator. By viewing the decoder as a smooth mapping to data space, one can define a Riemannian metric that accounts for local distortions and yields distances aligned with variations in the generated observations {% cite hauberg2019bayeslearnmanifoldon arvanitidis2021prior %}. This perspective replaces flat latent representations with curved manifolds whose structure reflects the learned data distribution and alleviates several pathologies of Euclidean latent spaces.

A key component of this geometry is uncertainty. Far from the training data, the decoder is poorly constrained, and distances that ignore predictive variance may favor paths through regions where the model lacks support. Uncertainty therefore plays a geometric role: it directly shapes distances, curvature, and geodesics in latent space, making its modeling central to meaningful interpolation and navigation.

Despite its appeal, Riemannian geometry in latent spaces poses significant practical challenges. Exact geodesic computation requires repeated evaluation of Jacobians and higher-order derivatives and relies on numerically solving differential equations. These operations are often unstable and computationally prohibitive even in moderate settings, limiting the applicability of exact geometric methods at scale.

In this work, we revisit geometry-aware latent distances from a practical standpoint. Using a VAE as a testbed {% cite kingma2022autoencodingvariationalbayes chadebec2021dataaugmentationvariationalautoencoders %}, we empirically analyze the behavior of exact Riemannian geodesics and a scalable energy-based approximation, and study how uncertainty modeling shapes the induced geometry {% cite rygaard2025likelyinterpolantsgenerativemodels %}. We further propose a simple density-aware modification of RBF-based variance models {% cite pmlr-v51-que16 %} to address failure modes in regions of sparse data support. Our results contribute to ongoing discussions on geometry at scale in deep generative models {% cite yang2018geodesicclusteringdeepgenerative pouplin2023identifyinglatentdistancesfinslerian %}.

### Contributions

Our main contributions are threefold:  
(i) we empirically demonstrate the computational and numerical limitations of exact Riemannian geodesic computation in VAE latent spaces;  
(ii) we evaluate an energy-based approximation and show that it captures meaningful manifold structure at drastically lower computational cost;  
(iii) we propose a density-aware modification of the RBF variance model, yielding more informative uncertainty geometry and more meaningful geodesics.

---

## Background and Related Work

Deep generative models aim to learn a low-dimensional latent representation that captures the structure of high-dimensional data. In this setting, distances in latent space are routinely used to guide generation, interpolation, clustering, and exploratory analysis. In practice, these distances are almost always measured using the Euclidean norm, primarily due to its simplicity and computational efficiency.

However, when the decoder is nonlinear, Euclidean distances in latent space do not generally correspond to meaningful distances in data space. Nonlinear generators can stretch and compress regions of the latent space unevenly, creating shortcuts through regions unsupported by data and distorting neighborhood relations {% cite hauberg2019bayeslearnmanifoldon arvanitidis2021latentspaceodditycurvature %}. This observation has motivated a line of work that seeks to endow latent spaces with a geometry that reflects the structure induced by the generator rather than assuming flatness.

A principled approach is to view the latent space as a Riemannian manifold equipped with a metric induced by the generator. For a deterministic decoder $g : \mathbb{R}^d \to \mathbb{R}^n$, the local geometry is captured by the Riemannian metric tensor

$$
\mathbf{G}(\mathbf{z}) = J_g(\mathbf{z})^\top J_g(\mathbf{z})
$$

where $J_g(\mathbf{z})$ denotes the Jacobian of the generator at $\mathbf{z}$ {% cite hauberg2019bayeslearnmanifoldon arvanitidis2021prior %}. Distances between latent points are then defined as geodesic lengths under this metric:

$$
d_{\mathrm{R}}(\mathbf{z}_{1}, \mathbf{z}_{2})
= \min_{\gamma}
\int_{0}^{1}
\sqrt{
\dot{\gamma}(t)^{\top}
\mathbf{G}(\gamma(t))
\dot{\gamma}(t)
} \, \mathrm{d}t
$$

This geometric perspective allows to compute a relevant distance between points on manifolds. While Euclidean distances remain widely used, they implicitly assume a flat and homogeneous latent space, an assumption that is rarely satisfied in deep generative models.

Several alternative strategies have been proposed to address latent-space distortion. Earlier work includes Jacobian-based regularization in deterministic autoencoders, Gaussian Process latent variable models that induce an exact but computationally expensive metric, and methods that explicitly learn a latent-space metric separately from the generator. More recently, geometry-aware approaches based on Riemannian and Finsler metrics have been used to analyze latent manifolds in deep generative models, enabling geometry-aware clustering and interpolation {% cite yang2018geodesicclusteringdeepgenerative arvanitidis2021prior %}.

Despite their theoretical appeal, Riemannian methods face practical challenges. Computing Jacobians, metric tensors, and geodesics quickly becomes computationally prohibitive as dimensionality and model complexity increase. As a result, most existing applications rely on approximations or simplified metrics that trade geometric fidelity for tractability. This tension between faithful geometric modeling and scalability motivates our focus on practical, approximate geometry-aware distances in variational autoencoders.

---

## Experiments

We evaluate geometry-aware distances in latent spaces by reproducing and extending the methodology of {% cite arvanitidis2021latentspaceodditycurvature %}. All experiments are conducted on the MNIST dataset using a VAE {% cite kingma2022autoencodingvariationalbayes %}. Our goals are threefold: (i) assess the practical feasibility of exact Riemannian geodesic computation, (ii) evaluate a scalable approximation based on energy minimization {% cite yang2018geodesicclusteringdeepgenerative %}, and (iii) analyze how uncertainty modeling affects the induced geometry.

### Experimental Setup

We trained a VAE with a two-dimensional latent space to allow direct visualization of the learned manifold. The encoder and mean decoder architectures follow the specifications reported in Appendix D of {% cite arvanitidis2021latentspaceodditycurvature %}. The variance of the decoder is modeled using a Radial Basis Function (RBF) network with centers fixed via $k$-means clustering in latent space.

Training is performed in two stages: first optimizing the standard ELBO objective, then fitting the RBF variance model as prescribed by Algorithm 1 in {% cite arvanitidis2021latentspaceodditycurvature %}. Unless otherwise stated, we use $K=64$ RBF centers and restrict experiments to three digit classes (1, 8, and 9) for clarity.

![Latent space representation](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/LatentSpace189.png)

---

## Uncertainty Modeling in Latent Space

The RBF variance network is designed to assign high uncertainty to regions far from the training data.

![RBF uncertainty](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_a1.3_k64.png)

As intended, uncertainty increases sharply outside the support of the data, inducing strong geometric penalties in low-density regions. This uncertainty structure plays a central role in shaping geometry-aware distances and geodesics.

---

## Limitations of Exact Riemannian Geodesics

We first attempted to compute exact geodesics by solving the Riemannian geodesic ODE as a boundary value problem. The metric tensor is defined as:

$$
M_z = J_\mu(z)^\top J_\mu(z) + J_\sigma(z)^\top J_\sigma(z)
$$

Despite being theoretically well-founded, this approach proved impractical even in two-dimensional latent spaces. Computing the gradient of the metric requires second-order automatic differentiation, leading to severe computational overhead and memory usage. Moreover, the sharp curvature induced by the RBF variance produces stiff differential equations, causing numerical solvers to fail or diverge.

---

## Energy-Based Geodesic Approximation

We adopt an alternative formulation based on energy minimization:

$$
\bar{\mathcal{E}} \approx \sum_{i=0}^{n-1} \left( \lVert \mu(\gamma_{t_i}) - \mu(\gamma_{t_{i+1}}) \rVert^2 + \lVert \sigma(\gamma_{t_i}) \rVert^2 + \lVert \sigma(\gamma_{t_{i+1}}) \rVert^2 \right)
$$

This formulation depends only on forward passes of the generator and avoids second-order derivatives entirely.

![Geodesics vs Euclidean](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/geovseucl.png)

---

## Density-Aware Modification of the RBF Variance Model

The original formulation:

$$
v_k(z) = \exp(-\lambda_k \|z - c_k\|^2)
$$

is replaced by:

$$
v_k(z) = \frac{|C_k|}{N} \exp(-\lambda_k \|z - c_k\|^2)
$$

This weighting penalizes traversal through sparse regions.

![Modified RBF](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_modif_a1.5_k64.png)

![Geodesics modified](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/Geodesics_vs_euclidean_RBF_modified.png)

---

## Interpolation

![Interpolation comparison](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/InterpolationAC.png)

---

## Conclusion and Perspectives

We revisited geometry-aware distances in variational autoencoders with a focus on practical deployment. Our results confirm that Euclidean distances are poorly aligned with similarity in latent spaces shaped by nonlinear generators, while geometry induced by the decoder better reflects the data manifold.

At the same time, we show that exact Riemannian geodesic computation is computationally expensive and numerically unstable, limiting its practical use even in low-dimensional settings.

To address this, we evaluated an energy-based approximation, which captures key geometric effects at a substantially lower cost. We further showed that uncertainty modeling plays a decisive geometric role, and proposed a simple density-aware modification of RBF variance models.

Looking forward, extending these analyses to higher-dimensional latent spaces and improving curve parametrizations remain promising directions.

---

## Appendix

![Variance 0.7](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_a0.7_k64.png)

![Variance 1](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_a1_k64.png)

![Variance 1.3](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_a1.3_k64.png)

![Variance 2](/img/2026-04-16-latent-space-geometry-in-vaes-practical-approximations/RBF_a2_k64.png)

## References

{% bibliography --cited %}
