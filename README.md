# Fuzzy Gradient Optimisation

This repository contains the full implementation and experimental framework for the undergraduate final-year project:

**“A Comparative Study of Gradient and Non-Gradient Optimisation Methods for Fuzzy Systems”**  
School of Computer Science, University of Nottingham  
Author: Xingqi Hou  

The project investigates the performance, robustness, and efficiency of gradient-based and non-gradient optimisation methods for Mamdani-type fuzzy inference systems under a unified experimental framework.

---

## Project Overview

Fuzzy inference systems (FIS) are widely used due to their interpretability, but their performance strongly depends on effective parameter optimisation. While many optimisation methods have been proposed in the literature, systematic comparison under controlled conditions is limited.

This project provides:
- A unified, modular optimisation framework for fuzzy inference systems
- A systematic comparison of gradient-based and non-gradient optimisation methods
- Large-scale experiments across multiple benchmark regression datasets
- Quantitative and qualitative analysis of optimisation behaviour

---

## Optimisation Methods Implemented

### Gradient-Based Optimisers
- Stochastic Gradient Descent (SGD)
- RMSprop
- Adam

### Non-Gradient Optimisers
- Genetic Algorithm (GA)
- Particle Swarm Optimisation (PSO)
- Differential Evolution (DE)
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

All optimisers operate on the same fuzzy-system architecture and parameter representation to ensure fair comparison.

---

## Fuzzy Inference System

- Mamdani-type fuzzy inference system
- Fully vectorised and differentiable implementation
- Trainable membership-function parameters
- Unified `TrainableFIS` interface supporting both gradient and non-gradient optimisation

---

## Experimental Setup

- **Datasets:** 20 public benchmark regression datasets
- **Preprocessing:** Min–Max normalisation to [0,1]
- **Evaluation metric:** Test-set Mean Squared Error (MSE)
- **Repetitions:** Multiple random seeds per dataset
- **Hyperparameter selection:** Successive Halving (SH)
- **Reproducibility:** Fixed seeds and configuration-driven experiments

