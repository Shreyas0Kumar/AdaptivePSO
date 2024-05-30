### Project Name

**PSO with Dynamic Velocity Adaptation for Improved Exploration**

### Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

### Introduction

This project explores the use of Adaptive Particle Swarm Optimization (APSO) and Traditional Particle Swarm Optimization (TPSO) for training neural networks, focusing on optimizing weight settings for binary classification tasks. APSO dynamically adjusts inertia and cognitive coefficients, contrasting with the static parameters of TPSO, to improve convergence speed and solution quality.
![0_2qd8USbbeNLSj1Xp](https://github.com/Shreyas0Kumar/AdaptivePSO/assets/87386540/94e4072b-75aa-4261-b7f5-0a42c7c77639)


### Features

- Comparative analysis of Adaptive PSO and Traditional PSO for neural network training.
- Dynamically adaptive inertia and cognitive coefficients in APSO.

- Traditional, parameter-static approach in TPSO.
- Optimization of neural network weight settings for binary classification.

### Particle Position and Velocity Update Formulas

The new position and velocity of a particle \(i\) in dimension \(d\) are updated using the following formulas:

Velocity Update: $v_{i, d} = \text{inertia} \times v_{i, d} + e \times (\text{cognitive component} + \text{social component})$

Position Update: $x_{i, d} = x_{i, d} + v_{i, d}$


### Code Implementation

In your `APSO` class, the velocity and position updates are implemented in the `update_velocity` and `update_position` methods, respectively:

```python
def update_velocity(self, distances, k_nearest):
    for i in range(self.n_particles):
        r1, r2 = random.random(), random.random()

        # Select k_nearest particles
        nearest_particles_idx = np.argsort(distances[i])[1:k_nearest+1]
        nearest_particles = self.particles[nearest_particles_idx]

        # Calculate adaptation factor
        e = 1 / (1 + distances[i][nearest_particles_idx])

        # Calculate cognitive and social components
        cognitive_component = self.c1 * r1 * np.mean(self.pbest[nearest_particles_idx] - self.particles[i], axis=0)
        social_component = self.c2 * r2 * np.mean(self.gbest - self.particles[i], axis=0)
        
        # Update velocity
        inertia = self.w * self.velocities[i]
        self.velocities[i] = inertia + e * (cognitive_component + social_component)

def update_position(self):
    for i in range(self.n_particles):
        self.particles[i] += self.velocities[i]
        fitness = self.fitness(self.particles[i])
        
        # Update personal best
        if fitness > self.fitness(self.pbest[i]):
            self.pbest[i] = self.particles[i]

        # Update global best
        if fitness > self.best_fitness:
            self.gbest = self.particles[i]
            self.best_fitness = fitness

    # Store current positions
    self.particles_positions_history.append(np.copy(self.particles))
```

These methods update the velocity and position of each particle based on the specified equations, considering the cognitive and social components as well as the inertia weight and adaptation factor.

### Results

- Adaptive PSO: Gradual and sustained improvement, effective navigation and exploitation of the search space.
![download](https://github.com/Shreyas0Kumar/AdaptivePSO/assets/87386540/7dbb34d0-e890-4677-964a-32f1c461dc93)

- Traditional PSO: Rapid improvements initially, slower progress later, potential early convergence towards sub-optimal solutions.
![download](https://github.com/Shreyas0Kumar/AdaptivePSO/assets/87386540/447e711c-1d10-4c0e-bbc6-dea28f1caab0)

### Conclusion

Adaptive PSO performed better than traditional PSO in terms of search space and consistency of improvement. The adaptive behavior of APSO, inspired by natural systems, allows for more thorough exploration of the search space, preventing premature convergence.

### References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proc. IEEE Int. Conf. Neural Networks, 4(2), 1942–1948.
2. Ardizzon, G., Cavazzini, G., & Pavesi, G. (2015). Adaptive acceleration coefficients for a new search diversification strategy in particle swarm optimization algorithms. Inf Sci, 299, 337–378.
3. Gosciniak, I. (2015). A new approach to particle swarm optimization algorithm. Expert Syst Appl, 42, 844–854.
