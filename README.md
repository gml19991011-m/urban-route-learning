# Learning and Adaptation in Urban Route Choice

A Python-based computational project exploring how repeated travel-time feedback may shape route preferences and behavioural adaptation in a changing urban environment.

## Project Overview

Urban navigation requires people to learn from repeated experience, form preferences among alternative routes, and adapt when familiar environments change.

This self-directed project develops a simple computational model of repeated route choice. The project begins with a stochastic three-route environment and will progressively incorporate reinforcement-learning mechanisms for value updating, probabilistic route selection, and adaptation following environmental disruption.

The broader aim is to explore how observable differences in route-choice behaviour may emerge from underlying learning processes.

## Research Questions

This project focuses on several simple questions:

How does an agent learn to prefer more advantageous routes through repeated travel-time feedback?
How does the agent adapt when a previously preferred route becomes unavailable or substantially less favourable?
How does the learning rate influence the speed and stability of adaptation following environmental change?

## Initial Environment

The current model contains three alternative routes with different travel-time characteristics:

| Route | Mean Travel Time | Variability | Description               |
| ----- | ---------------: | ----------: | ------------------------- |
| A     |           20 min |        High | Fast but variable         |
| B     |           24 min |         Low | Slower but stable         |
| C     |           28 min |    Moderate | Initially less attractive |

Travel times are sampled from probability distributions so that repeated journeys produce variable outcomes.

## Planned Model

The project will progressively incorporate a reinforcement-learning value update:

[
Q_{t+1}(a)=Q_t(a)+\alpha[r_t-Q_t(a)]
]

where:

* (Q_t(a)) represents the current learned value of route (a);
* (r_t) represents the experienced outcome;
* (\alpha) is the learning rate;
* (r_t-Q_t(a)) represents the prediction error.

Route choices will subsequently be generated using a softmax choice rule:

[
P(a)=\frac{e^{\beta Q(a)}}{\sum_j e^{\beta Q(j)}}
]

where (\beta) controls the degree to which choices favour routes with higher learned values.

## Development Roadmap

### Current

* [x] Define a stochastic three-route environment
* [x] Generate variable travel-time outcomes in Python

### Next

* [ ] Simulate repeated route-choice trials
* [ ] Implement reward and prediction-error calculations
* [ ] Implement reinforcement-learning value updating
* [ ] Add softmax-based probabilistic route choice
* [ ] Introduce an environmental disruption
* [ ] Visualise learned values and route-choice probabilities
* [ ] Compare agents with different learning rates
* [ ] Explore parameter recovery from simulated behavioural data

## Planned Environmental Change

A later version of the simulation will introduce an unexpected change in route conditions after an initial learning period.

For example, a route that was previously fast may become substantially slower because of congestion or infrastructure disruption.

This will allow the model to examine the transition from:

**learning → preference stabilisation → disruption → adaptation → new stabilisation**

## Tools

* Python
* NumPy
* pandas
* Matplotlib
* SciPy

## Project Status

This project is under active development.

It is intentionally designed as a small and interpretable computational behavioural modelling exercise rather than a realistic simulation of human urban navigation. The emphasis is on understanding the relationship between repeated experience, latent value updating, choice behaviour, and adaptation to environmental change.
