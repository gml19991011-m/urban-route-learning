"""
Learning and Adaptation in Urban Route Choice

A simple reinforcement-learning model of repeated route choice
and adaptation to environmental change.
"""

import numpy as np


# Define the available routes
routes = ["A", "B", "C"]


# Mean travel time for each route (minutes)
mean_time = {
    "A": 20,
    "B": 24,
    "C": 28
}


# Standard deviation of travel time for each route
sd_time = {
    "A": 5,
    "B": 2,
    "C": 3
}


# Random-number generator
rng = np.random.default_rng(42)


# Simulate repeated travel experiences for each route
n_trials = 100

for route in routes:

    travel_times = rng.normal(
        mean_time[route],
        sd_time[route],
        size=n_trials
    )

    observed_mean = np.mean(travel_times)
    observed_sd = np.std(travel_times)

    print(f"Route {route}")
    print(f"  Mean travel time: {observed_mean:.2f} minutes")
    print(f"  Standard deviation: {observed_sd:.2f} minutes")