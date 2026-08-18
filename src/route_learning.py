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


# Simulate one travel experience for each route
for route in routes:

    travel_time = rng.normal(
        mean_time[route],
        sd_time[route]
    )

    print(
        f"Route {route}: {travel_time:.2f} minutes"
    )