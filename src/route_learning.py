"""
Learning and Adaptation in Urban Route Choice

A simple reinforcement-learning model of repeated route choice
and adaptation to environmental change.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


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


# Initial learned value for each route
Q = {
    "A": -25.0,
    "B": -25.0,
    "C": -25.0
}


# Learning rate
alpha = 0.3


# Inverse temperature for softmax choice
beta = 0.3


def softmax(values, inverse_temp):
    values = np.array(values)

    exp_values = np.exp(
        inverse_temp * (values - np.max(values))
    )

    return exp_values / exp_values.sum()


# Number of repeated route choices
n_trials = 200


# Route A becomes slower after trial 100
disruption_trial = 100


# Store trial-by-trial simulation results
records = []


for trial in range(n_trials):

    # Current learned values
    q_values = [Q[route] for route in routes]

    # Convert Q-values into choice probabilities
    probabilities = softmax(
        q_values,
        beta
    )

    # Choose a route according to softmax probabilities
    choice = rng.choice(
        routes,
        p=probabilities
    )

    # Environmental disruption:
    # Route A becomes slower from Trial 101 onward
    if trial >= disruption_trial and choice == "A":
        current_mean_time = 35
    else:
        current_mean_time = mean_time[choice]

    # Experience travel time
    travel_time = rng.normal(
        current_mean_time,
        sd_time[choice]
    )

    # Shorter travel time = higher reward
    reward = -travel_time

    # Prediction error
    prediction_error = reward - Q[choice]

    # Update the value of the chosen route
    Q[choice] = (
        Q[choice]
        + alpha * prediction_error
    )

    # Save this trial
    records.append({
        "trial": trial + 1,
        "choice": choice,
        "travel_time": travel_time,
        "reward": reward,
        "prediction_error": prediction_error,
        "P_A": probabilities[0],
        "P_B": probabilities[1],
        "P_C": probabilities[2],
        "Q_A": Q["A"],
        "Q_B": Q["B"],
        "Q_C": Q["C"]
    })

    print(
        f"Trial {trial + 1}: "
        f"P_A = {probabilities[0]:.2f}, "
        f"P_B = {probabilities[1]:.2f}, "
        f"P_C = {probabilities[2]:.2f} | "
        f"Route {choice}, "
        f"{travel_time:.2f} minutes | "
        f"Q_A = {Q['A']:.2f}, "
        f"Q_B = {Q['B']:.2f}, "
        f"Q_C = {Q['C']:.2f}"
    )


# Convert all saved trials into a pandas DataFrame
df = pd.DataFrame(records)


# Display the first five rows
print("\nFirst five rows of the simulation data:")
print(df.head())


# Define project directories
project_root = Path(__file__).resolve().parents[1]
figures_dir = project_root / "figures"
results_dir = project_root / "results"

# Save simulation data
df.to_csv(
    results_dir / "simulation_results.csv",
    index=False
)


# Plot route choice probabilities across trials
plt.figure(figsize=(10, 5))

plt.plot(
    df["trial"],
    df["P_A"],
    label="Route A"
)

plt.plot(
    df["trial"],
    df["P_B"],
    label="Route B"
)

plt.plot(
    df["trial"],
    df["P_C"],
    label="Route C"
)

# Mark the disruption between Trial 100 and Trial 101
plt.axvline(
    x=100.5,
    linestyle="--",
    label="Route A disruption"
)

plt.xlabel("Trial")
plt.ylabel("Choice probability")
plt.title("Route Choice Probabilities Across Trials")

plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(
    figures_dir / "route_choice_probabilities.png",
    dpi=300
)
