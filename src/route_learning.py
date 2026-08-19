"""
Learning and Adaptation in Urban Route Choice

A simple reinforcement-learning model of repeated route choice
and adaptation to environmental change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Environment
# ============================================================

ROUTES = ["A", "B", "C"]

MEAN_TIME = {
    "A": 20,
    "B": 24,
    "C": 28
}

SD_TIME = {
    "A": 5,
    "B": 2,
    "C": 3
}

N_TRIALS = 200

# Route A becomes slower from Trial 101 onward
DISRUPTION_TRIAL = 100

# Mean travel time of Route A after disruption
DISRUPTED_A_MEAN = 35


# ============================================================
# Model parameters
# ============================================================

LEARNING_RATE = 0.3

# Inverse temperature for softmax choice
BETA = 0.3


# ============================================================
# Model functions
# ============================================================

def softmax(values: list[float], inverse_temp: float) -> list[float]:
    """Convert learned values into choice probabilities."""

    value_array = np.array(values, dtype=float)

    exp_values = np.exp(
        inverse_temp
        * (value_array - np.max(value_array))
    )

    probabilities = exp_values / np.sum(exp_values)

    return [
        float(probability)
        for probability in probabilities
    ]


def get_mean_travel_time(
    selected_route: str,
    trial_index: int
) -> float:
    """Return the current mean travel time for a selected route."""

    if (
        trial_index >= DISRUPTION_TRIAL
        and selected_route == "A"
    ):
        return DISRUPTED_A_MEAN

    return MEAN_TIME[selected_route]


def simulate_learning_rate(
    learning_rate_value: float,
    random_seed: int
) -> list[float]:
    """
    Simulate one agent and return the probability of choosing
    Route B at every trial.
    """

    local_rng = np.random.default_rng(random_seed)

    local_q = {
        "A": -25.0,
        "B": -25.0,
        "C": -25.0
    }

    route_b_probability_history = []

    for local_trial in range(N_TRIALS):

        current_values = [
            local_q[route]
            for route in ROUTES
        ]

        current_probabilities = softmax(
            current_values,
            BETA
        )

        # Save probability of choosing Route B
        route_b_probability_history.append(
            current_probabilities[1]
        )

        selected_route = str(
            local_rng.choice(
                ROUTES,
                p=current_probabilities
            )
        )

        current_mean = get_mean_travel_time(
            selected_route,
            local_trial
        )

        experienced_time = float(
            local_rng.normal(
                current_mean,
                SD_TIME[selected_route]
            )
        )

        experienced_reward = -experienced_time

        current_prediction_error = (
            experienced_reward
            - local_q[selected_route]
        )

        local_q[selected_route] = (
            local_q[selected_route]
            + learning_rate_value
            * current_prediction_error
        )

    return route_b_probability_history


# ============================================================
# Main simulation
# ============================================================

def main() -> None:
    """Run the simulation, save results, and generate figures."""

    rng = np.random.default_rng(42)

    q_values = {
        "A": -25.0,
        "B": -25.0,
        "C": -25.0
    }

    records = []

    for trial_index in range(N_TRIALS):

        current_q_values = [
            q_values[route]
            for route in ROUTES
        ]

        choice_probabilities = softmax(
            current_q_values,
            BETA
        )

        chosen_route = str(
            rng.choice(
                ROUTES,
                p=choice_probabilities
            )
        )

        current_mean = get_mean_travel_time(
            chosen_route,
            trial_index
        )

        travel_time = float(
            rng.normal(
                current_mean,
                SD_TIME[chosen_route]
            )
        )

        reward = -travel_time

        prediction_error = (
            reward
            - q_values[chosen_route]
        )

        q_values[chosen_route] = (
            q_values[chosen_route]
            + LEARNING_RATE
            * prediction_error
        )

        records.append({
            "trial": trial_index + 1,
            "choice": chosen_route,
            "travel_time": travel_time,
            "reward": reward,
            "prediction_error": prediction_error,
            "P_A": choice_probabilities[0],
            "P_B": choice_probabilities[1],
            "P_C": choice_probabilities[2],
            "Q_A": q_values["A"],
            "Q_B": q_values["B"],
            "Q_C": q_values["C"]
        })

        print(
            f"Trial {trial_index + 1}: "
            f"P_A = {choice_probabilities[0]:.2f}, "
            f"P_B = {choice_probabilities[1]:.2f}, "
            f"P_C = {choice_probabilities[2]:.2f} | "
            f"Route {chosen_route}, "
            f"{travel_time:.2f} minutes | "
            f"Q_A = {q_values['A']:.2f}, "
            f"Q_B = {q_values['B']:.2f}, "
            f"Q_C = {q_values['C']:.2f}"
        )

    # ========================================================
    # Convert results to DataFrame
    # ========================================================

    df = pd.DataFrame(records)

    print("\nFirst five rows of the simulation data:")
    print(df.head())

    # ========================================================
    # Project directories
    # ========================================================

    project_root = Path(__file__).resolve().parents[1]

    figures_dir = project_root / "figures"
    results_dir = project_root / "results"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trial-by-trial data
    df.to_csv(
        results_dir / "simulation_results.csv",
        index=False
    )

    # Convert plotting columns to ordinary Python lists
    trial_values = df["trial"].tolist()

    p_a_values = df["P_A"].tolist()
    p_b_values = df["P_B"].tolist()
    p_c_values = df["P_C"].tolist()

    q_a_values = df["Q_A"].tolist()
    q_b_values = df["Q_B"].tolist()
    q_c_values = df["Q_C"].tolist()

    prediction_errors = df["prediction_error"].tolist()

    # ========================================================
    # Figure 1: Route choice probabilities
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        trial_values,
        p_a_values,
        label="Route A"
    )

    plt.plot(
        trial_values,
        p_b_values,
        label="Route B"
    )

    plt.plot(
        trial_values,
        p_c_values,
        label="Route C"
    )

    plt.axvline(
        x=100.5,
        linestyle="--",
        label="Route A disruption"
    )

    plt.xlabel("Trial")
    plt.ylabel("Choice probability")

    plt.title(
        "Route Choice Probabilities Across Trials"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "route_choice_probabilities.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Figure 2: Learned Q-values
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        trial_values,
        q_a_values,
        label="Route A"
    )

    plt.plot(
        trial_values,
        q_b_values,
        label="Route B"
    )

    plt.plot(
        trial_values,
        q_c_values,
        label="Route C"
    )

    plt.axvline(
        x=100.5,
        linestyle="--",
        label="Route A disruption"
    )

    plt.xlabel("Trial")
    plt.ylabel("Learned Q-value")

    plt.title(
        "Learned Route Values Across Trials"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "q_values_across_trials.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Figure 3: Prediction errors
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        trial_values,
        prediction_errors
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.axvline(
        x=100.5,
        linestyle="--",
        label="Route A disruption"
    )

    plt.xlabel("Trial")
    plt.ylabel("Prediction error")

    plt.title(
        "Prediction Errors Across Trials"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "prediction_errors_across_trials.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Figure 4: Learning-rate comparison
    # ========================================================

    learning_rates = [
        0.1,
        0.3,
        0.7
    ]

    number_of_agents = 100

    comparison_trial_values = list(
        range(1, N_TRIALS + 1)
    )

    plt.figure(figsize=(10, 5))

    for rate_value in learning_rates:

        agent_histories = []

        for agent_index in range(number_of_agents):

            agent_history = simulate_learning_rate(
                learning_rate_value=rate_value,
                random_seed=1000 + agent_index
            )

            agent_histories.append(
                agent_history
            )

        # Calculate mean P(B) across agents at each trial
        mean_route_b_probability = [
            sum(trial_probabilities)
            / len(trial_probabilities)
            for trial_probabilities
            in zip(*agent_histories)
        ]

        plt.plot(
            comparison_trial_values,
            mean_route_b_probability,
            label=f"alpha = {rate_value}"
        )

    plt.axvline(
        x=100.5,
        linestyle="--",
        label="Route A disruption"
    )

    plt.xlabel("Trial")

    plt.ylabel(
        "Probability of choosing Route B"
    )

    plt.title(
        "Effect of Learning Rate on Adaptation"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "learning_rate_comparison.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Completion message
    # ========================================================

    print("\nSimulation complete.")
    print(
        "Figures saved to:",
        figures_dir
    )
    print(
        "Results saved to:",
        results_dir
    )


if __name__ == "__main__":
    main()
