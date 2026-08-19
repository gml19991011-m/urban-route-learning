"""
Parameter Recovery for the Urban Route-Choice Model

This script demonstrates a minimal likelihood-based parameter
recovery analysis.

Workflow:
1. Generate synthetic behavioural data using known alpha and beta.
2. Treat alpha and beta as unknown.
3. Test many candidate alpha-beta combinations.
4. Calculate the negative log-likelihood of the observed choices.
5. Recover the parameter combination that best explains behaviour.

This is a synthetic demonstration rather than an analysis of
empirical human data.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Environment
# ============================================================

ROUTES = ["A", "B", "C"]

MEAN_TIME = {
    "A": 20.0,
    "B": 24.0,
    "C": 28.0
}

SD_TIME = {
    "A": 5.0,
    "B": 2.0,
    "C": 3.0
}

N_TRIALS = 200

# Route A becomes worse from Trial 101 onward.
DISRUPTION_TRIAL = 100

DISRUPTED_A_MEAN = 35.0

INITIAL_Q = {
    "A": -25.0,
    "B": -25.0,
    "C": -25.0
}


# ============================================================
# 2. True parameters
# ============================================================

# These parameters generate the synthetic participant.
# During fitting, we pretend that we do not know them.

TRUE_ALPHA = 0.30
TRUE_BETA = 0.40

RANDOM_SEED = 31415


# ============================================================
# 3. Candidate parameter grid
# ============================================================

# Alpha:
# 0.05, 0.06, 0.07, ..., 0.95

ALPHA_GRID = [
    value / 100
    for value in range(5, 96)
]

# Beta:
# 0.05, 0.06, 0.07, ..., 1.00

BETA_GRID = [
    value / 100
    for value in range(5, 101)
]

# Avoid log(0) if a choice probability becomes extremely small.
MIN_PROBABILITY = 1e-12


# ============================================================
# 4. Softmax
# ============================================================

def softmax(
    values: list[float],
    inverse_temperature: float
) -> list[float]:
    """
    Convert learned Q-values into route-choice probabilities.
    """

    scaled_values = [
        inverse_temperature * value
        for value in values
    ]

    # Numerical-stability correction.
    maximum_value = max(
        scaled_values
    )

    exp_values = [
        math.exp(
            value - maximum_value
        )
        for value in scaled_values
    ]

    total = sum(
        exp_values
    )

    probabilities = [
        value / total
        for value in exp_values
    ]

    return probabilities


# ============================================================
# 5. Environmental change
# ============================================================

def get_current_mean_time(
    route: str,
    trial_index: int
) -> float:
    """
    Return the current mean travel time for a route.
    """

    if (
        route == "A"
        and trial_index >= DISRUPTION_TRIAL
    ):
        return DISRUPTED_A_MEAN

    return MEAN_TIME[route]


# ============================================================
# 6. Generate one synthetic participant
# ============================================================

def simulate_synthetic_participant() -> pd.DataFrame:
    """
    Generate behavioural data using known TRUE_ALPHA and TRUE_BETA.

    Only observable variables are saved:
    trial, choice, travel time, and reward.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    q_values = INITIAL_Q.copy()

    records = []

    for trial_index in range(
        N_TRIALS
    ):

        # ----------------------------------------------------
        # Step 1: Current learned route values
        # ----------------------------------------------------

        current_values = [
            q_values[route]
            for route in ROUTES
        ]

        # ----------------------------------------------------
        # Step 2: Convert Q-values into choice probabilities
        # ----------------------------------------------------

        probabilities = softmax(
            current_values,
            TRUE_BETA
        )

        # ----------------------------------------------------
        # Step 3: Choose a route
        # ----------------------------------------------------

        chosen_route = str(
            rng.choice(
                ROUTES,
                p=probabilities
            )
        )

        # ----------------------------------------------------
        # Step 4: Experience travel time
        # ----------------------------------------------------

        current_mean = get_current_mean_time(
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

        # ----------------------------------------------------
        # Step 5: Prediction error
        # ----------------------------------------------------

        prediction_error = (
            reward
            - q_values[chosen_route]
        )

        # ----------------------------------------------------
        # Step 6: Update Q-value using the TRUE alpha
        # ----------------------------------------------------

        q_values[chosen_route] += (
            TRUE_ALPHA
            * prediction_error
        )

        # ----------------------------------------------------
        # Step 7: Save only observable data
        # ----------------------------------------------------

        records.append({
            "trial": trial_index + 1,
            "choice": chosen_route,
            "travel_time": travel_time,
            "reward": reward
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# 7. Negative log-likelihood
# ============================================================

def negative_log_likelihood(
    observed_choices: list[str],
    observed_rewards: list[float],
    candidate_alpha: float,
    candidate_beta: float
) -> float:
    """
    Calculate how well one candidate alpha-beta combination
    explains the observed choices.

    Lower negative log-likelihood means a better fit.
    """

    q_values = INITIAL_Q.copy()

    total_nll = 0.0

    for (
        observed_choice,
        observed_reward
    ) in zip(
        observed_choices,
        observed_rewards
    ):

        # ----------------------------------------------------
        # Step 1:
        # Candidate model has its own current Q-values.
        # ----------------------------------------------------

        current_values = [
            q_values[route]
            for route in ROUTES
        ]

        # ----------------------------------------------------
        # Step 2:
        # Candidate beta converts Q-values into probabilities.
        # ----------------------------------------------------

        probabilities = softmax(
            current_values,
            candidate_beta
        )

        # ----------------------------------------------------
        # Step 3:
        # Find probability assigned to the ACTUAL choice.
        # ----------------------------------------------------

        choice_index = ROUTES.index(
            observed_choice
        )

        observed_probability = max(
            probabilities[
                choice_index
            ],
            MIN_PROBABILITY
        )

        # ----------------------------------------------------
        # Step 4:
        # Add this trial's negative log-likelihood.
        # ----------------------------------------------------

        total_nll -= math.log(
            observed_probability
        )

        # ----------------------------------------------------
        # Step 5:
        # Replay the ACTUAL observed reward.
        # ----------------------------------------------------

        prediction_error = (
            observed_reward
            - q_values[
                observed_choice
            ]
        )

        # ----------------------------------------------------
        # Step 6:
        # Candidate alpha determines the Q-value update.
        # ----------------------------------------------------

        q_values[
            observed_choice
        ] += (
            candidate_alpha
            * prediction_error
        )

    return total_nll


# ============================================================
# 8. Grid-search parameter recovery
# ============================================================

def recover_parameters(
    observed_choices: list[str],
    observed_rewards: list[float]
) -> tuple[
    float,
    float,
    float,
    list[list[float]]
]:
    """
    Test every candidate alpha-beta combination.

    The combination with the lowest negative log-likelihood
    is selected as the recovered estimate.
    """

    likelihood_surface = []

    best_nll = float(
        "inf"
    )

    recovered_alpha = (
        ALPHA_GRID[0]
    )

    recovered_beta = (
        BETA_GRID[0]
    )

    total_alpha_values = len(
        ALPHA_GRID
    )

    for (
        alpha_index,
        candidate_alpha
    ) in enumerate(
        ALPHA_GRID
    ):

        row_nll_values = []

        for candidate_beta in (
            BETA_GRID
        ):

            current_nll = (
                negative_log_likelihood(
                    observed_choices=observed_choices,
                    observed_rewards=observed_rewards,
                    candidate_alpha=candidate_alpha,
                    candidate_beta=candidate_beta
                )
            )

            row_nll_values.append(
                current_nll
            )

            # Keep the best-fitting parameter combination.
            if current_nll < best_nll:

                best_nll = (
                    current_nll
                )

                recovered_alpha = (
                    candidate_alpha
                )

                recovered_beta = (
                    candidate_beta
                )

        likelihood_surface.append(
            row_nll_values
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        completed = (
            alpha_index + 1
        )

        if (
            completed % 10 == 0
            or completed
            == total_alpha_values
        ):
            print(
                f"Parameter search: "
                f"{completed}/"
                f"{total_alpha_values} "
                f"alpha values completed"
            )

    return (
        recovered_alpha,
        recovered_beta,
        best_nll,
        likelihood_surface
    )


# ============================================================
# 9. Convert likelihood surface to DataFrame
# ============================================================

def create_likelihood_surface_dataframe(
    likelihood_surface: list[
        list[float]
    ]
) -> pd.DataFrame:
    """
    Convert the likelihood grid into a table for saving.
    """

    records = []

    for (
        alpha_index,
        alpha_value
    ) in enumerate(
        ALPHA_GRID
    ):

        for (
            beta_index,
            beta_value
        ) in enumerate(
            BETA_GRID
        ):

            records.append({
                "alpha": alpha_value,
                "beta": beta_value,
                "negative_log_likelihood":
                    likelihood_surface[
                        alpha_index
                    ][
                        beta_index
                    ]
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# 10. Plot likelihood surface
# ============================================================

def plot_likelihood_surface(
    likelihood_surface: list[
        list[float]
    ],
    recovered_alpha: float,
    recovered_beta: float,
    output_path: Path
) -> None:
    """
    Plot the negative log-likelihood surface.

    True and recovered parameters are marked on the same figure.
    """

    plt.figure(
        figsize=(9, 6)
    )

    contour = plt.contourf(
        BETA_GRID,
        ALPHA_GRID,
        likelihood_surface,
        levels=30
    )

    plt.colorbar(
        contour,
        label=(
            "Negative log-likelihood"
        )
    )

    # --------------------------------------------------------
    # True parameter location
    # --------------------------------------------------------

    plt.scatter(
        [TRUE_BETA],
        [TRUE_ALPHA],
        marker="x",
        s=120,
        label=(
            f"True: alpha = {TRUE_ALPHA:.2f}, "
            f"beta = {TRUE_BETA:.2f}"
        )
    )

    plt.scatter(
        [recovered_beta],
        [recovered_alpha],
        marker="o",
        s=90,
        label=(
            f"Recovered: alpha = {recovered_alpha:.2f}, "
            f"beta = {recovered_beta:.2f}"
        )
    )

    plt.xlabel(
        "Inverse temperature (beta)"
    )

    plt.ylabel(
        "Learning rate (alpha)"
    )

    plt.title(
        "Likelihood-Based Parameter Recovery"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    # Do not use plt.show() because of the previous
    # PyCharm Matplotlib backend issue.
    plt.close()


# ============================================================
# 11. Main analysis
# ============================================================

def main() -> None:
    """
    Generate synthetic behaviour and recover alpha and beta.
    """

    print(
        "\n=== Running parameter_recovery.py ==="
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    figures_dir = (
        project_root
        / "figures"
    )

    results_dir = (
        project_root
        / "results"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Part 1: Generate synthetic behavioural data
    # ========================================================

    print(
        "\nStep 1: "
        "Generating synthetic behavioural data..."
    )

    behavioural_data = (
        simulate_synthetic_participant()
    )

    behavioural_data.to_csv(
        results_dir
        / "synthetic_participant.csv",
        index=False
    )

    print(
        "\nFirst five observed trials:"
    )

    print(
        behavioural_data.head()
    )

    # --------------------------------------------------------
    # Extract observed choices and rewards.
    # These are the data used for fitting.
    # --------------------------------------------------------

    observed_choices = [
        str(value)
        for value in behavioural_data[
            "choice"
        ].tolist()
    ]

    observed_rewards = [
        float(value)
        for value in behavioural_data[
            "reward"
        ].tolist()
    ]

    # ========================================================
    # Part 2: Treat alpha and beta as unknown
    # ========================================================

    print(
        "\nStep 2: "
        "Treating alpha and beta as unknown..."
    )

    print(
        "The fitting procedure uses only "
        "the observed choices and rewards."
    )

    # ========================================================
    # Part 3: Fit candidate parameters
    # ========================================================

    print(
        "\nStep 3: "
        "Searching candidate alpha and beta values..."
    )

    (
        recovered_alpha,
        recovered_beta,
        best_nll,
        likelihood_surface
    ) = recover_parameters(
        observed_choices=observed_choices,
        observed_rewards=observed_rewards
    )

    # ========================================================
    # Part 4: Compare true and recovered parameters
    # ========================================================

    alpha_error = abs(
        TRUE_ALPHA
        - recovered_alpha
    )

    beta_error = abs(
        TRUE_BETA
        - recovered_beta
    )

    print(
        "\n=== Parameter Recovery Result ==="
    )

    print(
        f"True alpha:       "
        f"{TRUE_ALPHA:.2f}"
    )

    print(
        f"Recovered alpha:  "
        f"{recovered_alpha:.2f}"
    )

    print(
        f"Absolute error:   "
        f"{alpha_error:.2f}"
    )

    print()

    print(
        f"True beta:        "
        f"{TRUE_BETA:.2f}"
    )

    print(
        f"Recovered beta:   "
        f"{recovered_beta:.2f}"
    )

    print(
        f"Absolute error:   "
        f"{beta_error:.2f}"
    )

    print()

    print(
        f"Best negative log-likelihood: "
        f"{best_nll:.3f}"
    )

    # ========================================================
    # Part 5: Save summary table
    # ========================================================

    summary_data = pd.DataFrame({
        "true_alpha": [
            TRUE_ALPHA
        ],
        "recovered_alpha": [
            recovered_alpha
        ],
        "alpha_absolute_error": [
            alpha_error
        ],
        "true_beta": [
            TRUE_BETA
        ],
        "recovered_beta": [
            recovered_beta
        ],
        "beta_absolute_error": [
            beta_error
        ],
        "best_negative_log_likelihood": [
            best_nll
        ]
    })

    summary_data.to_csv(
        results_dir
        / "parameter_recovery_summary.csv",
        index=False
    )

    # ========================================================
    # Part 6: Save complete likelihood surface
    # ========================================================

    likelihood_surface_data = (
        create_likelihood_surface_dataframe(
            likelihood_surface
        )
    )

    likelihood_surface_data.to_csv(
        results_dir
        / "likelihood_surface.csv",
        index=False
    )

    # ========================================================
    # Part 7: Generate figure
    # ========================================================

    plot_likelihood_surface(
        likelihood_surface=likelihood_surface,
        recovered_alpha=recovered_alpha,
        recovered_beta=recovered_beta,
        output_path=(
            figures_dir
            / "parameter_recovery_surface.png"
        )
    )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\nParameter recovery complete."
    )

    print(
        "Synthetic data saved to:",
        results_dir
        / "synthetic_participant.csv"
    )

    print(
        "Summary saved to:",
        results_dir
        / "parameter_recovery_summary.csv"
    )

    print(
        "Likelihood surface saved to:",
        results_dir
        / "likelihood_surface.csv"
    )

    print(
        "Figure saved to:",
        figures_dir
        / "parameter_recovery_surface.png"
    )


if __name__ == "__main__":
    main()
