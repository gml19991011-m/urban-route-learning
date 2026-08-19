"""
Aggregate Reinforcement-Learning Simulation of Urban Route Choice

This script extends the single-agent route-choice model by running
multiple independent simulated agents and aggregating their behaviour.

It examines:
1. Mean route-choice probabilities across repeated simulations.
2. Mean learned Q-values across repeated simulations.
3. 95% confidence intervals across simulated agents.
4. The effect of learning rate on adaptation after disruption.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Environment
# ============================================================

ROUTES = ["A", "B", "C"]

# Mean travel times before disruption
MEAN_TIME = {
    "A": 20.0,
    "B": 24.0,
    "C": 28.0
}

# Standard deviations of travel time
SD_TIME = {
    "A": 5.0,
    "B": 2.0,
    "C": 3.0
}

# Number of trials completed by each agent
N_TRIALS = 200

# Route A changes from Trial 101 onward
DISRUPTION_TRIAL = 100

# Route A becomes substantially slower after disruption
DISRUPTED_A_MEAN = 35.0


# ============================================================
# Default model parameters
# ============================================================

DEFAULT_LEARNING_RATE = 0.3

# Inverse temperature controlling exploitation versus exploration
DEFAULT_BETA = 0.3

# Initial learned values
INITIAL_Q = {
    "A": -25.0,
    "B": -25.0,
    "C": -25.0
}

# Number of independent agents in the main aggregate simulation
N_AGENTS = 500

# Number of agents simulated for each learning-rate condition
N_AGENTS_PER_LEARNING_RATE = 500

# 95% confidence interval multiplier
CI_Z = 1.96


# ============================================================
# Core model functions
# ============================================================

def softmax(
    values: list[float],
    inverse_temperature: float
) -> list[float]:
    """
    Convert learned route values into choice probabilities.

    Higher Q-values produce higher choice probabilities.
    """

    value_array = np.asarray(
        values,
        dtype=float
    )

    centred_values = (
        value_array
        - np.max(value_array)
    )

    exponentiation_values = np.exp(
        inverse_temperature
        * centred_values
    )

    probability_array = (
        exponentiation_values
        / np.sum(exponentiation_values)
    )

    return [
        float(probability)
        for probability in probability_array
    ]


def get_current_mean_time(
    route: str,
    trial_index: int
) -> float:
    """
    Return the current mean travel time of a route.

    Route A changes from 20 minutes to 35 minutes
    after Trial 100.
    """

    if (
        route == "A"
        and trial_index >= DISRUPTION_TRIAL
    ):
        return DISRUPTED_A_MEAN

    return MEAN_TIME[route]


def simulate_agent(
    agent_id: int,
    random_seed: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    inverse_temperature: float = DEFAULT_BETA
) -> pd.DataFrame:
    """
    Simulate one agent completing repeated route choices.

    The agent:
    1. Converts current Q-values into route-choice probabilities.
    2. Selects a route using softmax choice.
    3. Experiences a stochastic travel time.
    4. Converts travel time into reward.
    5. Calculates prediction error.
    6. Updates the Q-value of the chosen route.

    Returns a trial-by-trial DataFrame.
    """

    local_rng = np.random.default_rng(
        random_seed
    )

    q_values = INITIAL_Q.copy()

    agent_records = []

    for trial_index in range(N_TRIALS):

        # --------------------------------------------
        # Choice probabilities
        # --------------------------------------------

        current_values = [
            q_values[route]
            for route in ROUTES
        ]

        probabilities = softmax(
            current_values,
            inverse_temperature
        )

        # --------------------------------------------
        # Route choice
        # --------------------------------------------

        chosen_route = str(
            local_rng.choice(
                ROUTES,
                p=probabilities
            )
        )

        # --------------------------------------------
        # Environmental outcome
        # --------------------------------------------

        current_mean = get_current_mean_time(
            chosen_route,
            trial_index
        )

        travel_time = float(
            local_rng.normal(
                current_mean,
                SD_TIME[chosen_route]
            )
        )

        # --------------------------------------------
        # Reinforcement-learning update
        # --------------------------------------------

        reward = -travel_time

        prediction_error = (
            reward
            - q_values[chosen_route]
        )

        q_values[chosen_route] = (
            q_values[chosen_route]
            + learning_rate
            * prediction_error
        )

        # --------------------------------------------
        # Save trial
        # --------------------------------------------

        agent_records.append({
            "agent": agent_id,
            "trial": trial_index + 1,
            "choice": chosen_route,
            "travel_time": travel_time,
            "reward": reward,
            "prediction_error": prediction_error,
            "P_A": probabilities[0],
            "P_B": probabilities[1],
            "P_C": probabilities[2],
            "Q_A": q_values["A"],
            "Q_B": q_values["B"],
            "Q_C": q_values["C"]
        })

    return pd.DataFrame(
        agent_records
    )


# ============================================================
# Population simulation
# ============================================================

def simulate_population(
    number_of_agents: int,
    base_seed: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    inverse_temperature: float = DEFAULT_BETA,
    progress_label: str = "Population"
) -> pd.DataFrame:
    """
    Run multiple independent simulated agents.

    Every agent begins with identical initial Q-values but receives
    a different random seed.
    """

    population_frames = []

    for agent_index in range(number_of_agents):

        agent_frame = simulate_agent(
            agent_id=agent_index + 1,
            random_seed=base_seed + agent_index,
            learning_rate=learning_rate,
            inverse_temperature=inverse_temperature
        )

        population_frames.append(
            agent_frame
        )

        completed_agents = (
            agent_index + 1
        )

        if (
            completed_agents % 100 == 0
            or completed_agents == number_of_agents
        ):
            print(
                f"{progress_label}: "
                f"{completed_agents}/"
                f"{number_of_agents} agents completed"
            )

    return pd.concat(
        population_frames,
        ignore_index=True
    )


# ============================================================
# Aggregation and confidence intervals
# ============================================================

def summarise_metric(
    population_data: pd.DataFrame,
    metric_name: str,
    probability_metric: bool = False
) -> pd.DataFrame:
    """
    Calculate trial-wise mean and 95% confidence intervals
    across independently simulated agents.

    CI = mean +/- 1.96 * SEM
    """

    grouped = (
        population_data
        .groupby("trial")[metric_name]
        .agg(
            mean="mean",
            std="std",
            count="count"
        )
        .reset_index()
    )

    grouped["sem"] = (
        grouped["std"]
        / np.sqrt(grouped["count"])
    )

    grouped["ci_lower"] = (
        grouped["mean"]
        - CI_Z * grouped["sem"]
    )

    grouped["ci_upper"] = (
        grouped["mean"]
        + CI_Z * grouped["sem"]
    )

    # Choice probabilities cannot fall below 0 or above 1
    if probability_metric:
        grouped["ci_lower"] = (
            grouped["ci_lower"]
            .clip(lower=0.0)
        )

        grouped["ci_upper"] = (
            grouped["ci_upper"]
            .clip(upper=1.0)
        )

    return grouped


# ============================================================
# Plotting
# ============================================================

def plot_summary_with_ci(
    summaries: dict[str, pd.DataFrame],
    y_label: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None
) -> None:
    """
    Plot mean curves with shaded 95% confidence intervals.
    """

    plt.figure(
        figsize=(10, 5)
    )

    for label, summary in summaries.items():

        trial_values = (
            summary["trial"]
            .tolist()
        )

        mean_values = (
            summary["mean"]
            .tolist()
        )

        lower_values = (
            summary["ci_lower"]
            .tolist()
        )

        upper_values = (
            summary["ci_upper"]
            .tolist()
        )

        line = plt.plot(
            trial_values,
            mean_values,
            label=label
        )[0]

        plt.fill_between(
            trial_values,
            lower_values,
            upper_values,
            alpha=0.18,
            color=line.get_color()
        )

    plt.axvline(
        x=100.5,
        linestyle="--",
        label="Route A disruption"
    )

    plt.xlabel("Trial")
    plt.ylabel(y_label)
    plt.title(title)

    if y_limits is not None:
        plt.ylim(
            y_limits[0],
            y_limits[1]
        )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    """
    Run aggregate simulations and generate summary figures.
    """

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
            / "aggregate"
    )

    results_dir = (
            project_root
            / "results"
            / "aggregate"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Main population simulation
    # --------------------------------------------------------

    print(
        "\nRunning main aggregate simulation..."
    )

    population_data = simulate_population(
        number_of_agents=N_AGENTS,
        base_seed=10000,
        learning_rate=DEFAULT_LEARNING_RATE,
        inverse_temperature=DEFAULT_BETA,
        progress_label="Main simulation"
    )

    # --------------------------------------------------------
    # Aggregate route-choice probabilities
    # --------------------------------------------------------

    choice_summaries = {
        "Route A": summarise_metric(
            population_data,
            "P_A",
            probability_metric=True
        ),
        "Route B": summarise_metric(
            population_data,
            "P_B",
            probability_metric=True
        ),
        "Route C": summarise_metric(
            population_data,
            "P_C",
            probability_metric=True
        )
    }

    plot_summary_with_ci(
        summaries=choice_summaries,
        y_label="Mean choice probability",
        title=(
            "Mean Route-Choice Probabilities "
            "Across Simulated Agents"
        ),
        output_path=(
            figures_dir
            / "aggregate_choice_probabilities_ci.png"
        ),
        y_limits=(0.0, 1.0)
    )

    # --------------------------------------------------------
    # Aggregate Q-values
    # --------------------------------------------------------

    q_summaries = {
        "Route A": summarise_metric(
            population_data,
            "Q_A"
        ),
        "Route B": summarise_metric(
            population_data,
            "Q_B"
        ),
        "Route C": summarise_metric(
            population_data,
            "Q_C"
        )
    }

    plot_summary_with_ci(
        summaries=q_summaries,
        y_label="Mean learned Q-value",
        title=(
            "Mean Learned Route Values "
            "Across Simulated Agents"
        ),
        output_path=(
            figures_dir
            / "aggregate_q_values_ci.png"
        )
    )

    # --------------------------------------------------------
    # Save aggregate summaries
    # --------------------------------------------------------

    aggregate_choice_output = []

    for route_label, summary in choice_summaries.items():

        temporary_frame = (
            summary.copy()
        )

        temporary_frame.insert(
            1,
            "route",
            route_label
        )

        aggregate_choice_output.append(
            temporary_frame
        )

    aggregate_choice_data = pd.concat(
        aggregate_choice_output,
        ignore_index=True
    )

    aggregate_choice_data.to_csv(
        results_dir
        / "aggregate_choice_summary.csv",
        index=False
    )

    aggregate_q_output = []

    for route_label, summary in q_summaries.items():

        temporary_frame = (
            summary.copy()
        )

        temporary_frame.insert(
            1,
            "route",
            route_label
        )

        aggregate_q_output.append(
            temporary_frame
        )

    aggregate_q_data = pd.concat(
        aggregate_q_output,
        ignore_index=True
    )

    aggregate_q_data.to_csv(
        results_dir
        / "aggregate_q_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Learning-rate comparison
    # --------------------------------------------------------

    print(
        "\nRunning learning-rate comparison..."
    )

    learning_rates = [
        0.1,
        0.3,
        0.7
    ]

    learning_rate_summaries = {}

    learning_rate_output = []

    for condition_index, learning_rate_value in enumerate(
        learning_rates
    ):

        print(
            "\nLearning rate:",
            learning_rate_value
        )

        condition_population = simulate_population(
            number_of_agents=N_AGENTS_PER_LEARNING_RATE,
            base_seed=(
                30000
                + condition_index * 10000
            ),
            learning_rate=learning_rate_value,
            inverse_temperature=DEFAULT_BETA,
            progress_label=(
                f"alpha={learning_rate_value}"
            )
        )

        route_b_summary = summarise_metric(
            condition_population,
            "P_B",
            probability_metric=True
        )

        label = (
            f"alpha = {learning_rate_value}"
        )

        learning_rate_summaries[
            label
        ] = route_b_summary

        output_frame = (
            route_b_summary.copy()
        )

        output_frame.insert(
            1,
            "learning_rate",
            learning_rate_value
        )

        learning_rate_output.append(
            output_frame
        )

    plot_summary_with_ci(
        summaries=learning_rate_summaries,
        y_label=(
            "Mean probability of choosing Route B"
        ),
        title=(
            "Effect of Learning Rate "
            "on Adaptation After Disruption"
        ),
        output_path=(
            figures_dir
            / "learning_rate_comparison_ci.png"
        ),
        y_limits=(0.0, 1.0)
    )

    learning_rate_data = pd.concat(
        learning_rate_output,
        ignore_index=True
    )

    learning_rate_data.to_csv(
        results_dir
        / "learning_rate_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\nAggregate simulation complete."
    )

    print(
        "Main agents simulated:",
        N_AGENTS
    )

    print(
        "Agents per learning-rate condition:",
        N_AGENTS_PER_LEARNING_RATE
    )

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
