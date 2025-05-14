"""
Script to visualise experiment outcomes.
"""

using Revise
using CSV, DataFrames
using Plots, StatsPlots
using PrettyTables
plotlyjs()

experiment_numbers = ["1", "2", "3", "4", "6", "7", "8", "9", "X", "Y", "10", "10b"]

# Load statistic data
stats = Dict(
    exp_no => CSV.read("data/exp$(exp_no)_stats.csv", DataFrame)
        for exp_no ∈ experiment_numbers
)

# Plot statistics
exp1_plt = @df stats["1"] plot(:n, :t_mean, group = :estimator,
    marker=:scatter,
    title="Experiment 1",
    xlabel="n",
    ylabel="Rounds",
)
exp2_plt = @df stats["2"] plot(:p, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment 2",
    xlabel="p",
    ylabel="Rounds",
)
exp3_plt = @df stats["3"] plot(:α, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment 3",
    xlabel="α",
    ylabel="Rounds",
)
exp4_plt = @df stats["4"] plot(:χ, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment 4",
    xlabel="χ",
    ylabel="Rounds",
)
exp6_plt = @df stats["6"] plot(:r, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment 6",
    xlabel="r",
    ylabel="Rounds",
)

expX_plt = @df stats["X"] plot(:α, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment X",
    xlabel="α",
    ylabel="Rounds",
)

expY_plt = @df stats["Y"] plot(:p, :t_mean, group = (:estimator, :n),
    marker=:scatter,
    title="Experiment Y",
    xlabel="p",
    ylabel="Rounds",
)

exp10_plt = @df stats["10"] plot(:n, :t_mean, group = (:estimator),
    marker=:scatter,
    title="Experiment 10",
    xlabel="n",
    ylabel="Rounds",
)


exp10b_plt = @df stats["10b"] plot(:n, :t_mean, group = (:estimator),
    marker=:scatter,
    title="Experiment 10b",
    xlabel="n",
    ylabel="Rounds",
)


# Verify that heterogeneous experiments were successful
max_rounds_required = maximum(exp7_df[!,:t])
all_converge = max_rounds_required < MAXROUNDS + 1
all_converge ? println("All instances converged within $(max_rounds_required) rounds.") : println("Some instances didn't converge within $(MAXROUNDS).")

max_rounds_required = maximum(exp8_df[!,:t])
all_converge = max_rounds_required < MAXROUNDS + 1
all_converge ? println("All instances converged within $(max_rounds_required) rounds.") : println("Some instances didn't converge within $(MAXROUNDS).")

max_rounds_required = maximum(exp9_df[!,:t])
all_converge = max_rounds_required < MAXROUNDS + 1
all_converge ? println("All instances converged within $(max_rounds_required) rounds.") : println("Some instances didn't converge within $(MAXROUNDS).")



PrettyTables.pretty_table(
    # io,
    exp1_stats;
    backend = Val(:latex),
    show_subheader = false,
    # alignment=[:l, :c, :c, :c, :c],
    formatters = PrettyTables.ft_printf("%1.2f"),
    tf = PrettyTables.tf_latex_booktabs
)