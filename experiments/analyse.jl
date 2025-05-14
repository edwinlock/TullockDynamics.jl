"""
Script to analyse experiment outcomes.
"""

using CSV, DataFrames
using Pipe
using Statistics
using StatsPlots

countmissing(col) = count(x -> ismissing(x), col)

"""
Define function that analyses each experiment's data and returns a stats df.
"""
function analyse(data; groups=[:estimator, :n])
    stats = @pipe data |> groupby(_, groups) |> combine(_,
        :t => mean,
        :t => std,
        :t => (x -> [extrema(x)]) => [:min, :max], :t => countmissing => "#nonconverge")
    return stats
end

experiment_numbers = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "C1", "C2", "C3", "C4"]

# Specify how to group the datapoints in each experiment:
groups = Dict(
    "A1" => [:estimator, :n],
    "A2" => [:estimator, :n, :p],
    "A3" => [:estimator, :n, :α],
    "A4" => [:estimator, :n, :χ],
    "A5" => [:estimator, :n, :r],
    "B1" => [:estimator, :n, :α],
    "B2" => [:estimator, :n, :p],
    "C1" => [:estimator, :n],
    "C2" => [:estimator, :n],
    "C3" => [:estimator, :n],
    "C4" => [:estimator, :n],
)


# Load the data
data = Dict(
    exp_no => CSV.read("data/exp$(exp_no)_data.csv", DataFrame)
        for exp_no ∈ experiment_numbers
)

# Analyse the data
stats = Dict(
    exp_no => analyse(data[exp_no], groups=groups[exp_no]) for exp_no in experiment_numbers
)

# Save data
for exp_no in experiment_numbers
    CSV.write("data/exp$(exp_no)_stats.csv", stats[exp_no])
end