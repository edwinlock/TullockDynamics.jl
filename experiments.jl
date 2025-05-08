using Revise
using DataFrames
using Statistics
using Random
using ProgressMeter
using TullockDynamics
using Pipe
using Plots
using StatsPlots

Random.seed!(123456789)

const ε = 0.01
const MAXROUNDS = 5000

const MAXAGENTS = 10
const EFFORTROWS = 10
const EFFORTS = rand(Float64, (EFFORTROWS, MAXAGENTS))

const ESTIMATORS = Dict(
    :standard => classic_estimator,
    :mle => max_likelihood_estimator,
    :detmle => deterministic_max_likelihood_estimator,
    :simple => dumb_estimator,
    :bayesian => bayesian_estimator,
)

""" Return `num_efforts` effort for `n` agents. """
get_efforts(row, n) = EFFORTS[row, 1:n]

select_estimator(est::Symbol) = ESTIMATORS[est]


function create_df()
    df = DataFrame(
        n=Int[],
        effort_row=Int[],
        rep=Int[],
        estimator=Symbol[],
        p=Float64[],
        α=Float64[],
        χ=Float64[],
        h=Int[],
        a=Float64[],
        r=Float64[],
        t=Union{Int, Missing}[],
    )
    return df
end


function run_instance(
        estimator::Symbol,
        x::Vector{Float64},
        p::Float64,
        α::Float64,
        χ::Float64,
        h::Int,
        a::Float64,
        r::Float64;
        ε=ε,
        showprogress=false
    )
    n = length(x)
    est = select_estimator(estimator)  # get estimator function for e symbol
    agents = [Agent(est, p, α, χ, h, a, r) for _ in 1:n]
    contest = TullockContest(agents, x, MAXROUNDS)
    num_rounds = run!(contest, ε=ε, showprogress=showprogress)
    return num_rounds ≤ MAXROUNDS ? num_rounds : missing
end


function run_experiment(;
        num_agents,
        effort_rows,
        estimators,
        pvals,
        αvals,
        χvals,
        hvals,
        avals,
        rvals,
        reps
    )
    df = create_df()
    for e in estimators
        @info "Running experiments with $(e) estimator."
        for n in num_agents
            @info "Starting n=$(n)."
            iter = Iterators.product(reps, effort_rows, pvals, αvals, χvals, hvals, avals, rvals)
            # chunks = Iterators.partition(iter, length(iter) ÷ Threads.nthreads())
            # tasks = map(chunks) do chunk
            #     Threads.@spawn run_experiment_chunk(chunk)
            # end
            @showprogress for (rep, row, p, α, χ, h, a, r) ∈ iter
                # Retrieve efforts and run instance
                x = get_efforts(row, n)
                result = run_instance(e, x, p, α, χ, h, a, r)
                # Add results as a row in data frame
                push!(df, [n row rep e p α χ h a r result])
            end
        end
    end
    return df
end


# function visualise(df, col_name)
#     plt = plot()
#     for gdf in groupby(df, :estimator)
#         lbl = String(first(gdf[!, :estimator]))
#         @df gdf plot!(:n, :t_mean, label=lbl)
#     end
#     return plt
# end

### Homogeneous agent experiments

all_estimators = [:standard, :detmle, :mle, :simple, :bayesian]
reps = 1:100


### Experiment 1: vary the number of agents
Random.seed!(1783347)
@info "Running Experiment 1: varying the number of agents."
exp1_df = run_experiment(
            num_agents=2:5,  # <- relevant bit
            effort_rows=1:10,
            estimators=all_estimators,
            pvals=[1.],
            αvals=[1.],
            χvals=[0.05],
            hvals=[MAXROUNDS],
            avals=[1.],
            rvals=[1.],
            reps=reps,
        )
# Compute statistics for each estimator
exp1_stats = @pipe exp1_df |> groupby(_, [:estimator, :n]) |> combine(_, :t => mean, :t => std, :t => (x -> [extrema(x)]) => [:min, :max])
# Plot the statistics
@df exp1_stats plot(:n, :t_mean, group = :estimator)


### Experiment 2: vary the probability of playing p.
Random.seed!(3492334)
@info "Running Experiment 2: varying the probability of playing for each agent."
exp2_df = run_experiment(
            num_agents=2:3,
            effort_rows=1:10,
            estimators=all_estimators,
            pvals=0.1:0.1:1, # <- relevant bit
            αvals=[1.],
            χvals=[0.05],
            hvals=[MAXROUNDS],
            avals=[1.],
            rvals=[1.],
            reps=reps,
        )
# Compute statistics for each estimator
exp2_stats = @pipe exp2_df |> groupby(_, [:estimator, :n, :p]) |> combine(_, :t => mean, :t => std, :t => (x -> [extrema(x)]) => [:min, :max])
# Plot the statistics
@df exp2_stats plot(:p, :t_mean, group = [:estimator, :n])


### Experiment 3: vary the step size α.
Random.seed!(8236234)
@info "Running Experiment 3: varying the step size α."
exp3_df = run_experiment(
            num_agents=2:3,
            effort_rows=1:10,
            estimators=all_estimators,
            pvals=[1.],
            αvals=0.1:0.1:1,  # <- relevant bit
            χvals=[0.05],
            hvals=[MAXROUNDS],
            avals=[1.],
            rvals=[1.],
            reps=reps,
        )
# Compute statistics for each estimator
exp3_stats = @pipe exp3_df |> groupby(_, [:estimator, :n, :α]) |> combine(_, :t => mean, :t => std, :t => (x -> [extrema(x)]) => [:min, :max])
# Plot the statistics
@df exp3_stats plot(:α, :t_mean, group = (:estimator, :n), marker=:scatter)



### Experiment 4: vary the lower bound on effort.
Random.seed!(1233765789)
@info "Running Experiment 4: varying the lower bound on effort, χ."
exp4_df = run_experiment(
            num_agents=2:3,
            effort_rows=1:10,
            estimators=all_estimators,
            pvals=[1.],
            αvals=[1.],
            χvals=0.02:0.02:0.2,    # <- relevant bit
            hvals=[MAXROUNDS],
            avals=[1.],
            rvals=[1.],
            reps=reps,
        )
# Compute statistics for each estimator
exp4_stats = @pipe exp4_df |> groupby(_, [:estimator, :n, :χ]) |> combine(_, :t => mean, :t => std, :t => (x -> [extrema(x)]) => [:min, :max])
# Plot the statistics
@df exp4_stats plot(:χ, :t_mean, group = (:estimator, :n))


### Experiment 5: vary cost function exponent.
Random.seed!(1294565789)
@info "Running Experiment 5: varying cost function exponent r."
exp5_df = run_experiment(
            num_agents=3:3,
            effort_rows=1:10,
            estimators=all_estimators,
            pvals=[1.],
            αvals=[1.],
            χvals=[0.05],    # <- relevant bit
            hvals=[MAXROUNDS],
            avals=[1.],
            rvals=1.0:0.02:1.1,
            reps=1:10,
        )
# Compute statistics for each estimator
exp5_stats = @pipe exp5_df |> groupby(_, [:estimator, :n, :r,]) |> combine(_, :t => mean, :t => std, :t => (x -> [extrema(x)]) => [:min, :max])
# Plot the statistics
@df exp5_stats plot(:r, :t_mean, group = (:estimator, :n))

# avals = 0.25:0.25:1
# rvals=1.25:0.025:1.1
# results_matrix = zeros(Float64, 5, 5)
# for a ∈ avals, r ∈ rvals
#     results_matrix[a,b] = 
# end