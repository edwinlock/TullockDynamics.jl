using Revise
using DataFrames
using Statistics
using Random
using ProgressMeter
using TullockDynamics
using Pipe
using CSV

println("Using $(Threads.nthreads()) threads.")
 
### Preliminaries
const ESTIMATORS = Dict(
    :standard => classic_estimator,
    :mle => max_likelihood_estimator,
    :detmle => deterministic_max_likelihood_estimator,
    :heuristic => dumb_estimator,
    :bayesian => bayesian_estimator,
)

const ε = 0.001
const MAXROUNDS = 20000
const MAXAGENTS = 10
const EFFORTROWS = 1000

Random.seed!(123456789)
const EFFORTS = rand(Float64, (EFFORTROWS, MAXAGENTS))

efforts_df = DataFrame(EFFORTS, :auto)
CSV.write("data/efforts.csv", efforts_df)

""" Return `num_efforts` effort for `n` agents. """
get_efforts(row, n) = EFFORTS[row, 1:n]

select_estimator(est::Symbol) = ESTIMATORS[est]

all_estimators = [:standard, :detmle, :mle, :heuristic, :bayesian]
fast_estimators = [:detmle, :mle, :heuristic]

### Preliminary functions

"""
Run a homogeneous instance.
"""
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
    try
        return run!(contest, ε=ε, showprogress=showprogress)
    catch e
        println("Encountered error $(e)")
        println("Instance was:")
        println("Estimator $(estimator)")
        println("p: $(p)")
        println("α: $(α)")
        println("χ: $(χ)")
        println("a: $(a)")
        println("r: $(r)")
    end
    throw(e)
end


"""
Run a heterogeneous instance.
"""
function run_instance(
        estimator::Symbol,
        x::Vector{Float64},
        p::Vector{Float64},
        α::Vector{Float64},
        χ::Vector{Float64},
        h::Vector{Int},
        a::Vector{Float64},
        r::Vector{Float64};
        ε=ε,
        showprogress=false
    )
    n = length(x)
    est = select_estimator(estimator)  # get estimator function for `estimator` symbol
    agents = [Agent(est, p[i], α[i], χ[i], h[i], a[i], r[i]) for i in 1:n]
    contest = TullockContest(agents, x, MAXROUNDS)
    num_rounds = run!(contest, ε=ε, showprogress=showprogress)
    # return num_rounds
    return num_rounds ≤ MAXROUNDS ? num_rounds : missing
end

"""
Create empty dataframe for homogeneous agents experiments.
"""
function create_hom_df()
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

"""
Create empty dataframe for heterogeneous agents experiments.
"""
function create_het_df()
    df = DataFrame(
        n=Int[],
        rep=Int[],
        estimator=Symbol[],
        t=Union{Int, Missing}[],
    )
    return df
end


""" Run an experiment with homogeneous agents. """
function run_hom_experiment(;
        num_agents,
        effort_rows,
        estimators,
        pvals,
        αvals,
        χvals,
        hvals,
        avals,
        rvals,
        reps::Int
    )
    df = create_hom_df()
    for e in estimators
        @info "Running experiments with $(e) estimator."
        for n in num_agents
            @info "Starting n=$(n)."
            iter = Iterators.product(1:reps, effort_rows, pvals, αvals, χvals, hvals, avals, rvals)
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


""" Return random number in range first(r) and last(r). """
randrange(r::Tuple) = first(r) + (last(r) - first(r))*rand()
""" Return a vector of random numbers in range first(r) and last(r). """
randrange(r::Tuple, n) = [ first(r) + (last(r) - first(r))*rand() for _ in 1:n ]


""" Run an experiment with heterogeneous agents. """
function run_het_experiment(;
        num_agents,
        estimators,
        xrange,
        prange,
        αrange,
        χrange,
        arange,
        rrange,
        reps::Int
    )
    df = create_het_df()
    for e in estimators
        @info "Running experiments with $(e) estimator."
        for n in num_agents
            @info "Starting n=$(n)."
            @showprogress for rep ∈ 1:reps
                # Generate random efforts and agent parameters
                x = randrange(xrange, n)
                p = randrange(prange, n)
                α = randrange(αrange, n)
                χ = randrange(χrange, n)
                h = fill(MAXROUNDS, n)
                a = randrange(arange, n)
                r = randrange(rrange, n)
                result = run_instance(e, x, p, α, χ, h, a, r)
                # Add results as a row in data frame
                push!(df, [n rep e result])
                if ismissing(result)
                    @warn "Instance with estimator $(e) and $(n) agents did not converge in $(MAXROUNDS) rounds."
                    return df
                end
            end
        end
    end
    return df
end

# Define homogeneous experiments
reps_hom = 1
effort_rows = 1:10

# Experiment Block A: instances with homogeneous agents in which we vary one parameter at a time
# Experiment A1: vary n.
function experimentA1()
    experiment_number = "A1"
    Random.seed!(1783344867)
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=2:4,  # <- relevant bit
        effort_rows=effort_rows,
        estimators=all_estimators,
        pvals=[1.],
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_hom,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end

# Experiment A2: vary p.
function experimentA2()
    Random.seed!(3934683492334)
    experiment_number = "A2"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4],
        effort_rows=effort_rows,
        estimators=all_estimators,
        pvals=0.1:0.1:1, # <- relevant bit
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment A2b: vary the probability of playing p for all fast dynamics
function experimentA2b()
    Random.seed!(3492321434)
    experiment_number = "A2b"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4,6,8],
        effort_rows=effort_rows,
        estimators=fast_estimators,
        pvals=0.1:0.1:1, # <- relevant bit
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=1,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment A3: vary the step size α.
function experimentA3()
    Random.seed!(82362124234)
    experiment_number = "A3"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4],
        effort_rows=effort_rows,
        estimators=all_estimators,
        pvals=[1.],
        αvals=0.1:0.1:1,    # <- relevant bit
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment A3b: vary the step size α for all estimators apart from bayesian
function experimentA3b()
    Random.seed!(8236123867124234)
    experiment_number = "A3b"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4,6,8],
        effort_rows=effort_rows,
        estimators=fast_estimators,
        pvals=[1.],
        αvals=0.1:0.1:1,  # <- relevant bit
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=1,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end



# Experiment A4: vary the lower bound on effort.
function experimentA4()
    Random.seed!(1233765789)
    experiment_number = "A4"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4],
        effort_rows=effort_rows,
        estimators=all_estimators,
        pvals=[1.],
        αvals=[1.],
        χvals=0.02:0.02:0.2,    # <- relevant bit
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment A4b: vary the lower bound on effort for all estimators apart from Bayesian.
function experimentA4b()
    Random.seed!(9233765789)
    experiment_number = "A4b"
    @info "Starting Experiment $(experiment_number)."
    df = run_hom_experiment(
        num_agents=[2,4,6,8],
        effort_rows=effort_rows,
        estimators=fast_estimators,
        pvals=[1.],
        αvals=[1.],
        χvals=0.02:0.02:0.2,    # <- relevant bit
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=1,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment A5: vary cost function parameter r.
function experimentA5()
    Random.seed!(7891294565)
    experiment_number = "A5"
    @info "Starting Experiment $(experiment_number)."
    rvals = 1.0:0.1:1.5
    df = run_hom_experiment(
        num_agents=[2,4],
        effort_rows=effort_rows,
        estimators=all_estimators,
        pvals=[1.],
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=rvals,  # <- relevant bit
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


### Experiment A5b: vary cost function parameter r.
function experimentA5b()
    Random.seed!(6578912945)
    experiment_number = "A5b"
    @info "Starting Experiment $(experiment_number)."
    rvals = 1.0:0.025:1.1
    df = run_hom_experiment(
        num_agents=[2,4,6,8],
        effort_rows=effort_rows,
        estimators=fast_estimators,
        pvals=[1.],
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=rvals,  # <- relevant bit
        reps=1,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment Block B: see for which values of p and α the standard dynamics fails to converge.

# Experiment B1: vary α
function experimentB1()
    Random.seed!(84572324234)
    experiment_number = "B1"
    @info "Starting Experiment $(experiment_number)."
    αvals = 0.5:0.1:1.0
    df = run_hom_experiment(
        num_agents=[2,3,4,5,6],
        effort_rows=effort_rows,
        estimators=[:standard],
        pvals=[1.],
        αvals=αvals,    # <- relevant bit
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end

# Experiment B2: vary p
function experimentB2()
    Random.seed!(12384524234)
    experiment_number = "B2"
    @info "Starting Experiment $(experiment_number)."
    pvals = 0.5:0.1:1
    df = run_hom_experiment(
        num_agents=[2,3,4,5,6],
        effort_rows=effort_rows,
        estimators=[:standard],
        pvals=pvals,    # <- relevant bit
        αvals=[1.],
        χvals=[0.05],
        hvals=[MAXROUNDS],
        avals=[1.],
        rvals=[1.],
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


### Heterogeneous agents experiments

reps_het = 1000

# Experiment Block C: heterogeneous agents with randomly chosen values.

# Experiment C1: linear costs but random values of p and α
function experimentC1()
    Random.seed!(1783251242)
    experiment_number = "C1"
    @info "Starting Experiment $(experiment_number)."
    df = run_het_experiment(
        num_agents=2:2,
        estimators=all_estimators,
        xrange=(0.05, 1),
        prange=(0.1, 1.),
        αrange=(0.1, 1.),
        χrange=(0.05,0.05),
        arange=(1.,1.),
        rrange=(1.,1.),
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment C2: arbitrary p, α, and r ∈ (1, 1.5), for the three fast dynamics.
function experimentC2()
    Random.seed!(937531242)
    experiment_number = "C2"
    @info "Starting Experiment $(experiment_number)."
    df = run_het_experiment(
        num_agents=[2, 4, 6],
        estimators=[:mle, :detmle, :heuristic],
        xrange=(0.05, 1),
        prange=(0.1, 1.),
        αrange=(0.1, 1.),
        χrange=(0.05, 0.05),
        arange=(1.0, 1.0),
        rrange=(1.0, 1.5),
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment C3: arbitrary p, α, and r = 1, for the fast dynamics.
function experimentC3()
    Random.seed!(246351242)
    experiment_number = "C3"
    @info "Starting Experiment $(experiment_number)."
    df = run_het_experiment(
        num_agents=[2, 4, 6, 8],
        estimators=[:mle],
        xrange=(0.05, 1),
        prange=(0.1, 1.),
        αrange=(0.1, 1.),
        χrange=(0.05,0.05),
        arange=(1.,1.),
        rrange=(1.,1.),
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment C4: small values for p and α?
function experiment9()
    Random.seed!(45682745)
    experiment_number = 9
    @info "Starting Experiment $(experiment_number)."
    df = run_het_experiment(
        num_agents=2:3,
        estimators=all_estimators,
        xrange=(0.05, 1),
        prange=(0.1, 1.),
        αrange=(0.1, 1.),
        χrange=(0.05, 0.05),
        arange=(1.0, 1.0),
        rrange=(1.0, 1.5),
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


# Experiment S1: 
function experimentC2()
    Random.seed!(6201783347)
    experiment_number = "C2"
    @info "Starting Experiment $(experiment_number)."
    df = run_het_experiment(
        num_agents=2:3,
        estimators=all_estimators,
        xrange=(0.05, 1),
        prange=(1., 1.),
        αrange=(1., 1.),
        χrange=(0.05, 0.05),
        arange=(1., 1.),
        rrange=(1., 1.1),
        reps=reps_het,
    )
    CSV.write("data/exp$(experiment_number)_data.csv", df)
    return nothing
end


using ArgParse

function parse_commandline(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--experiments"
            help = "specify which experiments to run (1-9)"
            arg_type = String
            default = "[1,2,3,4,5,7,8,9]"
    end

    return parse_args(args, s)
end

function (@main)(args)
    parsed_args = parse_commandline(args)
    experiments = split(parsed_args["experiments"], ",")
    "A1" ∈ experiments && experimentA1()
    "A2" ∈ experiments && experimentA2()
    "A3" ∈ experiments && experimentA3()
    "A4" ∈ experiments && experimentA4()
    "A5" ∈ experiments && experimentA5()
    "B1" ∈ experiments && experimentB1()
    "B2" ∈ experiments && experimentB2()
    "C1" ∈ experiments && experimentC1()
    "C2" ∈ experiments && experimentC2()
    "C3" ∈ experiments && experimentC3()
    "C4" ∈ experiments && experimentC4()
end



