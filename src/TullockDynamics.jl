module TullockDynamics

using StatsBase  # used to determine winner in each round
using ForwardDiff  # used to differentiate cost function
using Integrals  # used to compute Bayesian estimates
using Plots
using ProgressMeter
using Measures

include("agents.jl")
include("contests.jl")
include("dynamics.jl")
include("estimators.jl")
include("utils.jl")

# Exports from agents.jl
export Agent, MLEAgent, DetMLEAgent, DumbAgent, BayesianAgent
export utility, best_response, nash_gap, max_agent_effort
# Exports from contests.jl
export TullockContest
export nash_gap, num_rounds
# Exports from dynamics.jl
export set_efforts!, set_utilities!, step!, run!
# Exports from estimators.jl
export max_likelihood_estimator, deterministic_max_likelihood_estimator, dumb_estimator, bayesian_estimator, classic_estimator
# Exports from utils.jl
export find_root, show, final_efforts, visualise, convergence_status

end # module TullockDynamics
