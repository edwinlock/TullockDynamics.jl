using Test
using TullockDynamics
using StatsBase
using Random

# Include all test files
include("test_utils.jl")
include("test_agents.jl") 
include("test_contests.jl")
include("test_estimators.jl")
include("test_dynamics.jl")
include("test_integration.jl")
include("test_edge_cases.jl")
include("test_performance.jl")
include("test_mathematical_properties.jl")
include("test_regression.jl")
include("test_plotting_extension.jl")
include("test_extension_setup.jl")