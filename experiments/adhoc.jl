using Revise
using TullockDynamics
using CSV
using DataFrames
using Random

maxrounds = 100000
est = select_estimator(:standard)
p = 1.0
α = 0.5
χ = 0.00005
h = maxrounds
a = 1.0
r = 1.0
n = 5
x = rand(n)
agents = [Agent(est, p, α, χ, h, a, r) for _ in 1:n]
contest = TullockContest(agents, x, maxrounds);
@profview rounds = run!(contest, ε=-1, showprogress=true)
nash_gap(contest, rounds)