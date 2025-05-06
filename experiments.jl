using DataFrames
using Random


function run_experiments(outfile; estimators, xvals, pvals, αvals, χvals, hvals, ε=0.01)
    df = DataFrame()
    for (est, x, p, α, χ, h) ∈ Iterators.product(estimators, xvals, pvals, αvals, χvals, hvals)

    end
end

"""
Internal notes:
- In each experiment, we can keep the estimator, p, α, χ and h fixed, and randomly generate efforts x.
- The metric we care about is #rounds to Nash gap.


Experiment 1:
    DetMLE agents with random efforts between 0 and .25, p=1, α=1, χ=0.01, h=T.

Experiment 2:
    MLE agents with random efforts between 0 and .25, p=1, α=1, χ=0.01, h=T.

Experiment 3:
    Classic agents with random efforts between 0 and .25, p=1, α=1, χ=0.01, h=T.

Experiment 4:
    Dumb agents with random efforts between 0 and .25, p=1, α=1, χ=0.01, h=T.
"""

