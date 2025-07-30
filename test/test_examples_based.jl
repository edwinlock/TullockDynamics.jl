"""
Tests based on examples/ directory to ensure mathematical correctness
and catch regressions that break real usage scenarios.
"""

using Test
using TullockDynamics
using Random

@testset "Examples-Based Mathematical Correctness Tests" begin

    @testset "Homogeneous Bayesian Agents (Linear Cost)" begin
        # Based on examples/homogeneous-Bayesian-agents.jl
        cost(x) = x
        
        @testset "Small Contest Convergence" begin
            # Test with parameters similar to the example
            n = 3
            T = 100  # Shorter for testing
            χ = 0.05
            
            function generate_Bayesian_agent_contest(n::Int, T::Int; χ)
                agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
                initial_efforts = [rand() for _ in 1:n]
                return TullockContest(agents, initial_efforts, T)
            end
            
            Random.seed!(42)  # For reproducibility
            contest = generate_Bayesian_agent_contest(n, T; χ = χ)
            final_round = run!(contest)
            
            final_efforts_vals = final_efforts(contest)
            
            # For linear cost with symmetric agents, Nash equilibrium is x = 1/(n+1)
            expected_nash = 1.0/(n+1)  # = 0.25 for 3 agents
            
            # Test convergence properties
            @test all(effort -> effort > 0, final_efforts_vals)  # All efforts positive
            @test all(effort -> effort < 1.0, final_efforts_vals)  # All efforts reasonable
            
            # Test convergence toward Nash equilibrium (with some tolerance)
            for effort in final_efforts_vals
                @test abs(effort - expected_nash) < 0.1  # Within 10% of Nash
            end
            
            # Test Nash gap decreased from initial
            actual_final_round = min(final_round, num_rounds(contest))
            final_nash_gap = nash_gap(contest, actual_final_round)
            initial_nash_gap = nash_gap(contest, 1)
            @test final_nash_gap < initial_nash_gap  # Should improve
            @test final_nash_gap < 0.1  # Should be reasonably small
        end
        
        @testset "Bayesian Agent PDF Properties" begin
            # Test the Bayesian estimator returns valid PDFs
            n = 2
            T = 5
            χ = 0.05
            
            agents = [BayesianAgent(cost; χ=χ) for _ in 1:n]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, T)
            
            # Set up some history
            contest.efforts[1, 1:3] = [0.2, 0.4, 0.6]
            contest.efforts[2, 1:3] = [0.3, 0.1, 0.8]
            contest.winners[1, 1:3] = [false, true, false]  # 1 win, 2 losses
            
            # Get Bayesian estimator PDF
            pdf_func = bayesian_estimator(contest, 1, 1:3)
            
            # Test PDF properties
            lb = contest.workspace.min_other_efforts[1]
            ub = contest.workspace.max_other_efforts[1]
            
            @test pdf_func(lb) ≥ 0  # Non-negative at bounds
            @test pdf_func(ub) ≥ 0
            @test pdf_func((lb+ub)/2) ≥ 0  # Non-negative at midpoint
            
            # Test zero outside bounds
            @test pdf_func(lb - 0.1) == 0
            @test pdf_func(ub + 0.1) == 0
            
            # Test integration (approximately 1)
            using QuadGK
            integral, error = quadgk(pdf_func, lb, ub)
            @test integral ≈ 1.0 atol=1e-8
        end
    end
    
    @testset "Mixed Agent Types Contest" begin
        # Test contests with different agent types
        cost(x) = 0.5 * x^2
        
        @testset "MLE vs DetMLE vs Dumb Agents" begin
            agents = [
                MLEAgent(cost),
                DetMLEAgent(cost), 
                DumbAgent(cost)
            ]
            initial_efforts = [0.1, 0.15, 0.2]
            contest = TullockContest(agents, initial_efforts, 50)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # All agents should reach reasonable effort levels
            @test all(effort -> effort > 0, final_efforts_vals)
            @test all(effort -> effort < 2.0, final_efforts_vals)  # Reasonable upper bound
            
            # Contest should converge or at least improve
            actual_final_round = min(final_round, num_rounds(contest))
            final_nash_gap = nash_gap(contest, actual_final_round)
            initial_nash_gap = nash_gap(contest, 1)
            @test final_nash_gap ≤ initial_nash_gap  # Should not get worse
        end
        
        @testset "StandardAgent with Different Step Sizes" begin
            agents = [
                StandardAgent(cost, α=1.0),   # Full adaptation
                StandardAgent(cost, α=0.5),   # Moderate
                StandardAgent(cost, α=0.1)    # Conservative
            ]
            initial_efforts = [0.2, 0.2, 0.2]
            contest = TullockContest(agents, initial_efforts, 30)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # All should converge to reasonable values
            @test all(effort -> effort > 0, final_efforts_vals)
            @test all(effort -> effort < 1.5, final_efforts_vals)
            
            # Full adaptation agent should potentially reach different equilibrium
            # than conservative agents, but all should be stable
            @test all(isfinite, final_efforts_vals)
        end
    end
    
    @testset "Cost Function Variations" begin
        
        @testset "Linear Cost Function" begin
            cost(x) = x
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.1, 0.2]
            contest = TullockContest(agents, initial_efforts, 50)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # For 2 agents with linear cost, Nash equilibrium is x = 1/3 each
            expected_nash = 1.0/3.0
            
            for effort in final_efforts_vals
                @test abs(effort - expected_nash) < 0.15  # Within reasonable tolerance
            end
        end
        
        @testset "Quadratic Cost Function" begin
            cost(x) = 0.5 * x^2
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.1, 0.2]
            contest = TullockContest(agents, initial_efforts, 50)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # Should converge to reasonable values
            @test all(effort -> effort > 0, final_efforts_vals)
            @test all(effort -> effort < 1.0, final_efforts_vals)
            
            # Nash gap should improve
            actual_final_round = min(final_round, num_rounds(contest))
            final_nash_gap = nash_gap(contest, actual_final_round)
            initial_nash_gap = nash_gap(contest, 1)
            @test final_nash_gap ≤ initial_nash_gap
        end
        
        @testset "High Power Cost Function" begin
            cost(x) = 0.1 * x^3
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.05, 0.1]  # Lower initial efforts for high cost
            contest = TullockContest(agents, initial_efforts, 40)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # Should still converge reasonably
            @test all(effort -> effort > 0, final_efforts_vals) 
            @test all(effort -> effort < 2.0, final_efforts_vals)
            @test all(isfinite, final_efforts_vals)
        end
    end
    
    @testset "Edge Cases and Robustness" begin
        
        @testset "Very Small Minimum Efforts" begin
            cost(x) = x^2
            agents = [MLEAgent(cost, χ=1e-8), DetMLEAgent(cost, χ=1e-8)]
            initial_efforts = [1e-6, 2e-6]
            contest = TullockContest(agents, initial_efforts, 20)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # Should handle tiny values gracefully
            @test all(effort -> effort ≥ 1e-8, final_efforts_vals)  # Above minimum
            @test all(isfinite, final_efforts_vals)
            @test all(effort -> effort > 0, final_efforts_vals)
        end
        
        @testset "Asymmetric Agents" begin
            # Different cost functions and minimum efforts
            cost1(x) = x
            cost2(x) = 2*x^2
            cost3(x) = 0.5*x^1.5
            
            agents = [
                MLEAgent(cost1, χ=0.01),
                DetMLEAgent(cost2, χ=0.05),
                DumbAgent(cost3, χ=0.02)
            ]
            initial_efforts = [0.1, 0.2, 0.15]
            contest = TullockContest(agents, initial_efforts, 40)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # Should reach different but reasonable equilibria
            @test all(effort -> effort > 0, final_efforts_vals)
            @test all(isfinite, final_efforts_vals)
            
            # Different cost functions should lead to different effort levels
            @test length(unique(round.(final_efforts_vals, digits=2))) >= 2
        end
        
        @testset "Many Agents Contest" begin
            cost(x) = x^2
            n = 7  # Odd number of agents
            agents = [MLEAgent(cost) for _ in 1:n]
            initial_efforts = rand(n) * 0.2  # Random small initial efforts
            contest = TullockContest(agents, initial_efforts, 30)
            
            final_round = run!(contest)
            final_efforts_vals = final_efforts(contest)
            
            # Should handle many agents
            @test length(final_efforts_vals) == n
            @test all(effort -> effort > 0, final_efforts_vals)
            @test all(isfinite, final_efforts_vals)
            
            # For symmetric agents, efforts should be roughly similar
            effort_std = std(final_efforts_vals)
            effort_mean = mean(final_efforts_vals)
            @test effort_std / effort_mean < 0.5  # Coefficient of variation < 50%
        end
    end
    
    @testset "Workspace Consistency During Simulation" begin
        cost(x) = x^2
        agents = [MLEAgent(cost), DetMLEAgent(cost), BayesianAgent(cost)]
        initial_efforts = [0.1, 0.15, 0.2]
        contest = TullockContest(agents, initial_efforts, 20)
        
        # Run simulation and check workspace consistency at each round
        for t in 1:min(10, num_rounds(contest))
            step!(contest, t)
            ws = contest.workspace
            
            # Check workspace matches contest state
            @test ws.total_efforts[t] ≈ sum(contest.efforts[:, t])
            
            for i in 1:3
                expected_other = ws.total_efforts[t] - contest.efforts[i, t] 
                @test ws.other_efforts[i, t] ≈ expected_other
            end
            
            # Check bounds are still correct
            expected_min_total = sum(agent.χ for agent in contest.agents)
            expected_max_total = sum(agent.max_effort for agent in contest.agents)
            @test ws.min_total_efforts ≈ expected_min_total
            @test ws.max_total_efforts ≈ expected_max_total
        end
    end
    
    @testset "Estimator Mathematical Properties" begin
        
        @testset "MLE Estimator Equation Solving" begin
            # Test that MLE estimator actually solves the MLE equation
            cost(x) = x^2
            agents = [MLEAgent(cost), MLEAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 10)
            
            # Set up known scenario
            contest.efforts[1, 1:4] = [0.2, 0.4, 0.6, 0.8]
            contest.winners[1, 1:4] = [true, false, true, false]  # 2 wins out of 4
            
            estimate = max_likelihood_estimator(contest, 1, 1:4)
            
            # Verify the MLE equation: sum(x_i / (x_i + y)) = wins
            x_vals = [0.2, 0.4, 0.6, 0.8]
            wins = 2.0
            mle_sum = sum(x / (x + estimate) for x in x_vals)
            
            @test abs(mle_sum - wins) < 1e-10  # Should solve equation precisely
        end
        
        @testset "Dumb Estimator Formula" begin
            # Test dumb estimator follows its heuristic formula
            cost(x) = x
            agents = [DumbAgent(cost), DumbAgent(cost)]
            initial_efforts = [0.3, 0.4]
            contest = TullockContest(agents, initial_efforts, 8)
            
            # Set up scenario: 3 wins out of 5 rounds
            contest.efforts[1, 1:5] = [0.3, 0.4, 0.5, 0.6, 0.7]
            contest.winners[1, 1:5] = [true, false, true, true, false]  # 3 wins
            
            estimate = dumb_estimator(contest, 1, 1:5)
            
            # Manual calculation: (rounds/wins - 1) * avg_effort
            rounds = 5
            wins = 3
            avg_effort = (0.3 + 0.4 + 0.5 + 0.6 + 0.7) / 5  # = 0.5
            expected = (rounds/wins - 1) * avg_effort  # = (5/3 - 1) * 0.5 = (2/3) * 0.5 = 1/3
            
            @test estimate ≈ expected atol=1e-12
        end
    end

    @testset "Performance and Numerical Stability" begin
        
        @testset "No NaN or Inf Values" begin
            # Test various scenarios don't produce invalid values
            cost(x) = x^2
            
            test_scenarios = [
                ([MLEAgent(cost), DetMLEAgent(cost)], [0.001, 0.002]),
                ([BayesianAgent(cost), DumbAgent(cost)], [0.1, 0.15]),
                ([StandardAgent(cost), MLEAgent(cost)], [0.05, 0.08])
            ]
            
            for (agents, initial) in test_scenarios
                contest = TullockContest(agents, initial, 15)
                final_round = run!(contest)
                
                # Check no NaN/Inf in final results
                final_efforts_vals = final_efforts(contest)
                @test all(isfinite, final_efforts_vals)
                
                # Check utilities and nash gaps
                actual_final = min(final_round, num_rounds(contest))
                for t in 1:actual_final
                    @test all(isfinite, contest.utilities[:, t])
                    @test all(isfinite, contest.nash_gaps[:, t])
                end
            end
        end
        
        @testset "Convergence Rate Reasonable" begin
            # Test that convergence doesn't take unreasonably long
            cost(x) = x
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 200)
            
            final_round = run!(contest, ε=0.01)  # Convergence threshold
            
            # Should converge in reasonable time for simple linear case
            @test final_round ≤ 150  # Should not take full 200 rounds
        end
    end
end