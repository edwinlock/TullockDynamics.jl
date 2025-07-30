"""
Comprehensive mathematical correctness tests for TullockDynamics.jl

These tests focus on algorithmic correctness and mathematical properties
rather than just code execution, addressing the critical gap that allowed
convergence issues to go undetected.
"""

using Test
using TullockDynamics
using Random
using StatsBase

# Set seed for reproducible tests
Random.seed!(12345)

@testset "Mathematical Correctness Tests" begin

    @testset "Agent Construction and Type Stability" begin
        
        @testset "Constructor Functions Type Stability" begin
            cost(x) = x^2
            
            # Test all agent constructors
            agents = [
                MLEAgent(cost),
                DetMLEAgent(cost), 
                DumbAgent(cost),
                StandardAgent(cost),
                BayesianAgent(cost)
            ]
            
            for agent in agents
                # Test that functions return correct types
                @test agent.p(5) isa Float64
                @test agent.α(5) isa Float64
                @test agent.h(5) isa UnitRange{Int64}
                
                # Test mathematical properties
                @test 0 ≤ agent.p(5) ≤ 1  # probability bounds
                @test 0 ≤ agent.α(5) ≤ 1  # step size bounds
                @test agent.χ ≥ 0  # minimum effort non-negative
                @test agent.max_effort > agent.χ  # max > min effort
                
                # Test history window properties
                h_result = agent.h(10)
                @test h_result == 1:9  # should be full history
                @test length(h_result) == 9
            end
        end
        
        @testset "Cost Function Validation" begin
            # Test various cost functions for mathematical properties
            cost_functions = [
                x -> x,           # linear
                x -> x^2,         # quadratic
                x -> 0.5*x^1.5,   # fractional power
                x -> x^3,         # cubic
            ]
            
            for cost in cost_functions
                agent = MLEAgent(cost)
                
                # Test cost function properties
                @test cost(0) ≥ 0  # non-negative at zero
                @test cost(1) > 0  # positive for positive input
                @test cost(2) > cost(1)  # increasing (assuming convex)
                
                # Test max effort calculation
                max_eff = max_agent_effort(cost)
                @test max_eff > 0
                @test cost(max_eff) ≈ 1.0 atol=1e-10  # cost equals reward at max
            end
        end
    end
    
    @testset "Utility Function Mathematical Properties" begin
        cost(x) = x^2
        agent = MLEAgent(cost)
        
        @testset "Utility Function Correctness" begin
            # Test utility = winning_prob - cost
            x, s = 0.5, 1.0
            expected_utility = x/(x+s) - cost(x)
            @test utility(agent, x, s) ≈ expected_utility
            
            # Test boundary conditions
            @test utility(agent, 0, 1) == 0 - cost(0)  # zero effort
            @test utility(agent, 1, 0) == 1 - cost(1)  # opponent zero effort
            
            # Test monotonicity properties
            s_fixed = 1.0
            x1, x2 = 0.1, 0.2
            # For small efforts, increasing x should increase utility initially
            if cost(x2) - cost(x1) < x2/(x2+s_fixed) - x1/(x1+s_fixed)
                @test utility(agent, x2, s_fixed) > utility(agent, x1, s_fixed)
            end
        end
        
        @testset "Nash Gap Correctness" begin
            x, s = 0.5, 1.0
            gap = nash_gap(agent, x, s)
            
            # Nash gap should be non-negative
            @test gap ≥ 0
            
            # Nash gap should be zero at best response
            br = best_response(agent, s)
            nash_gap_at_br = nash_gap(agent, br, s)
            @test nash_gap_at_br ≈ 0 atol=1e-10
            
            # Nash gap should equal difference in utilities
            br_utility = utility(agent, br, s)
            current_utility = utility(agent, x, s)
            @test gap ≈ br_utility - current_utility atol=1e-12
        end
    end
    
    @testset "Best Response Function Correctness" begin
        
        @testset "Scalar Best Response" begin
            cost(x) = x^2
            agent = MLEAgent(cost, χ=0.01)
            
            # Test mathematical properties
            for s in [0.1, 0.5, 1.0, 2.0]
                br = best_response(agent, s)
                
                # Best response should be ≥ minimum effort
                @test br ≥ agent.χ
                
                # Best response should satisfy first-order condition
                # For utility u(x,s) = x/(x+s) - cost(x)
                # FOC: s/(x+s)^2 - cost'(x) = 0
                ε = 1e-8
                u_left = utility(agent, br - ε, s)
                u_right = utility(agent, br + ε, s)
                
                # At optimum, utility should be maximized (or at boundary)
                if br > agent.χ + 1e-10  # Not at boundary
                    @test u_left ≤ utility(agent, br, s) + 1e-10
                    @test u_right ≤ utility(agent, br, s) + 1e-10
                end
            end
        end
        
        @testset "PDF Best Response" begin
            cost(x) = x^2
            agent = BayesianAgent(cost, χ=0.01)
            
            # Create a simple uniform PDF for testing
            lb, ub = 0.5, 1.5
            uniform_pdf(y) = (lb ≤ y ≤ ub) ? 1/(ub-lb) : 0.0
            
            br = best_response(agent, uniform_pdf; 
                             min_other_efforts=lb, max_other_efforts=ub)
            
            # Best response should be ≥ minimum effort
            @test br ≥ agent.χ
            
            # For uniform distribution, should match expected value calculation
            expected_s = (lb + ub) / 2
            br_scalar = best_response(agent, expected_s)
            
            # Should be reasonably close (not exact due to Jensen's inequality)
            @test abs(br - br_scalar) < 0.5
        end
    end
    
    @testset "Estimator Mathematical Correctness" begin
        
        @testset "MLE Estimator Correctness" begin
            # Create a known scenario and verify MLE estimation
            cost(x) = x^2
            agents = [MLEAgent(cost), MLEAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 10)
            
            # Manually set up a scenario with known outcomes
            # Agent 1 efforts: [0.2, 0.4, 0.6]
            # Agent 1 wins: [false, true, false] -> 1 win
            contest.efforts[1, 1:3] = [0.2, 0.4, 0.6]
            contest.efforts[2, 1:3] = [0.3, 0.1, 0.9]  # opponent efforts
            contest.winners[1, 1:3] = [false, true, false]
            
            # Test MLE estimation
            estimate = max_likelihood_estimator(contest, 1, 1:3)
            
            # Verify by solving the MLE equation manually
            # Sum(x_i / (x_i + y)) = wins, where x_i = [0.2, 0.4, 0.6], wins = 1
            # 0.2/(0.2+y) + 0.4/(0.4+y) + 0.6/(0.6+y) = 1
            
            x_vals = [0.2, 0.4, 0.6]
            f(y) = sum(x/(x+y) for x in x_vals) - 1.0
            
            @test abs(f(estimate)) < 1e-10  # Should solve the equation
            @test estimate > 0  # Should be positive
        end
        
        @testset "Deterministic MLE Correctness" begin
            cost(x) = x^2
            agents = [DetMLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 5)
            
            # Set up known total efforts
            contest.efforts[1, 1:3] = [0.2, 0.4, 0.6]
            contest.efforts[2, 1:3] = [0.3, 0.1, 0.9]
            
            estimate = deterministic_max_likelihood_estimator(contest, 1, 1:3)
            
            # Calculate expected wins manually
            expected_wins = 0.2/0.5 + 0.4/0.5 + 0.6/1.5  # = 0.4 + 0.8 + 0.4 = 1.6
            
            # Verify MLE equation: sum(x_i / (x_i + y)) = expected_wins
            x_vals = [0.2, 0.4, 0.6]
            f(y) = sum(x/(x+y) for x in x_vals) - expected_wins
            
            @test abs(f(estimate)) < 1e-10
        end
        
        @testset "Dumb Estimator Correctness" begin
            cost(x) = x^2
            agents = [DumbAgent(cost), DumbAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 5)
            
            # Set up scenario: 2 wins out of 4 rounds
            contest.efforts[1, 1:4] = [0.2, 0.4, 0.6, 0.8]
            contest.winners[1, 1:4] = [true, false, true, false]
            
            estimate = dumb_estimator(contest, 1, 1:4)
            
            # Manual calculation: (rounds/wins - 1) * avg_effort
            # = (4/2 - 1) * (0.2+0.4+0.6+0.8)/4 = 1 * 0.5 = 0.5
            expected = (4/2 - 1) * 0.5
            @test estimate ≈ expected
        end
        
        @testset "Classic Estimator Correctness" begin
            cost(x) = x^2
            agents = [StandardAgent(cost), StandardAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 5)
            
            # Set final round efforts
            contest.efforts[1, 4] = 0.6
            contest.efforts[2, 4] = 0.4
            
            estimate = classic_estimator(contest, 1, 2:4)
            
            # Should return other agent's effort in last round
            expected = 0.4  # agent 2's effort in round 4
            @test estimate ≈ expected
        end
    end
    
    @testset "Bayesian Estimator Correctness" begin
        cost(x) = x^2
        agents = [BayesianAgent(cost, χ=0.1), BayesianAgent(cost, χ=0.1)]
        initial_efforts = [0.2, 0.3]
        contest = TullockContest(agents, initial_efforts, 5)
        
        # Set up scenario
        contest.efforts[1, 1:3] = [0.2, 0.4, 0.6]
        contest.winners[1, 1:3] = [false, true, false]  # 1 win, 2 losses
        
        # Get the PDF estimator
        pdf_estimator = bayesian_estimator(contest, 1, 1:3)
        
        @testset "PDF Properties" begin
            lb = contest.workspace.min_other_efforts[1]
            ub = contest.workspace.max_other_efforts[1]
            
            # Test PDF properties
            @test pdf_estimator(lb) ≥ 0
            @test pdf_estimator(ub) ≥ 0
            @test pdf_estimator((lb+ub)/2) ≥ 0
            
            # PDF should be zero outside bounds
            @test pdf_estimator(lb - 0.1) == 0
            @test pdf_estimator(ub + 0.1) == 0
            
            # Test normalization (approximately)
            using QuadGK
            integral, error = quadgk(pdf_estimator, lb, ub)
            @test integral ≈ 1.0 atol=1e-10
        end
        
        @testset "Bayesian Update Correctness" begin
            # Compare with theoretical expectation
            # More losses should shift distribution toward higher opponent efforts
            
            # Create scenario with all losses
            contest_all_losses = TullockContest(agents, initial_efforts, 5)
            contest_all_losses.efforts[1, 1:3] = [0.2, 0.4, 0.6]
            contest_all_losses.winners[1, 1:3] = [false, false, false]  # all losses
            
            pdf_all_losses = bayesian_estimator(contest_all_losses, 1, 1:3)
            
            # Create scenario with all wins  
            contest_all_wins = TullockContest(agents, initial_efforts, 5)
            contest_all_wins.efforts[1, 1:3] = [0.2, 0.4, 0.6]
            contest_all_wins.winners[1, 1:3] = [true, true, true]  # all wins
            
            pdf_all_wins = bayesian_estimator(contest_all_wins, 1, 1:3)
            
            # Expected value with all losses should be higher than with all wins
            lb = contest.workspace.min_other_efforts[1]
            ub = contest.workspace.max_other_efforts[1]
            
            # Calculate expected values
            expected_losses, error1 = quadgk(y -> y * pdf_all_losses(y), lb, ub)
            expected_wins, error2 = quadgk(y -> y * pdf_all_wins(y), lb, ub)
            
            @test expected_losses > expected_wins
        end
    end
    
    @testset "Contest Dynamics Correctness" begin
        
        @testset "Single Round Correctness" begin
            cost(x) = x
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.3, 0.7]
            contest = TullockContest(agents, initial_efforts, 10)
            
            # Test first round
            step!(contest, 1)
            
            # Check utilities
            expected_u1 = 0.3/1.0 - cost(0.3)  # = 0.3 - 0.3 = 0
            expected_u2 = 0.7/1.0 - cost(0.7)  # = 0.7 - 0.7 = 0
            @test contest.utilities[1,1] ≈ expected_u1
            @test contest.utilities[2,1] ≈ expected_u2
            
            # Check nash gaps
            br1 = best_response(agents[1], 0.7)  # other agent's effort
            br2 = best_response(agents[2], 0.3)
            
            expected_gap1 = utility(agents[1], br1, 0.7) - contest.utilities[1,1]
            expected_gap2 = utility(agents[2], br2, 0.3) - contest.utilities[2,1]
            
            @test contest.nash_gaps[1,1] ≈ expected_gap1 atol=1e-12
            @test contest.nash_gaps[2,1] ≈ expected_gap2 atol=1e-12
            
            # Check winner selection (exactly one winner)
            @test sum(contest.winners[:,1]) == 1
        end
        
        @testset "Workspace Consistency" begin
            cost(x) = x^2
            agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]
            initial_efforts = [0.2, 0.3, 0.4]
            contest = TullockContest(agents, initial_efforts, 5)
            
            # Run a few steps
            for t in 1:3
                step!(contest, t)
                
                # Verify workspace consistency
                ws = contest.workspace
                
                # Check total_efforts matches sum of contest.efforts
                expected_total = sum(contest.efforts[:, t])
                @test ws.total_efforts[t] ≈ expected_total
                
                # Check other_efforts  
                for i in 1:3
                    expected_other = ws.total_efforts[t] - contest.efforts[i, t]
                    @test ws.other_efforts[i, t] ≈ expected_other
                end
                
                # Check pre-computed bounds
                expected_min_total = sum(agent.χ for agent in contest.agents)
                expected_max_total = sum(agent.max_effort for agent in contest.agents)
                @test ws.min_total_efforts ≈ expected_min_total
                @test ws.max_total_efforts ≈ expected_max_total
                
                for i in 1:3
                    @test ws.min_other_efforts[i] ≈ expected_min_total - contest.agents[i].χ
                    @test ws.max_other_efforts[i] ≈ expected_max_total - contest.agents[i].max_effort
                end
            end
        end
    end
    
    @testset "Convergence Properties" begin
        
        @testset "Linear Cost Nash Equilibrium" begin
            # Test convergence to known Nash equilibrium for linear cost
            cost(x) = x
            
            # For symmetric agents with cost(x) = x, Nash equilibrium is x = 1/(n+1)
            n = 3
            expected_nash = 1.0/(n+1)  # = 0.25 for 3 agents
            
            @testset "MLE Agents Convergence" begin
                agents = [MLEAgent(cost) for _ in 1:n]
                initial_efforts = [0.1, 0.15, 0.2]
                contest = TullockContest(agents, initial_efforts, 100)
                
                final_round = run!(contest)
                final_efforts_val = final_efforts(contest)
                
                # Should converge close to Nash equilibrium
                for effort in final_efforts_val
                    @test abs(effort - expected_nash) < 0.05  # Within 5% of Nash
                end
                
                # Total Nash gap should be small
                actual_final = min(final_round, num_rounds(contest))
                total_nash_gap = nash_gap(contest, actual_final)
                @test total_nash_gap < 0.01
            end
            
            @testset "Bayesian Agents Convergence" begin
                agents = [BayesianAgent(cost) for _ in 1:n]
                initial_efforts = [0.1, 0.15, 0.2]
                contest = TullockContest(agents, initial_efforts, 200)  # More rounds for Bayesian
                
                final_round = run!(contest)
                final_efforts_val = final_efforts(contest)
                
                # Bayesian agents should also converge to Nash
                for effort in final_efforts_val
                    @test abs(effort - expected_nash) < 0.08  # Slightly more tolerance
                end
            end
        end
        
        @testset "Quadratic Cost Convergence" begin
            # Test convergence for quadratic cost function
            cost(x) = 0.5 * x^2
            
            agents = [MLEAgent(cost), DetMLEAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 100)
            
            final_round = run!(contest)
            
            # Should reach reasonable equilibrium (not testing exact values due to complexity)
            final_efforts_val = final_efforts(contest)
            
            # Efforts should be positive and reasonable
            for effort in final_efforts_val
                @test effort > 0
                @test effort < 2.0  # Should not be unreasonably high
            end
            
            # Nash gap should decrease
            actual_final = min(final_round, num_rounds(contest))
            total_nash_gap = nash_gap(contest, actual_final)
            @test total_nash_gap < nash_gap(contest, 1)  # Should improve from initial
        end
    end
    
    @testset "Numerical Stability Tests" begin
        
        @testset "Edge Case Handling" begin
            cost(x) = x^2
            
            # Test with very small minimum efforts
            agent_small = MLEAgent(cost, χ=1e-10)
            @test agent_small.χ == 1e-10
            
            # Test with efforts near zero
            s_tiny = 1e-12
            br = best_response(agent_small, s_tiny)
            @test br ≥ agent_small.χ
            @test isfinite(br)
            
            # Test utility calculation with tiny values
            u = utility(agent_small, 1e-10, 1e-10)
            @test isfinite(u)
        end
        
        @testset "Large Value Handling" begin
            cost(x) = 0.001 * x^2  # Very cheap cost
            agent_cheap = MLEAgent(cost)
            
            # Should handle large efforts
            large_s = 100.0
            br = best_response(agent_cheap, large_s)
            @test isfinite(br)
            @test br > 0
        end
        
        @testset "Root Finding Accuracy" begin
            # Test find_root function with known roots
            
            # Linear function: f(x) = 2 - x, root at x = 2
            f1(x) = 2.0 - x
            root1 = find_root(f1, 0.0)
            @test abs(root1 - 2.0) < 1e-10
            
            # Quadratic: f(x) = 4 - x^2, root at x = 2
            f2(x) = 4.0 - x^2
            root2 = find_root(f2, 0.0)
            @test abs(root2 - 2.0) < 1e-10
            
            # More complex: f(x) = 1/(1+x) - 0.25, root at x = 3
            f3(x) = 1.0/(1.0 + x) - 0.25
            root3 = find_root(f3, 0.0)
            @test abs(root3 - 3.0) < 1e-10
        end
    end
    
    @testset "Cache Correctness" begin
        
        @testset "Best Response Cache" begin
            cost(x) = x^2
            agent = MLEAgent(cost)
            
            # Clear cache first
            clear_best_response_cache!()
            
            # First call should compute and cache
            br1 = best_response(agent, 1.0)
            
            # Second call with same parameters should return same result
            br2 = best_response(agent, 1.0)
            @test br1 ≈ br2
            
            # Different parameters should give different results
            br3 = best_response(agent, 2.0)
            @test abs(br3 - br1) > 1e-10  # Should be different
        end
        
        @testset "Bayesian Cache" begin
            cost(x) = x^2
            agents = [BayesianAgent(cost), BayesianAgent(cost)]
            initial_efforts = [0.2, 0.3]
            contest = TullockContest(agents, initial_efforts, 5)
            
            # Set up scenario
            contest.efforts[1, 1:2] = [0.2, 0.4]
            contest.winners[1, 1:2] = [false, true]
            
            clear_bayesian_cache!()
            
            # First call
            pdf1 = bayesian_estimator(contest, 1, 1:2)
            val1 = pdf1(0.5)
            
            # Second call should use cache and return same PDF
            pdf2 = bayesian_estimator(contest, 1, 1:2)
            val2 = pdf2(0.5)
            @test val1 ≈ val2
        end
    end
end