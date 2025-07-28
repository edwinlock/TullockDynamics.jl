@testset "Agents Tests" begin
    
    @testset "Agent construction" begin
        cost(x) = x^2
        estimator = max_likelihood_estimator
        p(t) = 0.5
        α(t) = 0.8
        χ = 0.05
        h(t) = max(1, t-5):t-1
        
        agent = Agent(estimator, cost, p, α, χ, h)
        
        @test agent.estimator == estimator
        @test agent.cost(2.0) == 4.0
        @test agent.p(10) == 0.5
        @test agent.α(5) == 0.8
        @test agent.χ == 0.05
        @test agent.h(6) == 1:5
        @test agent.max_effort == max_agent_effort(cost)
    end
    
    @testset "MLEAgent constructor" begin
        cost(x) = 0.5 * x^2
        
        # Test default parameters
        agent = MLEAgent(cost)
        @test agent.cost(2.0) == 2.0
        @test agent.χ == 0.01
        @test agent.p(10) == 1.0
        @test agent.α(10) == 1.0
        @test agent.h(5) == 1:4
        @test agent.estimator == max_likelihood_estimator
        
        # Test custom parameters
        agent = MLEAgent(cost; χ=0.02)
        @test agent.χ == 0.02
    end
    
    @testset "DetMLEAgent constructor" begin
        cost(x) = x
        
        agent = DetMLEAgent(cost)
        @test agent.cost(3.0) == 3.0
        @test agent.χ == 0.01
        @test agent.estimator == deterministic_max_likelihood_estimator
        
        # Test custom parameters
        agent = DetMLEAgent(cost; χ=0.005)
        @test agent.χ == 0.005
    end
    
    @testset "DumbAgent constructor" begin
        cost(x) = x^2
        
        agent = DumbAgent(cost)
        @test agent.cost(2.0) == 4.0
        @test agent.χ == 0.01
        @test agent.estimator == dumb_estimator
        
        # Test with custom parameters
        agent = DumbAgent(cost; χ=0.1)
        @test agent.χ == 0.1
    end
    
    @testset "BayesianAgent constructor" begin
        cost(x) = 0.1 * x^3
        
        agent = BayesianAgent(cost)
        @test agent.cost(2.0) == 0.8
        @test agent.χ == 0.01
        @test agent.estimator == bayesian_estimator
        
        # Test with custom parameters
        agent = BayesianAgent(cost; χ=0.02)
        @test agent.χ == 0.02
    end
    
    @testset "utility function" begin
        cost(x) = x
        agent = MLEAgent(cost)
        
        # Basic utility test: u = x/(x+s) - c(x)
        x, s = 2.0, 3.0
        expected = x / (x + s) - cost(x)  # 2/5 - 2 = -1.6
        @test utility(agent, x, s) ≈ expected
        
        # Edge case: s = 0 (no opponent effort)
        x, s = 1.0, 0.0
        expected = x / (x + s) - cost(x)  # 1/1 - 1 = 0
        @test utility(agent, x, s) ≈ expected
        
        # Edge case: x = 0 (no own effort)
        x, s = 0.0, 2.0
        expected = x / (x + s) - cost(x)  # 0/2 - 0 = 0
        @test utility(agent, x, s) ≈ expected
    end
    
    @testset "best_response function" begin
        cost(x) = x^2
        agent = MLEAgent(cost)
        
        # Test best response computation
        opponent_effort = 1.0
        br = best_response(agent, opponent_effort)
        
        # Best response should be positive
        @test br >= agent.χ
        @test br < 10.0  # Should be reasonable
        
        # Test with bounds
        min_other = 0.5
        max_other = 2.0
        br_bounded = best_response(agent, opponent_effort; 
                                  min_other_efforts=min_other, 
                                  max_other_efforts=max_other)
        @test br_bounded >= agent.χ
        
        # Test edge case: zero opponent effort
        br_zero = best_response(agent, 0.0)
        @test br_zero >= agent.χ
    end
    
    @testset "nash_gap function" begin
        cost(x) = x
        agent = MLEAgent(cost)
        
        # Test Nash gap for best response (should be ~0)
        s = 2.0
        br = best_response(agent, s)
        gap = nash_gap(agent, br, s)
        @test gap >= 0.0
        @test gap < 1e-4  # Should be very small for best response
        
        # Test Nash gap for non-optimal choice
        x_suboptimal = 0.1
        gap = nash_gap(agent, x_suboptimal, s)
        @test gap >= 0.0
        
        # Nash gap should be larger for suboptimal choices
        @test gap > nash_gap(agent, br, s)
    end
    
    @testset "max_agent_effort with different costs" begin
        # Linear cost
        agent1 = MLEAgent(x -> x)
        max_eff1 = max_agent_effort(agent1.cost)
        @test max_eff1 ≈ 1.0 atol=1e-6
        
        # Quadratic cost
        agent2 = MLEAgent(x -> 0.5 * x^2)
        max_eff2 = max_agent_effort(agent2.cost)
        @test max_eff2 ≈ sqrt(2.0) atol=1e-6
        
        # Exponential cost (should find reasonable max)
        agent3 = MLEAgent(x -> exp(x) - 1)
        max_eff3 = max_agent_effort(agent3.cost)
        @test max_eff3 > 0.0
        @test agent3.cost(max_eff3) ≈ 1.0 atol=1e-6
    end
end