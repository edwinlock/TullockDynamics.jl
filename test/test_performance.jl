@testset "Performance and Benchmarking Tests" begin
    
    @testset "Allocation efficiency" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2], 10)
        
        # Warm up
        run!(contest)
        
        # Create fresh contest for measurement
        contest_fresh = TullockContest(agents, [0.1, 0.2], 10)
        
        # Measure allocations for single step
        allocs_step = @allocated begin
            TullockDynamics.step!(contest_fresh, 1)
        end
        
        # Should be minimal allocations due to workspace optimization
        @test allocs_step < 20000  # Reasonable threshold for complex operations
        
        # Measure allocations for full run
        contest_alloc = TullockContest(agents, [0.1, 0.2], 5)
        allocs_run = @allocated begin
            run!(contest_alloc)
        end
        
        # Should scale reasonably with contest size
        @test allocs_run < 100000  # Reasonable threshold for full simulation
    end
    
    @testset "Scalability tests" begin
        cost(x) = x
        
        # Test scaling with number of agents
        agent_counts = [2, 5, 10, 20]
        times = Float64[]
        
        for n in agent_counts
            agents = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:n]
            contest = TullockContest(agents, rand(n) .* 0.2, 10)
            
            elapsed = @elapsed run!(contest)
            push!(times, elapsed)
            
            # Should complete in reasonable time
            @test elapsed < 2.0  # 2 second limit per run
        end
        
        # Performance should scale reasonably (not exponentially)
        # Allow for some variation but check general trend
        @test times[end] < times[1] * 50  # At most 50x slower for 10x more agents
    end
    
    @testset "Memory usage consistency" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Run multiple contests and check memory stays bounded
        for i in 1:10
            contest = TullockContest(agents, [0.1, 0.2], 15)
            final_round = run!(contest)
            
            converged, actual_final = convergence_status(contest, final_round)
            @test actual_final >= 1
            
            # Check workspace is reused properly
            @test length(contest.workspace.own_efforts) == 2
            @test contest.workspace.weights_obj isa StatsBase.Weights
        end
        
        # Force garbage collection to ensure no memory leaks
        GC.gc()
    end
    
    @testset "Estimator performance comparison" begin
        cost(x) = x
        agents = [MLEAgent(cost), DetMLEAgent(cost), DumbAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2, 0.15], 10)
        
        # Set up realistic data
        for t in 2:5
            contest.efforts[:, t] = rand(3) .* 0.3 .+ 0.1
            winner_idx = argmax(contest.efforts[:, t])
            contest.winners[winner_idx, t] = true
        end
        
        window = 2:5
        
        # Benchmark each estimator
        estimators = [
            ("MLE", max_likelihood_estimator),
            ("DetMLE", deterministic_max_likelihood_estimator),
            ("Dumb", dumb_estimator),
            ("Classic", classic_estimator)
        ]
        
        for (name, estimator_func) in estimators
            elapsed = @elapsed begin
                for agent_idx in 1:3
                    for _ in 1:100  # Multiple calls to get meaningful timing
                        estimator_func(contest, agent_idx, window)
                    end
                end
            end
            
            # Each estimator should be reasonably fast
            @test elapsed < 1.0
        end
        
        # Bayesian estimator is expected to be slower due to integration
        elapsed_bayesian = @elapsed begin
            for agent_idx in 1:3
                for _ in 1:10  # Fewer iterations due to expected slower performance
                    bayesian_estimator(contest, agent_idx, window)
                end
            end
        end
        
        @test elapsed_bayesian < 5.0  # More generous limit for Bayesian
    end
    
    @testset "Large contest performance" begin
        cost(x) = x
        
        # Large number of agents (stress test)
        n_agents = 50
        agents = [rand() < 0.5 ? MLEAgent(cost) : DumbAgent(cost) for _ in 1:n_agents]
        contest_large = TullockContest(agents, rand(n_agents) .* 0.2, 8)  # Shorter duration for large contest
        
        elapsed_large = @elapsed begin
            final_round = run!(contest_large)
        end
        
        @test elapsed_large < 10.0  # Should complete within 10 seconds
        
        converged, actual_final = convergence_status(contest_large, final_round)
        @test actual_final >= 1
        
        # Verify data integrity for large contest
        for t in 1:actual_final
            @test sum(contest_large.winners[:, t]) == 1  # Exactly one winner
            @test all(isfinite, contest_large.efforts[:, t])
            @test all(contest_large.efforts[:, t] .>= 0.0)
        end
        
        # Check workspace scales correctly
        @test length(contest_large.workspace.own_efforts) == n_agents
        @test length(contest_large.workspace.min_other_bounds) == n_agents
        @test length(contest_large.workspace.max_other_bounds) == n_agents
    end
    
    @testset "Repeated simulation performance" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost)]
        
        # Measure performance of repeated simulations
        n_simulations = 20
        times = Float64[]
        
        for i in 1:n_simulations
            contest = TullockContest(agents, [0.1, 0.2], 15)
            elapsed = @elapsed run!(contest)
            push!(times, elapsed)
        end
        
        # Performance should be consistent
        mean_time = sum(times) / length(times)
        std_time = sqrt(sum((t - mean_time)^2 for t in times) / length(times))
        
        @test mean_time < 0.5  # Average under 0.5 seconds
        @test std_time < mean_time  # Standard deviation not too high
        
        # No major outliers (all within 5 standard deviations, relaxed for CI)
        @test all(abs(t - mean_time) < 5 * std_time for t in times)
    end
    
    @testset "Workspace reuse verification" begin
        cost(x) = x
        agents = [MLEAgent(cost), DumbAgent(cost), DetMLEAgent(cost)]
        contest = TullockContest(agents, [0.1, 0.2, 0.15], 5)
        
        # Get references to workspace objects
        workspace_ref = contest.workspace
        own_efforts_ref = contest.workspace.own_efforts
        weights_ref = contest.workspace.weights_obj
        
        # Run simulation
        run!(contest)
        
        # Verify same objects are still referenced (no reallocation)
        @test contest.workspace === workspace_ref
        @test contest.workspace.own_efforts === own_efforts_ref
        @test contest.workspace.weights_obj === weights_ref
        
        # Verify workspace data is updated
        @test contest.workspace.total_effort >= 0.0
        @test sum(contest.workspace.own_efforts) ≈ contest.workspace.total_effort
    end
    
    @testset "Convergence efficiency" begin
        cost(x) = x
        
        # Test convergence with different epsilon values
        epsilons = [0.1, 0.01, 0.001]
        agents = [MLEAgent(cost), MLEAgent(cost)]  # Symmetric agents more likely to converge
        
        for ε in epsilons
            contest = TullockContest(agents, [0.2, 0.2], 50)  # Symmetric initial conditions
            
            elapsed = @elapsed begin
                final_round = run!(contest; ε=ε)
            end
            
            converged, actual_final = convergence_status(contest, final_round)
            
            if converged
                total_gap = nash_gap(contest, actual_final)
                @test total_gap <= ε
                @test actual_final < 50  # Should converge before hitting limit
            end
            
            # Should not take excessively long even if not converging
            @test elapsed < 2.0
        end
    end
end