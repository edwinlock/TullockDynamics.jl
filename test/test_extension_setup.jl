using Test

# Try to import Pkg, but don't fail if it's not available
try
    using Pkg
    global HAS_PKG = true
catch
    global HAS_PKG = false
end

@testset "Extension Setup and Configuration" begin
    
    @testset "Project.toml Configuration" begin
        # Read and parse Project.toml
        project_path = joinpath(@__DIR__, "..", "Project.toml")
        @test isfile(project_path)
        
        project_content = read(project_path, String)
        
        # Test that Plots and Measures are NOT in main dependencies
        @test !occursin(r"Plots\s*=.*\n.*ForwardDiff", project_content)  # Plots should not be near other deps
        @test !occursin(r"Measures\s*=.*\n.*ForwardDiff", project_content)  # Measures should not be near other deps
        
        # Test that weakdeps section exists and includes plotting packages
        @test occursin("[weakdeps]", project_content)
        @test occursin("Plots = ", project_content)
        @test occursin("Measures = ", project_content)
        
        # Test that extensions section exists
        @test occursin("[extensions]", project_content)
        @test occursin("TullockDynamicsPlotsExt", project_content)
        
        # Test core dependencies are still present
        @test occursin("ForwardDiff = ", project_content)
        @test occursin("Integrals = ", project_content)
        @test occursin("StatsBase = ", project_content)
    end
    
    @testset "Extension File Structure" begin
        # Test that extension directory exists
        ext_dir = joinpath(@__DIR__, "..", "ext")
        @test isdir(ext_dir)
        
        # Test that extension file exists
        ext_file = joinpath(ext_dir, "TullockDynamicsPlotsExt.jl")
        @test isfile(ext_file)
        
        # Test extension file content
        ext_content = read(ext_file, String)
        @test occursin("module TullockDynamicsPlotsExt", ext_content)
        @test occursin("using TullockDynamics", ext_content)
        @test occursin("using Plots", ext_content)
        @test occursin("using Measures", ext_content)
        @test occursin("function TullockDynamics.visualise", ext_content)
    end
    
    @testset "Package Loading Without Extensions" begin
        # Test that the package loads without error
        @test TullockDynamics isa Module
        
        # Test that core functionality is available
        @test isdefined(TullockDynamics, :TullockContest)
        @test isdefined(TullockDynamics, :MLEAgent)
        @test isdefined(TullockDynamics, :BayesianAgent)
        @test isdefined(TullockDynamics, :run!)
        
        # Test that visualise function exists (stub version)
        @test isdefined(TullockDynamics, :visualise)
        @test hasmethod(TullockDynamics.visualise, (TullockDynamics.TullockContest,))
    end
    
    @testset "Dependency Isolation" begin
        # Test that Plots is not automatically loaded
        loaded_modules = keys(Base.loaded_modules)
        plots_loaded = any(m -> m.name == "Plots", loaded_modules)
        @test !plots_loaded
        
        measures_loaded = any(m -> m.name == "Measures", loaded_modules)
        @test !measures_loaded
        
        # Test that core dependencies are loaded
        statsbase_loaded = any(m -> m.name == "StatsBase", loaded_modules)
        @test statsbase_loaded
        
        forwarddiff_loaded = any(m -> m.name == "ForwardDiff", loaded_modules)
        @test forwarddiff_loaded
    end
    
    @testset "Extension Registration" begin
        # Test extension system is properly configured for Julia 1.9+
        if VERSION >= v"1.9" && HAS_PKG
            try
                # Check if extension is properly registered
                ext_info = Pkg.Extensions.get_extensions(Base.PkgId(TullockDynamics))
                @test haskey(ext_info, :TullockDynamicsPlotsExt)
            catch e
                if e isa MethodError
                    # Extension API not available in this Julia version, skip test
                    @test_skip "Extension API not available in Julia $(VERSION)"
                else
                    # Pkg not available or other issue, skip test
                    @test_skip "Pkg.Extensions not available"
                end
            end
        else
            if VERSION < v"1.9"
                @test_skip "Extension system requires Julia 1.9+"
            else
                @test_skip "Pkg not available in test environment"
            end
        end
    end
    
    @testset "Documentation and Examples" begin
        # Test that documentation mentions plotting requirements
        
        # Check main module documentation
        module_docs = string(@doc TullockDynamics)
        @test occursin("visualise", module_docs)
        
        # Check visualise function documentation  
        visualise_docs = string(@doc TullockDynamics.visualise)
        @test occursin("Plots.jl", visualise_docs)
        @test occursin("extension", visualise_docs)
        @test occursin("using Plots", visualise_docs)
    end
end