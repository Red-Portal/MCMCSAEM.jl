
include("common.jl")

struct Logistic{Mat <: AbstractMatrix, Vec <: AbstractVector}
    X::Mat
    y::Vec
end

function LogDensityProblems.capabilities(::Type{<:Logistic})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::Logistic, β::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y = model
    d    = size(X,2)
    s    = X*β   
    μ, σ = θ[1], θ[2]
    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(BernoulliLogit(sᵢ), yᵢ)
    end
    ℓp_β = logpdf(MvNormal(Fill(μ, d), σ), β)
    ℓp_x + ℓp_β
end

function MCMCSAEM.sufficient_statistic(
    ::Logistic, x::AbstractMatrix, θ::AbstractVector
)
    μ = θ[1]
    mean(eachcol(x)) do xᵢ
        [mean(xᵢ), mean(abs2, xᵢ .- μ)]
    end
end

function MCMCSAEM.maximize_surrogate(::Logistic, S::AbstractVector)
    μ, σ2 = S[1], S[2]
    [μ, sqrt(σ2)]
end

function MCMCSAEM.preconditioner(model::Logistic, θ::AbstractVector)
    I
end

function run_problem(::Val{:logistic}, mcmc_type, h, key=1, show_progress=false)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    ad   = ADTypes.AutoReverseDiff()

    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    T      = 100
    T_burn = 10
    γ₀     = 1e-0
    γ      = t -> γ₀ / sqrt(t)
    m      = 1    # n_chains
    n      = 1000 # n_datapoints
    d      = 100  # n_regressors

    κ        = 1000
    L        = 100
    X        = randn(rng, n, d)
    U, _, Vt = svd(X)
    D        = Diagonal(range(L, L/sqrt(κ); length=min(n, d)))
    X        = U*D*Vt

    θ_true   = [1.0, 0.1]
    β_true   = rand(rng, Normal(θ_true[1], θ_true[2]), d)
    y        = rand.(rng, BernoulliLogit.(X*β_true))

    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)

    model = Logistic(X, y)
    θ₀    = [0.0, 1.0]
    x₀    = randn(rng, d, m)

    θ_hist = zeros(length(θ₀), T)
    function callback!(t, x, θ, stats)
        θ_hist[:,t] = θ
        nothing
    end

    MCMCSAEM.mcmcsaem(
        rng, model, x₀, θ₀, T, T_burn, γ, h;
        ad, callback!, show_progress = show_progress,
        mcmc_type = mcmc_type,
        n_inner_mcmc = 4,
    )

    Plots.plot(θ_hist', xlabel="SAEM Iteration", ylabel="Value") |> display
    Plots.hline!(θ_true) |> display
    sum(abs2, θ_true - θ_hist[:,end]), θ_true, θ_hist
end

function error_scaling(mcmc_type)
    n_trials  = 32
    n_samples = 16

    h_range = 10.0.^range(-5, 0; length=n_samples)
    y       = @showprogress mapreduce(vcat, h_range) do h
        y_samples = @showprogress map(1:n_trials) do key
            run_problem(Val(:logistic), mcmc_type, h, key)[1]
        end
        reshape(y_samples, (1,:))
    end
    Plots.plot(h_range, mean(y, dims=2)[:,1], xscale=:log10) |> display

    h5open(datadir("exp_pro", "logistic_error_$(mcmc_type)_h.h5"), "w") do h5
        raw = repeat(h_range, outer=n_trials)
        write(h5, "h_raw", reshape(raw,:))
        write(h5, "y_raw", reshape(y,:))

        y_med = median(y,dims=2)[:,1]
        y_90  = [quantile(yᵢ, 0.90) for yᵢ ∈ eachrow(y)]
        y_10  = [quantile(yᵢ, 0.10) for yᵢ ∈ eachrow(y)]
        y_p   = abs.(y_90 - y_med)
        y_m   = abs.(y_10 - y_med)

        write(h5, "h",   h_range)
        write(h5, "err", hcat(y_med, y_p, y_m)' |> Array)
    end
end

function complexity(mcmc_type)
    n_trials  = 64
    n_samples = 16
    ϵ         = 0.001

    h_range = 10.0.^range(-5, 0; length=n_samples)
    y_cis   = @showprogress mapreduce(vcat, h_range) do h
        y_samples = @showprogress map(1:n_trials) do key
            θ_true, θ_hist = run_problem(Val(:logistic), mcmc_type, h, key)[2:3]

            err_hist = map(xᵢ -> sum(abs2, xᵢ), eachcol(θ_hist .- θ_true))
            T        = findfirst(err_hist .< ϵ)
            !isnothing(T) ? convert(Float64, T) : 1e+5
        end
        run_bootstrap(y_samples)
    end
    Plots.plot(h_range, [y_ci[1] for y_ci ∈ y_cis], xscale=:log10) |> display

    h5open(datadir("exp_pro", "logistic_complexity_$(mcmc_type)_h.h5"), "w") do h5
        y_mean = [accᵢ[1] for accᵢ ∈ y_cis]
        y_p    = [abs(yᵢ[2] - yᵢ[1]) for yᵢ ∈ y_cis]
        y_m    = [abs(yᵢ[3] - yᵢ[1]) for yᵢ ∈ y_cis]
        write(h5, "h",   h_range)
        write(h5, "err", hcat(y_mean, y_p, y_m)' |> Array)
    end
end

function deviation()
    for h ∈ [1e-1, 1e-2, 1e-3]
        n_trials = 16

        data = mapreduce((x, y) -> cat(x, y, dims=3), 1:n_trials) do key
            run_problem(Val(:logistic), h, key, true)[2]
        end

        h5open(datadir("exp_pro", "logistic_deviation_h=$(h).h5"), "w") do h5
            for i = 1:n_trials
                write(h5, "mean_$(i)", data[1,:,i])
            end

            for i = 1:n_trials
                write(h5, "stddev_$(i)", data[2,:,i] )
            end

            θ₁_μ = mean(data[1,:,:], dims=2)[:,1]
            θ₁_σ = std( data[1,:,:], dims=2, corrected=false)[:,1]
            write(h5, "mean_dev_pos", θ₁_μ + θ₁_σ)
            write(h5, "mean_dev_neg", θ₁_μ - θ₁_σ)

            θ₂_μ = mean(data[2,:,:], dims=2)[:,1]
            θ₂_σ = std( data[2,:,:], dims=2, corrected=false)[:,1]
            write(h5, "stddev_dev_pos", θ₂_μ + θ₂_σ)
            write(h5, "stddev_dev_neg", θ₂_μ - θ₂_σ)
        end
    end
end
