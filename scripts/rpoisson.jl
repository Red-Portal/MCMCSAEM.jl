
include("common.jl")

using RDatasets
using FastGaussQuadrature

struct RobustPoisson{
    Mat<:AbstractMatrix, QRFac, Vec<:AbstractVector
}
    X::Mat
    y::Vec
    QR::QRFac
end

function LogDensityProblems.capabilities(::Type{<:RobustPoisson})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::RobustPoisson, η::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y = model
    σ = θ[1]
    α = θ[2]
    β = θ[3:end]
    μ = X*β .+ α

    ℓp_η = logpdf(MvNormal(μ, σ), η)
    ℓp_x = mapreduce(+, η, y) do ηᵢ, yᵢ
        poislogpdf(exp(ηᵢ), yᵢ)
    end
    ℓp_x + ℓp_η
end

function MCMCSAEM.sufficient_statistic(::RobustPoisson, x::AbstractMatrix)
    mean(eachcol(x)) do xᵢ
        vcat(xᵢ, xᵢ.^2)
    end
end

function MCMCSAEM.maximize_surrogate(model::RobustPoisson, S::AbstractVector)
    SimpleUnPack.@unpack X, QR = model

    d   = div(length(S), 2)
    EX  = S[1:d]
    EX² = S[d+1:end]
    μ   = mean(EX)
    σ   = sqrt(mean(EX²) - μ^2)
    β   = QR\EX
    α   = mean(X*β .- μ)
    vcat([σ], [α], β)
end

function load_dataset(::Val{:medpar})
    data = RDatasets.dataset("COUNT", "medpar")   

    y = data[:,"Los"]
    X = data[:, [
        "HMO",
        "White",
        "Age80",
        "Type1",
        "Type2",
        "Type3",
    ]]
    X, y = Matrix(X), Vector(y)
    X, y
end

function load_dataset(::Val{:rwm5yr})
    data = RDatasets.dataset("COUNT", "rwm5yr")   

    y = data[:,"DocVis"]
    X = data[:, [
        "Age",
        "Educ",
        "HHNInc",
        "OutWork",
        "Female",
        "Married",
        "Kids",
        "Self",
        "EdLevel1",
        "EdLevel2",
        "EdLevel3",
        "EdLevel4",
    ]]

    y = Vector{Float32}(y)
    X = Matrix{Float32}(X)

    X[:,[1,2,3]] .-= mean(X[:,[1,2,3]])
    X[:,[1,2,3]]  /= std( X[:,[1,2,3]])
    X, y
end

function load_dataset(::Val{:azpro})
    data = RDatasets.dataset("COUNT", "azpro")   

    y = data[:,"Los"]
    X = data[:, [
        "Procedure",
        "Sex",
        "Admit",
        "Age75",
    ]]
    y = Vector{Float32}(y)
    X = Matrix{Float32}(X)
    X, y
end

function predictive_loglikelihood(model::RobustPoisson, X, y, β, α, σ)
    μ    = X*β .+ α
    x, w = gausshermite(1024)

    @tullio ℓp_y[i,j] := log(w[i]) + poislogpdf(exp(μ[j] + √2*σ*x[i]), y[j])
    mean(logsumexp(ℓp_y, dims=1))
end

function run_problem(::Val{:rpoisson}, dataset, mcmc_type, h, key=1, show_progress=false)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    ad   = ADTypes.AutoReverseDiff()

    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    T_burn = 500
    T      = 1000
    γ₀     = 1e-0
    γ      = t -> γ₀ / sqrt(t)
    m      = 1    # n_chains
    

    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)

    X, y = load_dataset(dataset)

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y; ratio=0.8)

    model = RobustPoisson(X_train, y_train, qr(X_train))

    X     = model.X
    d     = size(X, 2)
    σ₀    = .1
    β₀    = zeros(d)
    α₀    = 0.0
    θ₀    = vcat([σ₀], [α₀], β₀)
    μ₀    = X*β₀ .+ α₀
    x₀    = rand(rng, MvNormal(μ₀, σ₀), m)

    θ_hist = zeros(length(θ₀), T)
    V_hist = zeros(T)
    function callback!(t, x, θ, stats)
        θ_hist[:,t] = θ
        V_hist[t]   = stats.loglike
        nothing
    end
    θ, _ = MCMCSAEM.mcmcsaem(
        rng, model, x₀, θ₀, T, T_burn, γ, h;
        ad, callback!, show_progress = show_progress,
        mcmc_type = mcmc_type,
    )
    Plots.plot(V_hist) |> display
    #Plots.plot(θ_hist') |> display

    σ = θ[1]
    α = θ[2]
    β = θ[3:end]

    lpd = predictive_likelihood(X_test, y_test, β, α, σ)
    DataFrame(lpd=lpd)
end

function main(::Val{:rpoisson}, mcmc_type)
    n_trials = 32
    datasets = [
        (dataset = :medpar,),
        (dataset = :azpro,),
    ]
    stepsizes = [(stepsize = 10.0.^logstepsize,) for logstepsize ∈ range(-5, -1, length=11) ]

    configs = Iterators.product(datasets, stepsizes) |> collect
    configs = reshape(configs, :)
    configs = map(x -> merge(x...), configs)

    data = @showprogress mapreduce(vcat, configs) do config
        SimpleUnPack.@unpack stepsize, dataset = config
        dfs = @showprogress pmap(1:n_trials) do key
            run_problem(Val(:rpoisson), Val(dataset), mcmc_type, stepsize, key, false)
        end
        df = vcat(dfs...)
        for (k, v) ∈ pairs(config)
            df[:,k] .= v
        end
        df
    end

    JLD2.save(datadir("exp_pro", "robust_poisson.jld2"), "data", data)

    h5open(datadir("exp_pro", "robust_poisson.h5"), "w") do h5
        for dataset ∈ [:medpar, :azpro]
            data′ = data[data[:,:dataset] .== dataset,:]
            data′′ = @chain groupby(data′, :stepsize) begin
                @combine(:lpd_ci   = run_bootstrap(:lpd))
            end
            h  = data′′[:,:stepsize]
            
            lpd      = data′′[:,:lpd_ci]
            lpd_mean = [lpdᵢ[1] for lpdᵢ ∈ lpd]
            lpd_p    = [abs(lpdᵢ[2] - lpdᵢ[1]) for lpdᵢ ∈ lpd]
            lpd_m    = [abs(lpdᵢ[3] - lpdᵢ[1]) for lpdᵢ ∈ lpd]

            write(h5, "h_$(dataset)",   h)
            write(h5, "lpd_$(dataset)", hcat(lpd_mean, lpd_p, lpd_m)' |> Array)
        end
    end
    data
end
