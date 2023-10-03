
include("common.jl")

struct LogisticARD{Mat <: AbstractMatrix, Vec <: AbstractVector}
    X::Mat
    y::Vec
end

function LogDensityProblems.capabilities(::Type{<:LogisticARD})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(
    model::LogisticARD, β::AbstractVector, θ::AbstractVector
)
    SimpleUnPack.@unpack X, y = model
    d = size(X,2)
    σ = 1 ./ θ
    s = X*β[2:end] .+ β[1]
    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(BernoulliLogit(sᵢ), yᵢ)
    end
    ℓp_β = logpdf(MvNormal(Zeros(d), σ), β[2:end])
    ℓp_α = logpdf(Normal(0, 10), β[1])
    ℓp_x + ℓp_β + ℓp_α
end

function MCMCSAEM.project(::LogisticARD, θ::AbstractVector)
    @. clamp(θ, 1e-3, Inf)
end

function dataset(::Val{:colon})
    mat  = MAT.matread(datadir("dataset", "colon.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    y    = y .== 1.0
    X, y
end

function dataset(::Val{:leukemia})
    mat  = MAT.matread(datadir("dataset", "leukemia.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    X = X .- mean(X, dims=1)
    X = X ./ std(X, dims=1)
    y = y .== 1.0
    X, y
end

function dataset(::Val{:prostate})
    mat  = MAT.matread(datadir("dataset", "prostate.mat"))
    X, y = Array(mat["X"]), mat["Y"][:,1]
    X = X .- mean(X, dims=1)
    X = X ./ std(X, dims=1)
    y = y .== 2.0
    X, y
end

function main(::Val{:logisticard}, key=1)
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, key)
    ad   = ADTypes.AutoReverseDiff()

    X, y = dataset(Val(:prostate))

    X_train, y_train, X_test, y_test =  prepare_dataset(rng, X, y)

    d = size(X_train, 2)

    T_burn    = 500
    T         = 1000
    γ₀        = 1e-0
    γ         = t -> γ₀ / t
    h         = 1e-4
    m         = 1    # n_chains

    model = LogisticARD(X_train, y_train)
    σ₀    = 0.1
    θ₀    = fill(1/σ₀, d)
    x₀    = σ₀*randn(rng, d+1, m)

    θ_hist = zeros(length(θ₀), T)
    V_hist = zeros(T)
    function callback!(t, x, θ, stats)
        θ_hist[:,t] = θ
        V_hist[t]   = stats.loglike
        nothing
    end

    θ, x = MCMCSAEM.mcmcsaem(rng, model, x₀, θ₀, T, T_burn, γ, h; ad, callback!)
    Plots.plot!(1 ./ θ) |> display
    #Plots.plot(V_hist)

    # model_sel = LogisticARD(X_train[:,select_idx], y_train)
    # θ_sel     = θ[select_idx]
    # x_sel     = x[vcat([1], select_idx .+ 1),:]
    # β_post    = MCMCSAEM.mcmc(rng, model_sel, θ_sel, x_sel, 1e-4, 2000; ad)
    # β_post    = β_post[:,1000:10:end]
    # X_test    = hcat(ones(size(X_test,1)), X_test[:,select_idx])

    θ = mean(θ_hist, dims=2)[:,1]

    β_post = MCMCSAEM.mcmc(rng, model, θ, x, 1e-3, 2000; ad)
    X_test = hcat(ones(size(X_test,1)), X_test)

    logits = X_test*β_post
    p_test = mean(logistic.(logits), dims=2)[:,1]

    @info("",
        acc   = mean((p_test .> 0.5) .== y_test),
        mlpd1 = mapreduce((pᵢ, yᵢ) -> logpdf(Bernoulli(pᵢ), yᵢ), +, p_test, y_test) / length(y_test),
    )

end

# struct MyNormal{Dist}
#     d::Dist
# end

# function LogDensityProblems.capabilities(::Type{<:LogisticARD})
#     LogDensityProblems.LogDensityOrder{0}()
# end

# function LogDensityProblems.logdensity(
#     model::MyNormal, β::AbstractVector, θ::AbstractVector
# )
#     logpdf(model.d, β)
# end

# function main()
#     seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
#     rng  = Philox4x(UInt64, seed, 8)
#     set_counter!(rng, 1)
#     ad   = ADTypes.AutoReverseDiff()

#     μ     = randn(4)
#     σ     = [.1, .2, .3, .4]
#     model = MyNormal(MvNormal(μ, σ))

#     x = MCMCSAEM.mcmc(rng, model, [], randn(4), 3e-1, 10000; ad)
#     μ, x
# end
