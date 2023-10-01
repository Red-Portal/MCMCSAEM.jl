
using Distributions
using FillArrays 
using LogDensityProblems
using Plots
using Random, Random123
using SimpleUnPack

using ADTypes
using Enzyme, ForwardDiff, ReverseDiff, Zygote
using MCMCSAEM

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
    @unpack X, y = model
    d = size(X,2)
    σ = 1 ./ θ
    s = X*β   
    ℓp_x = mapreduce(+, s, y) do sᵢ, yᵢ
        logpdf(BernoulliLogit(sᵢ), yᵢ)
    end
    ℓp_β = logpdf(MvNormal(Zeros(d), σ), β)
    ℓp_x + ℓp_β
end

function MCMCSAEM.project(::LogisticARD, θ::AbstractVector)
    @. max(θ, 1e-5)
end


function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    n      = 500
    d      = 5
    X      = randn(rng, n, d)
    β_true = [1., 1., 1., -1., 0.]
    y      = rand.(rng, BernoulliLogit.(X*β_true))

    model = LogisticARD(X, y)

    T      = 1000
    γ₀     = 1e-1
    γ      = t -> γ₀*t^(-0.1)
    h      = 1e-2

    b   = 5
    x₀s = randn(d, b)
    θ₀  = ones(d) #softplus.(randn(rng, d))

    θ_hist = zeros(length(θ₀), T)
    function callback!(t, xs, θ)
        θ_hist[:,t] = θ
    end

    MCMCSAEM.mcmcsaem(rng, model, x₀s, θ₀, T, γ, h; ad = ADTypes.AutoForwardDiff(), callback!)

    #θ_mean = cumsum(θ_hist, dims=2) ./ repeat((1:T)', 2, 1)

    Plots.plot(  1 ./ θ_hist', label="μ trace", xlabel="SAEM Iteration", ylabel="Value") |> display
    #Plots.hline!(σ_true,             label="μ True") |> display
end
