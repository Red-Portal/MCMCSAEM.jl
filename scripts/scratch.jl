
using Plots
using Random, Random123

using ADTypes
using Enzyme, ForwardDiff, ReverseDiff, Zygote
using MCMCSAEM

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    T      = 10000
    γ₀     = 1e-2
    γ      = t -> γ₀*t^(-1.0)
    h      = 1e-3

    model, θ₀, x₀ = problem(rng, Val(:logistic))

    θ_hist = zeros(length(θ₀), T)
    function callback!(t, x, θ)
        θ_hist[:,t] = θ
    end

    MCMCSAEM.mcmcsaem(rng, model, x₀, θ₀, T, γ, h; ad = ADTypes.AutoForwardDiff(), callback!)

    #θ_mean = cumsum(θ_hist, dims=2) ./ repeat((1:T)', 2, 1)

    #Plots.plot(  1 ./ θ_hist', label="μ trace", xlabel="SAEM Iteration", ylabel="Value") |> display
    #Plots.hline!(σ_true,             label="μ True") |> display

    #θ_true = [1.0, invsoftplus(0.1)]
    θ_true = [1.0, 0.1]
    Plots.plot(θ_hist', xlabel="SAEM Iteration", ylabel="Value") |> display
    Plots.hline!(θ_true, xlabel="SAEM Iteration", ylabel="Value") |> display
end
