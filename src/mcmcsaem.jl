
function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function mcmcsaem(
    rng        ::Random.AbstractRNG,
    model,
    x₀         ::AbstractMatrix,
    θ₀         ::AbstractVector,
    T          ::Int,
    T_burn     ::Int,
    γ_schedule,
    h          ::Real;
    ad         ::ADTypes.AbstractADType = ADTypes.AutoZygote,
    mcmc_type     = :ula,
    callback!     = nothing,
    show_progress = true
)
    x    = x₀ isa Vector ? reshape(x₀, (:,1)) : copy(x₀)
    θ    = copy(θ₀)
    prog = Progress(T + T_burn; enabled=show_progress, showspeed=true)

    ∇ℓ_buf = DiffResults.DiffResult(zero(eltype(x)), similar(x, size(x, 1)))

    for t = 1:T_burn
        x, _ = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h, ∇ℓ_buf)

        stats = (t=t, state=:burn_markovchain)
        pm_next!(prog, stats)
    end
    S = sufficient_statistic(model, x)

    for t = 1:T
        x, α   = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h, ∇ℓ_buf)
        V      = mean(x′ -> LogDensityProblems.logdensity(model, x′, θ), eachcol(x))
        S′      = sufficient_statistic(model, x)
        H      = S′ - S
        S_prev = S
        γₜ     = γ_schedule(t)
        S      = S + γₜ*H
        θ      = maximize_surrogate(model, S)

        stats = (t=t, loglike=V, γₜ=γₜ, ΔS=norm(S_prev - S), state=:run_mcmcsaem, acc=α)

        if !isnothing(callback!)
            stat′  = callback!(t, x, θ, stats)
            stats = !isnothing(stat′) ? merge(stats, stat′) : stats
        end
        pm_next!(prog, stats)
    end
    θ, x
end
