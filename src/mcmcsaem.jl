
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
    mcmc_type         = :ula,
    callback!         = nothing,
    show_progress     = true,
    n_inner_mcmc::Int = 1
)
    x     = x₀ isa Vector ? reshape(x₀, (:,1)) : copy(x₀)
    θ     = copy(θ₀)
    prog  = Progress(T + T_burn; enabled=show_progress, showspeed=true)
    stats = Vector{NamedTuple}(undef, T + T_burn)

    ∇ℓ_buf = DiffResults.DiffResult(zero(eltype(x)), similar(x, size(x, 1)))

    for t = 1:T_burn
        x, _ = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h, ∇ℓ_buf)

        stats[t] = (t=t, state=:burn_markovchain)
        pm_next!(prog, stats[t])
    end
    S = sufficient_statistic(model, x, θ)

    for t = 1:T
        for _ = 1:n_inner_mcmc
            x, _ = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h, ∇ℓ_buf)
        end
        x, α   = mcmc_transition!(rng, ad, Val(mcmc_type), model, x, θ, h, ∇ℓ_buf)
        V      = mean(x′ -> LogDensityProblems.logdensity(model, x′, θ), eachcol(x))
        S′      = sufficient_statistic(model, x, θ)
        H      = S′ - S
        S_prev = S
        γₜ     = γ_schedule(t)
        S      = S + γₜ*H
        θ      = maximize_surrogate(model, S)

        stat = (t=t, loglike=V, γₜ=γₜ, ΔS=norm(S_prev - S), state=:run_mcmcsaem, acc=α)

        if !isnothing(callback!)
            stat′ = callback!(t, x, θ, stat)
            stat = !isnothing(stat′) ? merge(stat, stat′) : stat
        end
        stats[t+T_burn] = stat
        pm_next!(prog, stat)
    end
    θ, stats
end
