
function compute_update!(
    ad    ::ADTypes.AbstractADType,
    model,
    x     ::AbstractArray,
    θ     ::AbstractVector,
    H_buf
)
    q(θ′) = begin
        mean(eachcol(x)) do xᵢ
            LogDensityProblems.logdensity(model, xᵢ, θ′)
        end
    end
    value_and_gradient!(ad, q, θ, H_buf)
    DiffResults.gradient(H_buf), DiffResults.value(H_buf)
end

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
    callback!     = nothing,
    show_progress = true
)
    x    = x₀ isa Vector ? reshape(x₀, (:,1)) : copy(x₀)
    θ    = copy(θ₀)
    prog = Progress(T + T_burn; enabled=show_progress, showspeed=true)

    H_buf  = DiffResults.DiffResult(zero(eltype(θ)), similar(θ))
    ∇ℓ_buf = DiffResults.DiffResult(zero(eltype(x)), similar(x, size(x, 1)))

    for t = 1:T_burn
        x = ula_transition!(rng, ad, model, x, θ, h, ∇ℓ_buf)

        stats = (t=t, state=:burn_markovchain)
        pm_next!(prog, stats)
    end

    # for t = 1:T_burn
    #     H, _ = compute_update!(ad, model, x, θ, H_buf)
    #     γₜ   = γ_schedule(t)
    #     θ    = project(model, θ + γₜ*H)

    #     stats = (t=t, state=:burn_markovchain)
    #     pm_next!(prog, stats)
    # end

    for t = 1:T
        x    = ula_transition!(rng, ad, model, x, θ, h, ∇ℓ_buf)
        H, V = compute_update!(ad, model, x, θ, H_buf)

        γₜ     = γ_schedule(t)
        θ_prev = θ
        θ      = project(model, θ + γₜ*H)

        stats = (t=t, loglike=V, γₜ=γₜ, Δθ=norm(θ_prev - θ), state=:run_mcmc_saem)

        if !isnothing(callback!)
            stat′  = callback!(t, x, θ, stats)
            stats = !isnothing(stat′) ? merge(stats, stat′) : stats
        end
        pm_next!(prog, stats)
    end
    θ, x
end
