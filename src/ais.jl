

function ais(
    rng      ::Random.AbstractRNG,
    model,
    θ        ::AbstractVector,
    h        ::Real,
    q0,
    schedule ::AbstractVector,
    n_samples::Int;
    ad       ::ADTypes.AbstractADType = ADTypes.AutoZygote,
    mcmc_type::Symbol = :ula,
    mcmc_steps::Int   = 4,
    show_progress     = true,
)
    @assert first(schedule) == 0 && last(schedule)  == 1

    ∇ℓ_buf = DiffResults.DiffResult(
        zero(eltype(q0)), zeros(eltype(q0), length(q0))
    )

    prog = Progress(
        n_samples*length(schedule);
        enabled=show_progress,
        showspeed=true
    )

    hist = zeros(n_samples, length(schedule))

    mean(1:n_samples) do i
        logw      = 0.0
        x         = rand(rng, q0)
        tempered  = Accessors.@set model.temp = first(schedule)
        logp_prev = LogDensityProblems.logdensity(tempered, x, θ)
        avg_α     = 0.0
        for (j, temp) in enumerate(drop(schedule, 1))
            tempered  = Accessors.@set model.temp = temp
            logp_curr = LogDensityProblems.logdensity(tempered, x, θ)
            logw     += logp_curr - logp_prev

            for k in 1:mcmc_steps
                x, α = mcmc_transition!(
                    rng, ad, Val(mcmc_type), tempered, x, θ, h, ∇ℓ_buf
                )
                avg_α = (k-1)/k*avg_α + α/k
            end

            stats = (
                sample      = i,
                step        = j,
                temperature = temp,
                logw        = logw,
                acc_rate    = avg_α,
            )
            logp_prev = LogDensityProblems.logdensity(tempered, x, θ)
            pm_next!(prog, stats)
            hist[i,j] = logw
        end
        logw
    end
end
