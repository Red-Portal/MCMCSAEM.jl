
function value_and_gradient!(
    ad::ADTypes.AutoReverseDiff, f, θ::AbstractVector{<:Real}, out::DiffResults.MutableDiffResult
)
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

getchunksize(::ADTypes.AutoForwardDiff{chunksize}) where {chunksize} = chunksize

function value_and_gradient!(
    ad::ADTypes.AutoForwardDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    chunk_size = getchunksize(ad)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, θ)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size))
    end
    ForwardDiff.gradient!(out, f, θ, config)
    return out
end

function value_and_gradient!(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y = f(θ)
    DiffResults.value!(out, y)
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    return out
end

function value_and_gradient!(
    ad::ADTypes.AutoZygote, f, θ::AbstractVector{<:Real}, out::DiffResults.MutableDiffResult
)
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(y))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, only(∇θ))
    return out
end
