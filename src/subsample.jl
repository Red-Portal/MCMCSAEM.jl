
struct Subsampling{
    O <: AdvancedVI.AbstractVariationalObjective,
    B <: Integer,
    D <: AbstractVector,
} <: AdvancedVI.AbstractVariationalObjective
    objective::O
    batchsize::B
    data     ::D
end

function init_batch(
    rng      ::Random.AbstractRNG,
    data     ::AbstractVector,
    batchsize::Integer
)
    shuffled = Random.shuffle(rng, data)
    batches  = Iterators.partition(shuffled, batchsize)
    enumerate(batches)
end

function AdvancedVI.init(rng::Random.AbstractRNG, sub::Subsampling, λ, re)
    @unpack objective, batchsize, data = sub
    epoch     = 1
    sub_state = (epoch, init_batch(rng, data, batchsize))
    obj_state = AdvancedVI.init(rng, objective, λ, re)
    (sub_state, obj_state)
end

function update_subsampling(rng::Random.AbstractRNG, sub::Subsampling, sub_state)
    epoch, batch_itr         = sub_state
    (step, batch), batch_itr′ = Iterators.peel(batch_itr)
    epoch′, batch_itr′′        = if isempty(batch_itr′)
        epoch+1, init_batch(rng, sub.data, sub.batchsize)
    else
        epoch, batch_itr′
    end
    logstat = (epoch = epoch, step = step)
    batch, (epoch′, batch_itr′′), logstat
end

function AdvancedVI.estimate_gradient(
    rng    ::Random.AbstractRNG,
    ad     ::ADTypes.AbstractADType,
    sub    ::Subsampling,
    state,
    λ      ::AbstractVector{<:Real},
    re,
    out    ::DiffResults.MutableDiffResult
)
    objective = sub.objective

    sub_state, obj_state = state
    batch, sub_state′, sub_logstat = update_subsampling(rng, sub, sub_state)

    prob_sub          = subsample_problem(objective.prob, batch)
    obj_sub           = @set objective.prob = prob_sub

    out, obj_state′, obj_logstat = AdvancedVI.estimate_gradient(
        rng, ad, obj_sub, obj_state, λ, re, out
    )
    out, (sub_state′, obj_state′), merge(sub_logstat, obj_logstat)
end
