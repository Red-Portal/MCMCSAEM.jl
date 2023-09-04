
function prepare_dataset(rng::Random.AbstractRNG,
                         data_x::AbstractMatrix,
                         data_y::AbstractVector;
                         ratio::Real=0.9)
    n_data      = size(data_x, 1)
    shuffle_idx = Random.shuffle(rng, 1:n_data)
    data_x      = data_x[shuffle_idx,:]
    data_y      = data_y[shuffle_idx]

    n_train = floor(Int, n_data*ratio)
    x_train = data_x[1:n_train, :]
    y_train = data_y[1:n_train]
    x_test  = data_x[n_train+1:end, :]
    y_test  = data_y[n_train+1:end]

    X_train    = Array{Float32}(x_train')
    X_test     = Array{Float32}(x_test')
    μ_X        = mean(X_train, dims=2)[:,1]
    σ_X        = std(X_train, dims=2)[:,1]
    X_train  .-= μ_X
    X_test   .-= μ_X
    X_train  ./= σ_X
    X_test   ./= σ_X

    X_train, y_train, X_test, y_test
end

function load_dataset(::Val{:colon})
    data   = MAT.matread(datadir("dataset", "colon.mat"))
    data_x = data["X"]
    data_y = (data["Y"][:,1] .+ 1) / 2
    data_x, data_y
end

function load_dataset(::Val{:sml})
    dataset = MAT.matread(
        datadir("dataset", "sml.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]

    valid_features = std(data_x, dims=1)[1,:] .> 1e-5

    data_x = data_x[:, valid_features]
    data_x, data_y
end

function load_dataset(::Val{:energy})
    dataset = MAT.matread(
        datadir("dataset", "energy.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:skillcraft})
    dataset = MAT.matread(
        datadir("dataset", "skillcraft.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:parkinsons})
    dataset = MAT.matread(
        datadir("dataset", "parkinsons.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:kin40k})
    dataset = MAT.matread(
        datadir("dataset", "kin40k.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:airfoil})
    dataset = MAT.matread(
        datadir("dataset", "airfoil.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:gas})
    dataset = MAT.matread(
        datadir("dataset", "gas.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:wine})
    dataset = MAT.matread(
        datadir("dataset", "wine.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:boston})
    fname = datadir(joinpath("dataset", "housing.csv"))
    vals  = readdlm(fname)
    X = Array{Float32}(vals[:, 1:end-1])
    y = Array{Float32}(vals[:, end])
    X, y
end

function load_dataset(::Val{:concrete})
    dataset = MAT.matread(
        datadir("dataset", "concrete.mat"))["data"]
    data_x = dataset[:, 1:end-1]
    data_y = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:yacht})
    fname = datadir(joinpath("dataset", "yacht_hydrodynamics.data"))
    s     = open(fname, "r") do io
        s = read(io, String)
        replace(s, "  " => " ")
    end

    io   = IOBuffer(s)
    vals = readdlm(io, ' ', '\n', header=false)

    X = Array{Float32}(vals[:, 1:end-2])
    y = Array{Float32}(vals[:, end-1])
    X, y
end

function load_dataset(::Val{:naval})
    fname = datadir(joinpath("dataset", "naval_propulsion.txt"))
    s     = open(fname, "r") do io
        s = read(io, String)
        s = replace(s, "   " => " ")
    end

    io   = IOBuffer(s)
    vals = readdlm(io, ' ', '\n', header=false)
    vals = vals[:,2:end]

    X = Array{Float32}(vals[:, 1:end-2])
    y = Array{Float32}(vals[:, end-1])

    feature_idx = 1:size(X,2)
    X = Float64.(X[:, setdiff(feature_idx, (9, 12))])

    X, y
end

function load_dataset(::Val{:heart})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "heart-disease.csv"), ',', skipstart=1)
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:sonar})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "sonar.csv"), ',', skipstart=1)
    feature_idx = 1:size(dataset, 2)-1
    data_x      = Float64.(dataset[:, setdiff(feature_idx, 2),])
    data_y      = dataset[:, end]
    data_y      = map(data_y) do s
        if(s == "Rock")
            1.0
        else
            0.0
        end
    end
    data_x, data_y
end

function load_dataset(::Val{:ionosphere})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "ionosphere.csv"), ',')
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]

    data_y[data_y .== "g"] .= 1.0
    data_y[data_y .== "b"] .= 0.0

    valid_features = std(data_x, dims=1)[1,:] .> 1e-5
    data_x = data_x[:,valid_features]
    data_x = Float64.(data_x)
    data_y = Float64.(data_y)
    data_x, data_y
end

function load_dataset(::Val{:breast})
    dataset = DelimitedFiles.readdlm(datadir("dataset", "wdbc.data"), ',')
    data_x  = dataset[:, 3:end]
    data_y  = dataset[:, 2]

    data_y[data_y .== "M"] .= 1.0
    data_y[data_y .== "B"] .= 0.0
    data_x = Float64.(data_x)
    data_y = Float64.(data_y)

    data_x, data_y
end

function load_dataset(::Val{:australian})
    dataset = DelimitedFiles.readdlm(datadir("dataset", "australian.dat"), ' ')
    data_x  = dataset[:, 1:end-1]
    data_y  = dataset[:, end]
    data_x, data_y
end
