module LIMPSBridge

using HTTP
using Sockets
using JSON3
using LinearAlgebra
using Statistics
using Graphs
using SimpleWeightedGraphs
using NearestNeighbors
using StatsBase

# Try to detect a locally available AL_ULS module
const HAS_ALULS = let ok = false
    try
        @eval using AL_ULS
        ok = isdefined(@__MODULE__, :AL_ULS) && isdefined(AL_ULS, :optimize)
    catch
        ok = false
    end
    ok
end

struct CoherenceResult
    ghost_score::Float64
    avg_entropy::Float64
    avg_sensitivity::Float64
    non_idempotence::Float64
    non_commutativity::Float64
    non_associativity::Float64
    hotspots::Vector{String}
end

kfp_smooth(x; α=0.35) = α * x + (1 - α) * tanh(x)

function compute_coherence(result)
    fns = get(result, "functions", Any[])
    ent = Float64[]
    sens = Float64[]
    nonidem = Float64[]
    noncomm = Float64[]
    nonassoc = Float64[]
    hotspots = String[]

    for f in fns
        p = get(f, "probe", nothing)
        p === nothing && continue
        push!(ent, try parse(Float64, string(get(p, "entropy_bits", 0.0))) catch; 0.0 end)
        push!(sens, try parse(Float64, string(get(p, "sensitivity", 0.0))) catch; 0.0 end)
        idr = try parse(Float64, string(get(p, "idempotent_rate", 0.0))) catch; 0.0 end
        push!(nonidem, max(0.0, 1.0 - idr))
        cr = get(p, "commutative_rate", nothing)
        ar = get(p, "associative_rate", nothing)
        push!(noncomm, cr === nothing ? 0.0 : max(0.0, 1.0 - try parse(Float64, string(cr)) catch; 0.0 end))
        push!(nonassoc, ar === nothing ? 0.0 : max(0.0, 1.0 - try parse(Float64, string(ar)) catch; 0.0 end))
        if (idr < 0.4) || ((cr !== nothing) && cr < 0.5) || ((ar !== nothing) && ar < 0.5)
            push!(hotspots, String(get(f, "qualname", "")))
        end
    end

    μH = isempty(ent) ? 0.0 : mean(ent)
    μS = isempty(sens) ? 0.0 : mean(sens)
    μNid = isempty(nonidem) ? 0.0 : mean(nonidem)
    μNc = isempty(noncomm) ? 0.0 : mean(noncomm)
    μNa = isempty(nonassoc) ? 0.0 : mean(nonassoc)

    raw = 0.45 * μH / 8 + 0.25 * kfp_smooth(μS) + 0.2 * μNid + 0.05 * μNc + 0.05 * μNa
    ghost = 1 / (1 + exp(-4 * (raw - 0.5)))
    return CoherenceResult(ghost, μH, μS, μNid, μNc, μNa, hotspots)
end

function result_to_json(cr::CoherenceResult)
    return JSON3.write(Dict(
        :ghost_score => cr.ghost_score,
        :avg_entropy => cr.avg_entropy,
        :avg_sensitivity => cr.avg_sensitivity,
        :non_idempotence => cr.non_idempotence,
        :non_commutativity => cr.non_commutativity,
        :non_associativity => cr.non_associativity,
        :hotspots => cr.hotspots,
    ))
end

function _to_matrix(A)
    rows = Vector{Vector{Float64}}(undef, length(A))
    for i in 1:length(A)
        rows[i] = Float64.(A[i])
    end
    # Stack rows into a Matrix
    return reduce(vcat, (permutedims(r) for r in rows))
end

function run_al_uls(adj::Matrix{Float64}; kwargs...)
    if HAS_ALULS
        return AL_ULS.optimize(adj; kwargs...)
    else
        url = get(ENV, "AL_ULS_URL", "")
        isempty(url) && error("AL_ULS_URL not set and local AL_ULS not available")
        payload = JSON3.write(Dict(:adjacency => adj, :options => Dict(kwargs)))
        resp = HTTP.post(string(url, "/optimize"), ["Content-Type" => "application/json"], payload)
        resp.status == 200 || error("al-ULS HTTP error: $(resp.status)")
        return JSON3.read(String(resp.body))
    end
end

function handle(req::HTTP.Request)
    target = String(req.target)
    if req.method == "GET" && target == "/health"
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(Dict("ok" => true)))
    elseif req.method == "POST" && target == "/coherence"
        payload = JSON3.read(String(req.body))
        cr = compute_coherence(payload)
        return HTTP.Response(200, ["Content-Type" => "application/json"], result_to_json(cr))
    elseif req.method == "POST" && target == "/optimize"
        payload = JSON3.read(String(req.body))
        adj_json = get(payload, "adjacency", nothing)
        adj_json === nothing && return HTTP.Response(400, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "missing adjacency")))
        adj = _to_matrix(adj_json)
        mode = get(payload, "mode", "kfp")
        beta = try parse(Float64, string(get(payload, "beta", 0.8))) catch; 0.8 end
        adj2 = kfp_smooth.(adj .* beta)
        result = run_al_uls(adj2; mode=mode, beta=beta)
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(Dict(:ok => true, :mode => mode, :beta => beta, :n => size(adj2, 1), :result => result)))
    else
        return HTTP.Response(404, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "not found")))
    end
end

function serve(; host::AbstractString = "0.0.0.0", port::Integer = 8099)
    @info "LIMPSBridge listening" host=host port=port
    HTTP.serve(handle, host, port)
end

#############################
# Costa–Hero Intrinsic ID   #
#############################
module CostaHeroID

using NearestNeighbors, Statistics, StatsBase, LinearAlgebra, Random

export id_entropy_global, id_entropy_local

"""
Compute k-NN distances matrix for points X (d x N), returning:
 - idx :: Vector{Vector{Int}}   (indices of k neighbors, excluding self)
 - dst :: Vector{Vector{Float64}} (corresponding distances)
"""
function knn_distances(X::AbstractMatrix{<:Real}; k::Int=10, metric::Symbol=:euclidean)
    d, N = size(X)
    tree = KDTree(Matrix{Float64}(X))
    D, I = knn(tree, Matrix{Float64}(X), k + 1, true) # includes self
    # strip self (first column)
    idx = [vec(I[i][2:end]) for i in 1:N]
    dst = [vec(D[i][2:end]) for i in 1:N]
    return idx, dst
end

"""
Entropic length L_γ for a k-NN graph over a subset S (indices into 1..N).
We sum distance^γ over each directed edge i→j (j in kNN(i) ∩ S),
then divide by 2 to approximately avoid double-counting undirected edges.
"""
function L_gamma_subset(S::Vector{Int}, idx_all::Vector{Vector{Int}}, dst_all::Vector{Vector{Float64}}; γ::Float64=1.0)
    present = fill(false, length(idx_all))
    @inbounds for s in S; present[s] = true; end
    total = 0.0
    @inbounds for i in S
        neigh = idx_all[i]; dists = dst_all[i]
        for (j, dij) in zip(neigh, dists)
            if present[j]
                total += dij^γ
            end
        end
    end
    return total / 2.0
end

"""
Fit a line y = a x + b by least squares; returns (a, b).
"""
function ls_fit(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    x̄ = mean(xs); ȳ = mean(ys)
    num = sum((x - x̄)*(y - ȳ) for (x,y) in zip(xs,ys))
    den = sum((x - x̄)^2 for x in xs) + eps()
    a = num/den
    b = ȳ - a*x̄
    return a, b
end

"""
GLOBAL Costa–Hero estimate from k-NN entropic lengths.

Returns (m_hat, H_hat, table):
 - m_hat :: Float64
 - H_hat :: Float64 (Rényi α-entropy proxy via growth law)
 - table :: NamedTuple with diagnostics (a, b, points used, γ, α)
Arguments:
 - X (d x N): data matrix (columns are points, already in your vector space)
 - k: neighbors for k-NN graph
 - γ: entropic power (0.5..1 recommended)
 - α: Rényi order (0<α<1, typical 0.5 or match paper’s choice)
 - plist: vector of subset sizes p to sample
 - boots: number of bootstraps per p
"""
function id_entropy_global(X::AbstractMatrix{<:Real};
        k::Int=10, γ::Float64=0.5, α::Float64=0.5,
        plist::Vector{Int}=Int[], boots::Int=16, rng::AbstractRNG=Random.GLOBAL_RNG)

    d, N = size(X)
    plist = isempty(plist) ? collect(round.(Int, range(max(32, 2k), N; length=8))) : plist

    idx, dst = knn_distances(X; k=k)

    logp = Float64[]; logL = Float64[]
    for p in plist
        μ = 0.0
        trials = min(boots, max(1, cld(N, p)))
        for _ in 1:trials
            S = sample(rng, 1:N, p; replace=false) |> collect
            μ += L_gamma_subset(S, idx, dst; γ=γ)
        end
        μ /= trials
        push!(logp, log(p))
        push!(logL, log(max(μ, eps())))
    end

    a, b = ls_fit(logp, logL)    # log L̄_γ ≈ a log p + b

    # Growth-law mapping:
    # a = 1 - γ/m   ⇒   m = γ / (1 - a)
    m_hat = γ / max(1e-9, (1.0 - a))

    # Rényi α-entropy offset proxy: H_α ≈ (b - const)/γ  (const cancels in deltas).
    # For routing we only need relative H_α; take H_hat := b/γ.
    H_hat = b / γ

    return m_hat, H_hat, (a=a, b=b, γ=γ, α=α, k=k, plist=plist, boots=boots)
end

"""
LOCAL per-node estimates on r-neighborhood ego clouds.
Returns:
 - m_hat :: Vector{Float64} (length N)
 - H_hat :: Vector{Float64} (length N)
 - diag  :: NamedTuple with shared settings
"""
function id_entropy_local(X::AbstractMatrix{<:Real};
        k::Int=10, r::Int=64, γ::Float64=0.5, α::Float64=0.5,
        boots::Int=8, rng::AbstractRNG=Random.GLOBAL_RNG)

    d, N = size(X)
    idx, dst = knn_distances(X; k=max(k, r))  # build once

    m̂ = fill(Float64(NaN), N)
    Ĥ = fill(Float64(NaN), N)

    # Precompute neighbor rings (r nearest by Euclidean using the same KD tree result)
    for i in 1:N
        # pick the r shortest from idx[i] using the dst[i] distances
        order = sortperm(dst[i])[1:min(r, length(dst[i]))]
        Sfull = vcat(i, idx[i][order]) |> unique |> collect
        P = length(Sfull)
        if P < max(24, 2k)
            # fall back to a coarse global estimate for small clouds
            continue
        end

        # Build a small knn view for Sfull: reuse global idx/dst by masking
        # For simplicity, recompute Lγ on subsets drawn from Sfull using global idx/dst
        # (masking inside L_gamma_subset handles it).
        plist = round.(Int, clamp.(range(ceil(Int, 0.3P), P; length=6), 8, P))
        logp = Float64[]; logL = Float64[]
        for p in plist
            μ = 0.0
            trials = min(boots, max(1, cld(P, p)))
            for _ in 1:trials
                S = sample(rng, Sfull, p; replace=false) |> collect
                μ += L_gamma_subset(S, idx, dst; γ=γ)
            end
            μ /= trials
            push!(logp, log(p))
            push!(logL, log(max(μ, eps())))
        end
        a, b = ls_fit(logp, logL)
        m̂[i] = γ / max(1e-9, (1.0 - a))
        Ĥ[i] = b / γ
    end

    return m̂, Ĥ, (γ=γ, α=α, k=k, r=r, boots=boots)
end

end # module

# QVNM: quantum fidelity + geodesic blend
function quantum_fidelity(V::Matrix{Float32})
    S = transpose(V) * V
    return S .^ 2
end

function geodesic_kernel(nei::Vector{Vector{Int}}, w::Vector{Vector{Float32}})
    N = length(nei)
    g = SimpleWeightedGraph{Int, Float64}(N)
    for i in 1:N
        for (j, wij) in zip(nei[i], w[i])
            add_edge!(g, i, j, Float64(wij))
        end
    end
    D = fill(Inf, N, N)
    for s in 1:N
        dist = dijkstra_shortest_paths(g, s).dists
        D[s, :] = dist
    end
    return D
end

function qvnm_build(obj)
    d = Int(obj["d"]) ; N = Int(obj["N"]) ; Vvec = Vector{Float32}(obj["V"]) ;
    V = reshape(Vvec, d, N)
    mhat = Vector{Float32}(obj["m_hat"]) ; Hhat = Vector{Float32}(obj["H_hat"]) ;
    lam_m = Float64(get(obj, "lambda_m", 0.3)) ; lam_h = Float64(get(obj, "lambda_h", 0.3))

    # neighbors
    nei = [Vector{Int}(x) for x in obj["neighbors"]]
    wts = [Vector{Float32}(x) for x in obj["weights"]]

    # geodesic
    Dg = geodesic_kernel(nei, wts)

    # local sigma
    K = min(10, size(Dg,2)-1)
    sortD = mapslices(sort, Dg; dims=2)[:,1:K+1]
    sigma = [median(sortD[i,2:end]) for i in 1:size(Dg,1)]
    meanH = mean(Hhat); meanM = mean(mhat)
    for i in 1:length(sigma)
        sigma[i] *= exp(lam_h*(Hhat[i]-meanH) - lam_m*(mhat[i]-meanM))
    end

    # quantum fidelity
    colnorms = vec(sqrt.(sum(abs2, V; dims=1)))
    Vn = V ./ (colnorms .+ eps(Float32))
    F = quantum_fidelity(Vn)

    # kernel blend
    Nn = size(Dg,1)
    W = zeros(Float32, Nn, Nn)
    for i in 1:Nn, j in 1:Nn
        if i==j; continue; end
        W[i,j] = exp(- (Dg[i,j]^2) / (sigma[i]*sigma[j])) * F[i,j]
    end
    return Dict("W" => W)
end

function handle_qvnm_build(req::HTTP.Request)
    obj = JSON3.read(String(req.body))
    out = qvnm_build(obj)
    return HTTP.Response(200, JSON3.write(out))
end

function handle_qvnm_estimate_id(req::HTTP.Request)
    obj = JSON3.read(String(req.body))
    d = Int(obj["d"]) ; N = Int(obj["N"]) ; Vvec = Vector{Float32}(obj["V"]) ;
    V = reshape(Vvec, d, N) |> Array{Float64}

    k = Int(get(obj, "k", 10))
    γ = Float64(get(obj, "gamma", 0.5))
    α = Float64(get(obj, "alpha", 0.5))
    boots = Int(get(obj, "boots", 16))
    mode = String(get(obj, "mode", "global"))

    if mode == "global"
        m_hat, H_hat, diag = CostaHeroID.id_entropy_global(V; k=k, γ=γ, α=α, boots=boots)
        out = Dict("mode"=>"global", "m_hat"=>m_hat, "H_hat"=>H_hat, "diag"=>Dict(diag))
        return HTTP.Response(200, JSON3.write(out))
    else
        r = Int(get(obj, "r", 64))
        m_hat, H_hat, diag = CostaHeroID.id_entropy_local(V; k=k, γ=γ, α=α, boots=boots, r=r)
        out = Dict("mode"=>"local", "m_hat"=>m_hat, "H_hat"=>H_hat, "diag"=>Dict(diag))
        return HTTP.Response(200, JSON3.write(out))
    end
end

# Register new routes
function handle(req::HTTP.Request)
    target = String(req.target)
    if req.method == "GET" && target == "/health"
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(Dict("ok" => true)))
    elseif req.method == "POST" && target == "/coherence"
        payload = JSON3.read(String(req.body))
        cr = compute_coherence(payload)
        return HTTP.Response(200, ["Content-Type" => "application/json"], result_to_json(cr))
    elseif req.method == "POST" && target == "/optimize"
        payload = JSON3.read(String(req.body))
        adj_json = get(payload, "adjacency", nothing)
        adj_json === nothing && return HTTP.Response(400, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "missing adjacency")))
        adj = _to_matrix(adj_json)
        mode = get(payload, "mode", "kfp")
        beta = try parse(Float64, string(get(payload, "beta", 0.8))) catch; 0.8 end
        adj2 = kfp_smooth.(adj .* beta)
        result = run_al_uls(adj2; mode=mode, beta=beta)
        return HTTP.Response(200, ["Content-Type" => "application/json"], JSON3.write(Dict(:ok => true, :mode => mode, :beta => beta, :n => size(adj2, 1), :result => result)))
    elseif req.method == "POST" && target == "/qvnm/build"
        return handle_qvnm_build(req)
    elseif req.method == "POST" && target == "/qvnm/estimate_id"
        return handle_qvnm_estimate_id(req)
    else
        return HTTP.Response(404, ["Content-Type" => "application/json"], JSON3.write(Dict("error" => "not found")))
    end
end

function serve(; host::AbstractString = "0.0.0.0", port::Integer = 8099)
    @info "LIMPSBridge listening" host=host port=port
    HTTP.serve(handle, host, port)
end

end # module