using Flux
using LinearAlgebra
using Statistics
using Random
using Optimisers
using Zygote

# ============================================================
# DATASET
# ============================================================

struct DataSet
    docs::Vector{Vector{Int}}
    labels::Vector{Int}
end

# ============================================================
# EMBEDDER
# ============================================================

struct FluxEmbedder
    embedding::Flux.Embedding
    projection::Chain
end

Flux.@functor FluxEmbedder

function FluxEmbedder(vocab_size::Int, dim::Int=256)
    FluxEmbedder(
        Flux.Embedding(vocab_size, dim),
        Chain(
            Dense(dim, dim, relu),
            Dense(dim, dim)
        )
    )
end

function embed(model::FluxEmbedder, tokens::Vector{Int})

    x = model.embedding(tokens)

    v = sum(x, dims=2) ./ size(x, 2)
    v = vec(v)

    v = model.projection(v)

    return v ./ (norm(v) + 1e-8)
end

# ============================================================
# TOPIC MODEL (FINAL)
# ============================================================

struct TopicModel
    embedder::FluxEmbedder
    topic_centroids::Vector{Vector{Float32}}
    topic_docs::Vector{Vector{Int}}
end

Flux.@functor TopicModel

# ============================================================
# COSINE
# ============================================================

function cosine(a, b)
    a = vec(a)
    b = vec(b)
    dot(a, b) / (norm(a) * norm(b) + 1e-8)
end

# ============================================================
# EMBEDDINGS
# ============================================================

function build_embeddings(embedder, docs)
    [Float32.(embed(embedder, d)) for d in docs]
end

# ============================================================
# SIMPLE CLUSTERING (BERTOPIC-LIKE)
# ============================================================

function discover_topics(embeddings; threshold=0.75)

    topics = Vector{Vector{Int}}()

    for i in eachindex(embeddings)

        assigned = false

        for t in topics

            centroid = mean(reduce(hcat, embeddings[t]), dims=2)
            centroid = vec(centroid)

            score = cosine(embeddings[i], centroid)

            if score > threshold
                push!(t, i)
                assigned = true
                break
            end
        end

        if !assigned
            push!(topics, [i])
        end
    end

    return topics
end

# ============================================================
# CENTROIDS
# ============================================================

function build_topic_centroids(embeddings, topics)

    centroids = Vector{Vector{Float32}}()

    for t in topics
        c = mean(reduce(hcat, embeddings[t]), dims=2)
        c = vec(c)
        c ./= norm(c) + 1e-8
        push!(centroids, Float32.(c))
    end

    return centroids
end

# ============================================================
# ASSIGN TOPIC
# ============================================================

function assign_topic(model::TopicModel, tokens)

    emb = embed(model.embedder, tokens)

    best = 0
    best_score = -Inf

    for (i, c) in enumerate(model.topic_centroids)

        s = cosine(emb, c)

        if s > best_score
            best_score = s
            best = i
        end
    end

    return best, best_score
end

# ============================================================
# TRAIN (ONLY EMBEDDER)
# ============================================================

function contrastive_loss(embedder, a, b, c; margin=0.2f0)

    ea = embed(embedder, a)
    eb = embed(embedder, b)
    ec = embed(embedder, c)

    pos = cosine(ea, eb)
    neg = cosine(ea, ec)

    return Float32(max(0f0, neg - pos + margin))
end

function train!(model, pos_pairs, neg_pairs; epochs=10, lr=1e-3)

    opt = Flux.setup(Optimisers.Adam(lr), model)

    n = min(length(pos_pairs), length(neg_pairs))
    batch_size = 32

    for e in 1:epochs

        idx = randperm(n)
        loss_sum = 0f0

        for start in 1:batch_size:n

            batch_idx = idx[start:min(start+batch_size-1, n)]

            loss, grads = Flux.withgradient(model) do m
                l = 0f0

                for j in batch_idx
                    a, b = pos_pairs[j]
                    _, c = neg_pairs[j]

                    l += contrastive_loss(m, a, b, c)
                end

                l / length(batch_idx)
            end

            opt, model = Flux.update!(opt, model, grads[1])
            loss_sum += loss
        end

        println("Epoch $e loss = ", loss_sum / ceil(n / batch_size))
    end

    return model
end

# ============================================================
# PAIRS
# ============================================================

function build_pairs(docs, labels; max_pairs=50_000)

    pos = Tuple[]
    neg = Tuple[]

    n = length(docs)

    for _ in 1:max_pairs

        i = rand(1:n)
        j = rand(1:n)

        if i == j
            continue
        end

        if labels[i] == labels[j]
            push!(pos, (docs[i], docs[j]))
        else
            push!(neg, (docs[i], docs[j]))
        end
    end

    return pos, neg
end

# ============================================================
# EVALUATION
# ============================================================

function evaluate(model::TopicModel, test_docs, test_labels)

    correct = 0

    for i in eachindex(test_docs)

        pred, _ = assign_topic(model, test_docs[i])

        if pred == test_labels[i]
            correct += 1
        end
    end

    correct / length(test_docs)
end

# ============================================================
# PIPELINE (FINAL FIXED)
# ============================================================

function run_pipeline(dataset::DataSet;
    vocab_size=30000,
    embed_dim=256,
    train_ratio=0.8,
    seed=42)

    println("Preparing data...")

    n = length(dataset.docs)
    idx = shuffle(collect(1:n))

    split = Int(floor(train_ratio * n))

    train_docs = dataset.docs[idx[1:split]]
    test_docs  = dataset.docs[idx[split+1:end]]

    train_labels = dataset.labels[idx[1:split]]
    test_labels  = dataset.labels[idx[split+1:end]]

    println("Init embedder...")

    embedder = FluxEmbedder(vocab_size, embed_dim)

    println("Building pairs...")

    pos, neg = build_pairs(train_docs, train_labels)

    println("Training embedder...")

    embedder = train!(embedder, pos, neg)

    println("Building embeddings...")

    embs = build_embeddings(embedder, train_docs)

    println("Discovering topics...")

    topics = discover_topics(embs)

    println("Building centroids...")

    centroids = build_topic_centroids(embs, topics)

    model = TopicModel(embedder, centroids, topics)

    println("Evaluating...")

    acc = evaluate(model, test_docs, test_labels)

    println("Accuracy = ", acc)

    return model
end