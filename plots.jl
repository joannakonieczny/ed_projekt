using CairoMakie
using DataFrames
using Dates
using Base.Iterators: flatten

# ile wyszło w danym roku
function articles_per_year(df::DataFrame)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    years = map(year, df[!, :date])
    xs = minimum(years):maximum(years)
    ys = zeros(xs)
    for a in years
        ys[a] += 1
    end
    barplot!(ax, ys)
    return fig
end

# jeszcze chłop chciał długości abstraktów
function abstract_lengths(df::DataFrame)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    xs = 1:maximum(df[!, :abstract])
    ys = zeros(xs)
    for a in df[!, :abstract]
        ys[a] += 1
    end
    barplot!(ax, xs, ys)
    return fig
end

# odcina górnych p% auuuuu
function top_percent(values::AbstractVector, p::Real=1)
    # high_element = sort(values, rev=true)[div(p * size(values)[1], 100)]
    high_element = sort(values, rev=true)[p * length(values) / 100 |> round |> Int]
    return values |> filter(x -> x < high_element)
end

# jeszcze chłop chciał długości abstraktów (ale tutaj już odcina 5% 
# najwyższych wartości)
function abstract_lengths_norm(df::DataFrame, top_p::Real=1)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    abstracts = top_percent(df[:, :abstract], top_p)
    xs = 1:maximum(abstracts)
    ys = zeros(xs)
    for a in abstracts
        ys[a] += 1
    end
    barplot!(ax, xs, ys)
    return fig
end

# można też jakieś ile jest autorów per paper
function authors_per_paper(df::DataFrame)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    # TODO jak ?
    return fig
end

# ilość publikacji z daną ilością autorów
function author_amounts(df::DataFrame)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    authors = df[!, :authors] .|> length
    ys = zeros(minimum(authors):maximum(authors))
    for a in authors
        ys[a] += 1
    end
    barplot!(ax, ys)
    return fig
end

# ilość publikacji z daną ilością autorów
function author_amounts_norm(df::DataFrame, top_p::Real=0.5)
    fig = Figure(size=(2560, 1080))
    ax = Axis(fig[1,1])
    authors = top_percent(df[!, :authors] .|> length, top_p)
    ys = zeros(minimum(authors):maximum(authors))
    for a in authors
        ys[a] += 1
    end
    barplot!(ax, ys)
    return fig
end

function zero_pair(obj, zero=0)
    return obj => zero
end

# średnią długość abstraktu w zależności kategorii
function abstract_vs_category(df::DataFrame)
    category_counts = df[:, :categories] |> flatten |> unique .|> (e -> e => [0, 0]) |> Dict
    for (names, alen) in zip(df[:, :categories], df[:, :abstract])
        for name in names
            category_counts[name][1] += alen
            category_counts[name][2] += 1
        end
    end
    categories = category_counts |> collect .|> (p -> p.first => p.second[1] / p.second[2])
    tick_names = categories .|> first
    bar_sizes = categories .|> last
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(tick_names), tick_names),
    )
    barplot!(ax, bar_sizes; direction=:x)
    return fig
end

# średnią długość abstraktu w zależności od kategorii (sort !!)
function abstract_vs_category_sort(df::DataFrame)
    category_counts = df[:, :categories] |> flatten |> unique .|> (e -> e => [0, 0]) |> Dict
    for (names, alen) in zip(df[:, :categories], df[:, :abstract])
        for name in names
            category_counts[name][1] += alen
            category_counts[name][2] += 1
        end
    end
    categories = category_counts |> collect .|> (p -> p.first => p.second[1] / p.second[2])
    sorted_names = sort(categories, by=last)
    tick_names = sorted_names .|> first
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(tick_names), tick_names),
    )
    barplot!(ax, sorted_names .|> last; direction=:x)
    return fig
end

# w stylu, ile jest artykułów w danym typie
function article_types_vert(df::DataFrame)
    name_counts = df[:, :categories] |> flatten |> unique .|> zero_pair |> Dict
    for list in df[:, :categories]
        for name in list
            name_counts[name] += 1
        end
    end
    names = name_counts |> values |> collect
    fig = Figure(size=(2560, 1080))
    ax = Axis(
        fig[1,1],
        xticks=(eachindex(names), name_counts |> keys |> collect),
        xticklabelrotation=pi/2,
    )
    barplot!(ax, names |> eachindex, names)
    return fig
end

# w stylu, ile jest artykułów w danym typie
# (horajzontal barplot auuuuu)
function article_types(df::DataFrame)
    name_counts = df[:, :categories] |> flatten |> unique .|> zero_pair |> Dict
    for list in df[:, :categories]
        for name in list
            name_counts[name] += 1
        end
    end
    names = name_counts |> values |> collect
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(names), name_counts |> keys |> collect),
    )
    barplot!(ax, names; direction=:x)
    return fig
end

# w stylu, ile jest artykułów w danym typie
# (horajzontal barplot sort AUUUUUUU)
function article_types_sort(df::DataFrame)
    name_counts = df[:, :categories] |> flatten |> unique .|> zero_pair |> Dict
    for list in df[:, :categories]
        for name in list
            name_counts[name] += 1
        end
    end
    sorted_names = sort(name_counts |> collect, by=last)
    tick_names = sorted_names .|> first
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(tick_names), tick_names),
    )
    barplot!(ax, sorted_names .|> last; direction=:x)
    return fig
end

# średnią długość abstraktu w zależności od głównej kategorii
# pytanie czy to jest główna kategoria
function main_category(name::AbstractString)
    return eachsplit(name, '.') |> first
end

# średnią długość abstraktu w zależności od głównej kategorii
function abstract_vs_main_category(df::DataFrame)
    categories = map(df[:, :categories]) do list
        map(main_category, list) |> unique
    end
    name_counts = Dict()
    for list in categories
        for name in list
            name_counts[name] = get(name_counts, name, 0) + 1
        end
    end
    tick_names = name_counts |> keys |> collect
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(tick_names), tick_names),
    )
    barplot!(ax, name_counts |> values |> collect; direction=:x)
    return fig
end

# średnią długość abstraktu w zależności od głównej kategorii (sort auuu)
function abstract_vs_main_category_sort(df::DataFrame)
    categories = map(df[:, :categories]) do list
        map(main_category, list) |> unique
    end
    name_counts = Dict()
    for list in categories
        for name in list
            name_counts[name] = get(name_counts, name, 0) + 1
        end
    end
    sorted_names = sort(name_counts |> collect, by=last)
    tick_names = sorted_names .|> first
    fig = Figure(size=(1440, 2560))
    ax = Axis(
        fig[1,1],
        yticks=(eachindex(tick_names), tick_names),
    )
    barplot!(ax, sorted_names .|> last; direction=:x)
    return fig
end
