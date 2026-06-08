using JSON
using DataFrames
using Dates

__revise_mode__ = :eval

function parse_authors(authors)
    filtered = authors .|> filter(!isempty)
    return map(filtered) do author
        join(author, " ")
    end
end

default_dict = Dict{Symbol,Union{Type,Tuple{Type,<:Function}}}(
    :id => String,
    :categories => (Vector{String}, s -> split(s, ' ')),
    :abstract => (Int, length),
    :update_date => (Date, s -> Date(s)),
    :authors_parsed => (Vector{String}, parse_authors)
)

default_rename = Dict{Symbol,Symbol}(
    :authors_parsed => :authors,
    :update_date => :date,
)

function load_data!(
    df::DataFrame,
    parsed::JSON.LazyValue{String},
    fields::Dict{Symbol,Union{Nothing,<:Function}},
    filters::Nothing=nothing,
)
    result = Dict()
    for (category, parser) in fields
        value = parsed[category][]
        if parser === nothing
            result[category] = value
        else
            result[category] = parser(value)
        end
    end
    push!(df, result)
end

function load_data!(
    df::DataFrame,
    parsed::JSON.LazyValue{String},
    fields::Dict{Symbol,Union{Nothing,<:Function}},
    filters::Dict{Symbol,<:Function}
)
    for (category, filter) in filters
        if !filter(parsed[category][])
            return
        end
    end
    load_data!(df, parsed, fields)
end

function load_data(
    filename::AbstractString;
    fields::Dict{Symbol,Union{Type,Tuple{Type,<:Function}}}=default_dict,
    filters::Union{Dict{Symbol,<:Function},Nothing}=nothing,
    rename::Dict{Symbol,Symbol}=default_rename,
    start_lines::Union{Integer,Nothing}=nothing,
)
    result = DataFrame()
    parsers = Dict{Symbol,Union{<:Function,Nothing}}()
    for (category, parser) in fields
        if parser isa Type
            parsers[category] = nothing
            result[!, category] = Vector{parser}()
        else
            parsers[category] = parser[2]
            result[!, category] = Vector{parser[1]}()
        end
    end

    open(filename) do file
        lines = file |> eachline
        if start_lines !== nothing
            lines = first(lines, start_lines)
        end
        for line in lines
            load_data!(result, JSON.lazy(line), parsers, filters)
        end
    end

    rename!(result, rename)
    return result
end

function hardcoded_load_data(filename::AbstractString)
    result = DataFrame()
    fields = Dict(:categories => String, :update_date => Date)

    for (category, type) in fields
        result[!, category] = Vector{type}()
    end
    result[!, :abstract] = Vector{Int}()
    file = open(filename)

    update = Dict()
    for line in file |> eachline
        parsed = JSON.lazy(line)
        for (category, type) in fields
            update[category] = type(parsed[category][])
        end
        update[:abstract] = length(parsed[:abstract][])
        push!(result, update)
    end

    close(file)
    return result
end
