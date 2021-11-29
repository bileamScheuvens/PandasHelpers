module PandasHelpers

using DataFrames
using Random
using Statistics
using StatsBase

export get_dummies!, abs_max_norm!, train_test_split, naive_impute!, naive_impute

"""

`get_dummies!(df::DataFrame, columns::Vector{String})` 

`get_dummies!(df::DataFrame, columns::String)`

Dummy encode columns inplace

Example:

`df = DataFrame("color"=>["green","red"])`

|   | color |
|---|-------|
| 1 | green |
| 2 |  red  |

`get_dummies!(df, "food")`

|   | color_green | color_red |
|---|-------------|-----------|
| 1 |      1      |    0      |
| 2 |      0      |    1      |

"""
function get_dummies!(df::DataFrame, columns::Vector{String})
    for col in columns
        un = unique(df[!, col])
        transform!(df, @. col => ByRow(isequal(un)) .=> Symbol(col,"_", un))
        select!(df, Not(col))
    end
end
function get_dummies!(df::DataFrame, columns::String)
    col = columns
    un = unique(df[!, col])
    transform!(df, @. col => ByRow(isequal(un)) .=> Symbol(col,"_", un))
    select!(df, Not(col))
end

"""

`abs_max_norm!(df::DataFrame, columns::String)`

`abs_max_norm!(df::DataFrame, columns::Vector{String})`

Abs-max Normalize by dividing every entry by the absolute maximum of the column.

"""
function abs_max_norm!(df::DataFrame, columns::String)
    col = columns
    df[!, col] = df[!, col] ./ maximum(broadcast(abs, df[!, col]))
end
function abs_max_norm!(df::DataFrame, columns::Vector{String})
    for col in columns
        df[!, col] = df[!, col] ./ maximum(broadcast(abs, df[!, col]))
    end
end

"""

`train_test_split(df::DataFrame, target::String; ratio::Number, shuffled::Bool)`

perform train test split

returns: `train_x, train_y, test_x, test_y`

**Keywords**

`ratio`: percentage of samples in training set, default 0.8

`shuffled`: whether to shuffle rows, default true

"""
function train_test_split(df::DataFrame, target::String; ratio::Number=0.8, shuffled::Bool=true)
    df_copy = shuffled ? df[shuffle(axes(df, 1)), :] : df
    cutoff = Int(round(ratio * size(df_copy)[1]))
    cutoff_inv = Int(round((1-ratio) * size(df_copy)[1]))
    train_x = first(df_copy[:, Not(target)], cutoff)
    train_y = first(df_copy[:, target], cutoff)
    test_x = last(df_copy[:, Not(target)], cutoff_inv)
    test_y = last(df_copy[:, target], cutoff_inv)
    return train_x, train_y, test_x, test_y
end
"""

`naive_impute!(col)`

impute missing values inplace, for eltype <: Number use mean, else mode

"""
function naive_impute!(col)
    impute_value = eltype(col) <: Union{Missing, Number} ? mean(skipmissing(col)) : mode(skipmissing(col))
    replace!(col, missing=>impute_value)
end
"""

`naive_impute!(col)`

impute missing values, for eltype <: Number use mean, else mode

returns: col

"""
function naive_impute(col)
    impute_value = eltype(col) <: Union{Missing, Number} ? mean(skipmissing(col)) : mode(skipmissing(col))
    replace(col, missing=>impute_value)
end

end
