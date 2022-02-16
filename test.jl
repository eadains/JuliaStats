using DataFrames
using CSV
using Dates
using Turing
using LinearAlgebra
using Statistics

"""
	CalcRV(prices::AbstractVector{<:Number})

Calculates realized volatility from prices vector for a single day:

```math
RV = \\sum_{t=2}^T r_t^2
```
```math
r_t = \\ln(P_t) - \\ln(P_{t-1})
```
Where prices for that day are indexed ``P_1, P_2, ..., P_T``

Equation 1 from:
Barndorff-Nielsen, O. E. (2005). Econometrics of testing for jumps in financial economics using Bipower variation. Journal of Financial Econometrics, 4(1), 1–30. https://doi.org/10.1093/jjfinec/nbi022 
"""
function CalcRV(prices::AbstractVector{<:Number})
    rv = 0
    for t in 2:length(prices)
        rv += (log(prices[t]) - log(prices[t-1]))^2
    end
    return rv
end

"""
	MA(vec::AbstractVector{<:Number}, n::Integer)

Computes the moving average of a given vector over rolling periods of
length n

```math
\\frac{1}{n} \\sum_{t-n}^t x_t
```
"""
function MA(vec::AbstractVector{<:Number}, n::Integer)
    ma = Vector{Float64}(undef, length(vec) - n)
    for t in (n+1):length(vec)
        ma[t-n] = mean(vec[t-n:t])
    end
    return ma
end

"""
	CalcBV(prices::AbstractVector{<:Number})

Calculates bipower variation from prices vector for a single day:

```math
BV = \\sum_{t=3}^T |r_{t-1}||r_t|
```
```math
r_t = \\ln(P_t) - \\ln(P_{t-1})
```
Where prices for that day are indexed ``P_1, P_2, ..., P_T``

Equation 4 from:
Barndorff-Nielsen, O. E. (2005). Econometrics of testing for jumps in financial economics using Bipower variation. Journal of Financial Econometrics, 4(1), 1–30. https://doi.org/10.1093/jjfinec/nbi022 
"""
function CalcBV(prices::AbstractVector{<:Number})
    bv = 0
    for t in 3:length(prices)
        r_t = log(prices[t]) - log(prices[t-1])
        r_t1 = log(prices[t-1]) - log(prices[t-2])
        bv += abs(r_t1) * abs(r_t)
    end
    return bv
end

"""
    CalcQV(prices::AbstractVector{<:Number}, δ::Integer)

Calculates quadpower variation from prices vector representing a single day

```math
QV = \\delta^{-1} \\sum_{t=5}^T |r_{t-3}||r_{t-2}||r_{t-1}||r_t|
```
```math
r_t = \\ln(P_t) - \\ln(P_{t-1})
```
Where prices for that day are indexed ``P_1, P_2, ..., P_T`` and ``\\delta`` is the number of price observations in a single day. For example, using minute data ``\\delta \\approx 390``.

Pg.9 from: Barndorff-Nielsen, O. E. (2005). Econometrics of testing for jumps in financial economics using Bipower variation. Journal of Financial Econometrics, 4(1), 1–30. https://doi.org/10.1093/jjfinec/nbi022
"""
function CalcQV(prices::AbstractVector{<:Number}, δ::Integer)
    qv = 0
    for t in 5:length(prices)
        r_t = log(prices[t]) - log(prices[t-1])
        r_t1 = log(prices[t-1]) - log(prices[t-2])
        r_t2 = log(prices[t-2]) - log(prices[t-3])
        r_t3 = log(prices[t-3]) - log(prices[t-4])
        qv += abs(r_t) * abs(r_t1) * abs(r_t2) * abs(r_t3)
    end
    return δ * qv
end

"""
    JumpStatistic(δ::Integer, RV::Real, BV::Real, QV::Real)

Calculates the ratio jump test statistic
```math
\\hat J = \\frac{\\delta^{1/2}}{\\sqrt{\\theta max(1, \\hat q / \\tilde v^2)}} (\\frac{\\mu^{-2} \\tilde v}{\\hat v} - 1)
```
Where ``\\delta`` is the number of price observations in a day, ``\\theta = \\frac{\\pi^2}{4} + \\pi - 5``, ``\\hat q`` is quadpower variation, ``\\tilde v`` is bipower variation, and ``\\hat v`` is realized variation. This test statistic is asymptotically distributed according to a standard Normal.

See equation 14 in: Barndorff-Nielsen, O. E. (2005). Econometrics of testing for jumps in financial economics using Bipower variation. Journal of Financial Econometrics, 4(1), 1–30. https://doi.org/10.1093/jjfinec/nbi022
"""
function JumpStatistic(δ::Integer, RV::Real, BV::Real, QV::Real)
    θ = π^2 / 4 + π - 5
    μ = √2 / √π
    return (√δ / √(θ * max(1, QV / BV^2))) * (μ^-2 * BV / RV - 1)
end

SPX = DataFrame(CSV.File("./data/SPX_1min.csv", header = ["date", "open", "high", "low", "close", "volume"]))
SPX[!, :date] = Date.(SPX[!, :date], dateformat"y-m-d H:M:S")
SPX = SPX[!, [:date, :close]]

SPXGrouped = groupby(SPX, :date)

vols = combine(SPXGrouped, :close => CalcRV => :RV, :close => CalcBV => :BV, :close => (x -> CalcQV(x, 390)) => :QV)
# Calculate jump statistic for each row, then see if it falls below
# 0.01 quantile of a standard normal, indicating a one-sided confidence
# of 99.9%
transform!(vols, [:RV, :BV, :QV] => ByRow((x, y, z) -> JumpStatistic(390, x, y, z) <= quantile(Normal(0, 1), 0.01)) => :jump)
# Jump magnitude on days where there is a jump
transform!(vols, [:RV, :BV, :jump] => ByRow((rv, bv, j) -> j ? rv - bv : 0) => :magnitude)
# Seperate out continuous variation component. Equal to RV on days without
# jump and equal to BV on days with a jump
transform!(vols, [:RV, :BV, :jump] => ByRow((rv, bv, j) -> j ? bv : rv) => :RCV)

# 1 day ahead volatility
RV_ahead_jump = DataFrame(date = vols[1:end-1, :date], RV_ahead = vols[2:end, :RV])
# Get moving averages of continuous component of volatility
RCV = DataFrame(date = vols[!, :date], RCV = vols[!, :RCV])
RCV_5 = DataFrame(date = vols[6:end, :date], RCV_5 = MA(vols[!, :RCV], 5))
RCV_21 = DataFrame(date = vols[22:end, :date], RCV_21 = MA(vols[!, :RCV], 21))
# Get moving averages of jump magnitude
J = DataFrame(date = vols[!, :date], J = vols[!, :magnitude])
J_5 = DataFrame(date = vols[6:end, :date], J_5 = MA(vols[!, :magnitude], 5))
J_21 = DataFrame(date = vols[22:end, :date], J_21 = MA(vols[!, :magnitude], 21))
jump_data = innerjoin(RV_ahead_jump, RCV, RCV_5, RCV_21, J, J_5, J_21, vols[!, [:date, :jump]], on = :date)

# Log transform data
transform!(jump_data, [:RV_ahead, :RCV, :RCV_5, :RCV_21] .=> (x -> log.(x)) .=> [:RV_ahead, :RCV, :RCV_5, :RCV_21])
transform!(jump_data, [:J, :J_5, :J_21] .=> ByRow((x -> x > 0 ? log(x) : 0)) .=> [:J, :J_5, :J_21])

jump_train_index = round(Int, nrow(jump_data) * 0.70)
jump_train = jump_data[1:jump_train_index-1, :]
jump_test = jump_data[jump_train_index:end, :]

@model function HARJumps(x, y, jumps, ::Type{T} = Vector{Real}) where {T}
    μ₁ ~ Normal(0, 1)
    β₁ ~ Normal(0, 1)
    α₁ ~ Normal(0, 1)
    γ ~ MvNormal(3, 1.0)
    λ = T(undef, size(x, 1))
    λ[1] = μ₁
    for t in 2:length(λ)
        λ[t] = μ₁ + γ ⋅ x[t-1, 1:3] + (β₁ * λ[t-1] + (α₁ * jumps[t-1]))
    end

    α ~ Normal(0, 5)
    β ~ MvNormal(size(x, 2), 5.0)
    θ ~ Normal(0, 5)
    σ ~ truncated(Normal(0, 100), 0, Inf)
    y ~ MvNormal(α .+ x * β .+ θ .* λ, sqrt(σ))
end

jump_model = HARJumps(
    Matrix(select(jump_train, Not([:date, :RV_ahead, :jump]))),
    jump_train[!, :RV_ahead],
    jump_train[!, :jump],
)
jump_chain = sample(jump_model, NUTS(1000, 0.65), MCMCThreads(), 250, 4, progress = true)
print(describe(jump_chain))