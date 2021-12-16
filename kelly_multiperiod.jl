### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 7e8666e4-5eb3-11ec-18cc-8d20964cf4e8
md"""
Continuing my series about the Kelly Criterion, in this post I want to explore the multi-period problem in portfolio optimization.

Let's say that I know the distribution of payoffs for some asset in the next month. In the classical Markowitz mean-variance world my goal is to maximize my return given some variance constraint. Okay, so I run my optimization over the next month and move on. And that would be fine, given that I only cared about my investment over that month and no others. But obviously I'm going to invest my money the month after that, and so on. And it turns out, in the mean-variance world, optimizing over each month independently doesn't result in the most optimal outcome overall.

Let's see how that works.
"""

# ╔═╡ 74939155-c2c4-41bd-9cf7-b4fd834b1f2f
md"""
# The Multi-Period Problem in the Mean-Variance World


"""

# ╔═╡ Cell order:
# ╟─7e8666e4-5eb3-11ec-18cc-8d20964cf4e8
# ╠═74939155-c2c4-41bd-9cf7-b4fd834b1f2f
