### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 24708695-ee16-40dd-9a27-81ffd246dcbc
begin
	using Distributions
	using Statistics
end

# ╔═╡ 7e8666e4-5eb3-11ec-18cc-8d20964cf4e8
md"""
Continuing my series about the Kelly Criterion, in this post I want to explore the multi-period problem in portfolio optimization.

Let's say that I know the distribution of payoffs for some asset in the next month. In the classical Markowitz mean-variance world my goal is to minimize my variance subject to a return constraint. Okay, so I run my optimization over the next month and move on. And that would be fine, given that I only cared about my investment over that month and no others. But obviously I'm going to invest my money the month after that, and so on. And it turns out, in the mean-variance world, optimizing over each month independently doesn't result in the most optimal outcome overall.

Let's see how that works.
"""

# ╔═╡ 74939155-c2c4-41bd-9cf7-b4fd834b1f2f
md"""
# The Multi-Period Problem in the Mean-Variance World

Let the return of our portfolio be represented by a random variable ``X_t`` where ``t`` represents months. Then the total return of the portfolio over ``n`` periods is:

```math
r_n = \prod_{k=1}^n (1 + X_k)
```

So the goal, then, is to minimize the volatility of this terminal return subject to a return constraint.

```math
\min \sigma_n^2
```
```math
\mu_n = \mu
```

Where
```math
\mu_n = \mathbf{E}[r_n] = \mathbf{E}[\prod_{k=1}^n (1 + X_k)] = \prod_{k=1}^n \mathbf{E}[1 + X_k] = \prod_{k=1}^n \mathbf{E}[1] + \mathbf{E}[X_k] = \prod_{k=1}^n 1 + \mathbf{E}[X_k]
```

Moving the expectation inside the product holds if we assume that the return of each month is independent, which is somewhat reasonable depending on how much you believe in efficient markets. This implies that the mean of our terminal return is the means of each period multiplied together, so no problems yet.

And
```math
\sigma_n^2 = Var(r_n) = \mathbf{E}[r_n^2] - \mathbf{E}[r_n]^2 = \mathbf{E}[\prod_{k=1}^n (1 + X_k)^2] - \mu_n^2
```

The problem here is
```math
\mathbf{E}[\prod_{k=1}^n (1 + X_k)^2] \neq \prod_{k=1}^n \mathbf{E}[(1 + X_k)^2]
```
So the variance over ``n`` periods is not equal to the variance of each period multiplied together. Meaning that we cannot naively optimize over each monthly period and expect the variance of our terminal return to also be optimal.

We can see each of these statments easily with Monte Carlo methods, as usual!
"""

# ╔═╡ d8167fb7-af1e-4ebc-a1b3-c031c9c4b90c
md"""
So here we are gonna let ``n=5`` and each ``X_t`` is going to be distributed normally with varying means and standard deviations. The below generates 100,000 samples of ``1+X_k`` from the products above.
"""

# ╔═╡ fe01b759-32f1-4d11-914a-593dfa9f34c4
begin
	means = [1, 2, 3, 4, 5]
	stds = [1, 0.10, 0.50, 2, 5]
	returns = zeros(length(means), 100000)
	for x in 1:100000
		for k in eachindex(means)
			returns[k, x] = 1 + rand(Normal(means[k], stds[k]))
		end
	end
end

# ╔═╡ aa32b66e-3387-442b-b0ca-cc999e2760fc
md"""
Then by taking the mean of all the products for each sample, we get our Monte Carlo estimate of the mean.
"""

# ╔═╡ 065f3e70-3af6-4342-97a3-f5fd943a5571
begin
	products = mean(prod(returns, dims=1))
	products
end

# ╔═╡ 4069dd79-64c9-4730-81f5-11a3c13acecb
md"""
We can then compare that to the product of the expecations of each of the random variables (plus 1). And that expectation is just the mean of each variable, as specified above. So the theoretical mean is 720 and our estimate is quite close to that. So moving the expectation operator inside the product works when the variables are independent!
"""

# ╔═╡ f0ef7958-7f55-407f-bc07-cbe13a2b2335
prod(means .+ 1)

# ╔═╡ acd01e4f-343d-475e-a7d1-4a5e3b640645
md"""
Now we can do the same process for the variance. This time we take the product of the squares, as in the equation above. So here we can see the variance of our samples computed using the expectation equation.
"""

# ╔═╡ f803681e-bfd6-4a0f-80ec-d7430f24b372
mean(prod(returns.^2, dims=1)) - products^2

# ╔═╡ 37a91165-4fb8-43fb-8f2d-d1766b53078b
md"""
The problem comes when we look at the right side of that last equation.

```math
\prod_{k=1}^n \mathbf{E}[(1 + X_k)^2] = \prod_{k=1}^n \mathbf{E}[X_k^2 + 2X_k+1] = \prod_{k=1}^n \mathbf{E}[X_k^2] + 2\mathbf{E}[X_k] + 1
```

With

```math
\mathbf{E}[X_k^2] = Var(X_k) + \mathbf{E}[X_k]^2
```

from the expectation equation for variance.
"""

# ╔═╡ 71b61639-251e-4d38-89d0-6910306e429e
prod(stds.^2 + means.^2 + 2 * means .+ 1)

# ╔═╡ 441b8f76-69c9-41b9-a5d9-715101429b66
md"""
And that does not match with the Monte Carlo estimate from above.

Conclusion, then. If we care about the variance of our terminal wealth, we cannot minimize variance over each period and expect the minimum variance *overall*. In the language of dynamic programming, the variance is non-separable.
"""

# ╔═╡ 26d45493-b92b-473d-b6b1-ced0b3a60245
md"""
# The Kelly Criterion

The Kelly Criterion relieves all of these problems.

If we take the equation for terminal return again but instead consider the logarithm:

```math
\log r_n = \log(\prod_{k=1}^n (1 + X_k)) = \sum_{k=1}^n \log (1 + X_k)
```

The goal of the Kelly Criterion is to maximize the logarithm of terminal wealth, which turns the multiplication into addition. This fact makes it trivial to take the expectation:

```math
\mathbf{E}[\sum_{k=1}^n \log (1 + X_k)] = \sum_{k=1}^n \mathbf{E}[\log (1 + X_k)]
```

So we can easily see that the expectation of terminal wealth is simply the sum of expectations of each period. Here, because we don't care about variance, we *can* naively optimize over each period and expect the most optimal outcome overall.

As an additional benefit, because we are now summing random variables and not multiplying them, this equation still holds even for correlated variables, so we don't need the assumption of independence like we did above.

Because of the log, this is harder to show analytically, but can be shown by Monte Carlo! You can see below that taking the expectation of the sum is equivalent to taking the sum of the expectations. (Complex numbers because some of the "returns" generated above are negative, this wouldn't obviously happen in practice)
"""

# ╔═╡ 3a1fdaee-0172-4e80-85a0-b38ce60206e6
mean(sum(log.(Complex.(returns)), dims=1))

# ╔═╡ ffafeb57-a8d5-4766-93ba-74bb31ac89be
sum(mean(log.(Complex.(returns)), dims=2))

# ╔═╡ 2d94164d-f1d6-4d18-ada5-87359a65e982
md"""
Now, let's introduce some correlation between our random variables. Aside from ``k=1``, each variable is now its random component plus ``1.10*X_{k-1}``
"""

# ╔═╡ ea43b857-5576-4f80-979b-9de60a6188a0
begin
	returns_corr = zeros(length(means), 100000)
	for k in eachindex(means)
		if k == 1
			returns_corr[k, :] = returns[k, :]
		else
			returns_corr[k, :] = 1.10 * returns[k-1, :]
		end
	end
end

# ╔═╡ 25f7e6fc-f796-4c51-b622-a7493710e3f4
md"""
Here, you can see that the expectation of the products is now different from above.
"""

# ╔═╡ 5d7a0179-338f-46a9-93d3-b0f70ec213e9
mean(prod(returns_corr, dims=1))

# ╔═╡ 374dba6b-274f-4c38-b71d-061d7b1082f9
md"""
However, the sum equations for the expectation of the logarithms of the returns still match. So correlation does not effect the optimality of the Kelly Criterion.
"""

# ╔═╡ 4183c709-10a2-4843-9680-40b04344c103
mean(sum(log.(Complex.(returns_corr)), dims=1))

# ╔═╡ 30999e9a-5cfd-445c-ad2f-d0a8388624f7
sum(mean(log.(Complex.(returns_corr)), dims=2))

# ╔═╡ 64fa59d1-e9c9-4227-a565-54cabebb9dce
md"""
# Conclusion

This post reveals one of the most powerful facts about the Kelly Criterion: it's myopic. You can naively optimize over each time period independently, and you automatically end up with the most optimal outcome overall. In the mean-variance framework, if you want the most optimal portfolio over a year, you need to know the distribution for each rebalancing period in advance. With the Kelly Criterion, you can forecast one period ahead, optimize over that period, and move on.

There has been a lot of work done in expanding the mean-variance methodology to a multi-period framework, but it requires a lot of complicated math. [This paper goes over a possible method](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.34.1625&rep=rep1&type=pdf). Although it still assumes independent returns across time periods. Later work does relax that assumption, but things become even more complicated.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.36"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c1724611e6ae29c6094c8d9850e3136297ba7fff"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.36"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─7e8666e4-5eb3-11ec-18cc-8d20964cf4e8
# ╟─74939155-c2c4-41bd-9cf7-b4fd834b1f2f
# ╠═24708695-ee16-40dd-9a27-81ffd246dcbc
# ╟─d8167fb7-af1e-4ebc-a1b3-c031c9c4b90c
# ╠═fe01b759-32f1-4d11-914a-593dfa9f34c4
# ╟─aa32b66e-3387-442b-b0ca-cc999e2760fc
# ╠═065f3e70-3af6-4342-97a3-f5fd943a5571
# ╟─4069dd79-64c9-4730-81f5-11a3c13acecb
# ╠═f0ef7958-7f55-407f-bc07-cbe13a2b2335
# ╟─acd01e4f-343d-475e-a7d1-4a5e3b640645
# ╠═f803681e-bfd6-4a0f-80ec-d7430f24b372
# ╟─37a91165-4fb8-43fb-8f2d-d1766b53078b
# ╠═71b61639-251e-4d38-89d0-6910306e429e
# ╟─441b8f76-69c9-41b9-a5d9-715101429b66
# ╟─26d45493-b92b-473d-b6b1-ced0b3a60245
# ╠═3a1fdaee-0172-4e80-85a0-b38ce60206e6
# ╠═ffafeb57-a8d5-4766-93ba-74bb31ac89be
# ╟─2d94164d-f1d6-4d18-ada5-87359a65e982
# ╠═ea43b857-5576-4f80-979b-9de60a6188a0
# ╟─25f7e6fc-f796-4c51-b622-a7493710e3f4
# ╠═5d7a0179-338f-46a9-93d3-b0f70ec213e9
# ╟─374dba6b-274f-4c38-b71d-061d7b1082f9
# ╠═4183c709-10a2-4843-9680-40b04344c103
# ╠═30999e9a-5cfd-445c-ad2f-d0a8388624f7
# ╟─64fa59d1-e9c9-4227-a565-54cabebb9dce
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
