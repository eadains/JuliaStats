### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ cd5458d2-fd67-11ec-2731-ed59483eac64
begin
	using CSV
	using DataFrames

	slate = CSV.read("./data/slate.csv", DataFrame)
end

# ╔═╡ c1be1ff7-c910-45d5-b124-0525d5f5e6cd
begin
	using JuMP
	using GLPK

	function do_optim(slate::DataFrame)
		model = Model(GLPK.Optimizer)
		
		# G_h
	    games = unique(slate.Game)
		# T_l
	    teams = unique(slate.Team)
	
	    # x_i
	    @variable(model, x[slate.ID], binary = true)
	    # y_l
	    @variable(model, y[teams], binary = true)
	    # z_h
	    @variable(model, z[games], binary = true)
		
	    # Equation 2
		@constraint(model, sum(player.Salary * x[player.ID] for player in eachrow(slate)) <= 35000)
	    # Equation 3
	    @constraint(model, sum(x) == 9)
		# Equation 4
		@constraint(model, sum(x[player.ID] for player in eachrow(slate) if player.Position == "P") == 1)
		# Equation 5
		@constraint(model, 1 <= sum(x[player.ID] for player in eachrow(slate) if player.Position == "C/1B") <= 2)
		# Equation 6
		@constraint(model, 1 <= sum(x[player.ID] for player in eachrow(slate) if player.Position == "2B") <= 2)
		# Equation 7
		@constraint(model, 1 <= sum(x[player.ID] for player in eachrow(slate) if player.Position == "3B") <= 2)
		# Equation 8
		@constraint(model, 1 <= sum(x[player.ID] for player in eachrow(slate) if player.Position == "SS") <= 2)
		# Equation 9
		@constraint(model, 3 <= sum(x[player.ID] for player in eachrow(slate) if player.Position == "OF") <= 4)
	
	    for team in teams
	        # Equation 10
			@constraint(model, sum(x[player.ID] for player in eachrow(slate) if player.Team == team && player.Position != "P") <= 4)
	        # Equation 12
			@constraint(model, y[team] <= sum(x[player.ID] for player in eachrow(slate) if player.Team == team))
	    end
	    # Equation 13
	    @constraint(model, sum(y) >= 3)
	
	    for game in games
	        # Equation 15
			@constraint(model, z[game] <= sum(x[player.ID] for player in eachrow(slate) if player.Game == game))
	    end
	    # Equation 16
	    @constraint(model, sum(z) >= 2)
	
	    # Equation 17
		@objective(model, Max, sum(player.Projection * x[player.ID] for player in eachrow(slate)))
	
	    optimize!(model)
	    println(termination_status(model))
		# Round to integer to ensure values are exactly 1 or 0 to avoid
		# issues with possible machine zeros
	    return round.(Int, value.(x)).data
	end
end

# ╔═╡ 087ed059-4ea7-4554-9f41-dce0169ffbd9
md"""
I wanted to try something new outside the financial realm, so I decided I would give daily fantasy sports (DFS) a shot! It turns out to be a very interesting problem of its own, so in this post I want to cover the basics of using integer programming to find optimal DFS teams. I'm specifically going to cover baseball on FanDuel here, but the ideas can easily be applied to other sports and websites.
"""

# ╔═╡ d76c0ef7-8014-42f6-b4df-9fb8ae6dbd10
md"""
# Data

First thing, of course is getting data. I export the CSV of availiable players from FanDuel, and then join that information with point projections from [LineStar](https://www.linestarapp.com/). FanDuel has restrictions on the lineups you can form that I'll cover later, so for each player we need to know the following information: name, team, game, salary, and position. I drop any players that are not the starting pitcher, or that are not playing for whatever reason. I also handle players that can fill multiple positions by assuming that they can only fill the first position listed. For instance if a player has "2B/OF" as their position, they will appear in the table as only being "2B". Given that FanDuel provides the Utility slot, I assume that this has little effect on the optimality of the lineups generated. The data looks like this all said and done:
"""

# ╔═╡ 2adc4ecc-e3e7-4281-8512-853a309f433b
md"""
# The Optimization

I'm going to focus on generating the lineup with the highest expected number of points.

So, given the objective, MLB lineups on FanDuel are constrained in the following ways:
- Salary must be less than or equal to \$35,000
- Must pick 9 players with the following positions:
    - 1 pitcher
    - 1 catcher / first baseman
    - 1 second baseman
    - 1 third baseman
    - 1 shortstop
    - 3 outfielders
    - 1 non-pitcher player to fill the utility slot
- Must not select more than 4 players from each team, excluding the pitcher. This means we can technically have 5 players from one team, if one of them is the pitcher.
- Must select players from at least 3 different teams
- Must select players from at least 2 different games

In mathematical notation:
```math
\begin{align}
x_i \in \{0, 1\} &\; \forall i \in \{1, 2, \dots, N_p\} \tag{1} \\
\sum_{i=1}^{N_p} &x_i c_i \leq 35000 \tag{2} \\
\sum_{i=1}^{N_p} &x_i = 9 \tag{3} \\
\sum_{j \in P} &x_j = 1 \tag{4} \\
1 \leq \sum_{j \in C} &x_j \leq 2 \tag{5} \\
1 \leq \sum_{j \in 2B} &x_j \leq 2 \tag{6} \\
1 \leq \sum_{j \in 3B} &x_j \leq 2 \tag{7} \\
1 \leq \sum_{j \in SS} &x_j \leq 2 \tag{8} \\
3 \leq \sum_{j \in OF} &x_j \leq 4 \tag{9} \\
\sum_{j \in T_l \cup P^c} &x_j \leq 4 \quad \forall l \in \{1, 2, \dots, N_T\} \tag{10} \\
&y_l \in \{0, 1\} \tag{11} \\
&y_l \leq \sum_{j \in T_l} x_j \tag{12} \\
\sum_{l=1}^{N_T} &y_l \ge 3 \tag{13} \\
&z_h \in \{0, 1\} \tag{14} \\
&z_h \leq \sum_{j \in G_h} x_j \quad \forall h \in \{1, 2, \dots, N_G\} \tag{15} \\
\sum_{h=1}^{N_G} &z_h \ge 2 \tag{16}
\end{align}
```

Where in equation 1 $x_i$ denotes a boolean variable corresponding to each player with $N_p$ denoting the total number of players.

Equation 2 is the salary constraint where $c_i$ denotes the salary of player $i$. 

Equation 3 ensures we select 9 players in total.

Equations 4 through 9 are constraints ensuring we pick the right number of players for each position. Other than the pitcher, where there can only be 1, we can select up to 1 more of the other positions because the duplicate can fill the utility slot.

Equation 10 specifies the constraint where we can only select 4 players from one team, excluding the pitcher, where $T_l$ denotes each team with $N_T$ teams in total. We exclude the pitcher by summing over players on team $T_l$ and who are not in the pitcher set (union with the complement).

Equations 11 through 13 specify the constraint that we must select players from at least 3 teams. This is achieved by specifying a new boolean variable $y_l$ corresponding to each team, and ensuring that it's less than or equal to the number of selected players on that team. This effectively forces $y_l$ to be zero when we haven't selected any players from team $l$. Then, by adding the constraint in equation 13 that the $y_l$'s must sum to at least 3, we ensure that at least 3 $y_l$ values can be set to 1, implying that we have selected players from at least 3 teams.

A similar approach is taken in equations 14 through 16 to ensure that we have selected players from at least 2 games.

And obviously our objective function is:
```math
\text{maximize} \sum_{i=1}^{N_p} x_i p_i \tag{17}
```
where $p_i$ is the projected number of points for player $i$

Now for the code! I'm implemented this in [JuMP](https://jump.dev/JuMP.jl/stable/), a modeling language in Julia for writing and solving mathematical optimization problems. I've noted in comments what equation each line corresponds to.
"""

# ╔═╡ 87a6da16-9d51-44e1-8942-b1697db3916b
md"""
# Results

So given the slate of players, we can run the optimization and get back a vector of integers, representing the $x_i$'s. A 1 means the player was selected and 0 means they were not selected
"""

# ╔═╡ 3d833a01-9e0c-4b0b-8b6d-3100358d1f52
x = do_optim(slate)

# ╔═╡ ff1bb0de-37c3-42dc-aee1-5af7cb14471b
md"""
Here we can see the team the optimization selected. It looks like another outfielder is filling the utility slot in this case, but you can see that it meets all the position requirements.
"""

# ╔═╡ dbe40aeb-e104-4efa-b2ee-6ef78ddaeeae
team = slate[Bool.(x), :]

# ╔═╡ 0a93ac9a-7878-4916-9959-d86ff2ab391b
md"""
We can get the estimate of how many points the team will score:
"""

# ╔═╡ 9596aae7-c505-4852-9612-5bb5fa229d3c
sum(team[!, :Projection])

# ╔═╡ f66655b2-5d31-4236-8a28-93bb029514cf
md"""
And we can see that it used the salary allowance exactly:
"""

# ╔═╡ cf8c3dda-a15b-4b2f-bf30-9a99a46622f2
sum(team[!, :Salary])

# ╔═╡ 2747b53f-6352-4a59-93ae-c26465e5bd46
md"""
# Conclusion

This was a brief introduction to integer programming and its application to Daily Fantasy Sports. The optimization is very simple once you figure out the constraints, but nonetheless, I've been having success using lineups generated this way in cash contests. In a future post I want to discuss the much more interesting problem of generating lineups for top-heavy contest formats where maximizing expectation is not the most optimal strategy.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
GLPK = "~1.0.1"
JuMP = "~1.1.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "2dd813e5f2f7eec2d1268c57cf2373d3ee91fcea"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.1"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "c3cc0a7a4e021620f1c0e67679acdbf1be311eb0"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.0.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "534adddf607222b34a0a9bba812248a487ab22b7"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.1.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "10d26d62dab815306bbd2c46eb52460e98f01e46"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.6.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─087ed059-4ea7-4554-9f41-dce0169ffbd9
# ╟─d76c0ef7-8014-42f6-b4df-9fb8ae6dbd10
# ╠═cd5458d2-fd67-11ec-2731-ed59483eac64
# ╟─2adc4ecc-e3e7-4281-8512-853a309f433b
# ╠═c1be1ff7-c910-45d5-b124-0525d5f5e6cd
# ╟─87a6da16-9d51-44e1-8942-b1697db3916b
# ╠═3d833a01-9e0c-4b0b-8b6d-3100358d1f52
# ╟─ff1bb0de-37c3-42dc-aee1-5af7cb14471b
# ╠═dbe40aeb-e104-4efa-b2ee-6ef78ddaeeae
# ╟─0a93ac9a-7878-4916-9959-d86ff2ab391b
# ╠═9596aae7-c505-4852-9612-5bb5fa229d3c
# ╟─f66655b2-5d31-4236-8a28-93bb029514cf
# ╠═cf8c3dda-a15b-4b2f-bf30-9a99a46622f2
# ╟─2747b53f-6352-4a59-93ae-c26465e5bd46
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

function garch_gen(
	ϵ::AbstractVector{<:Real},
	ω::Real,
	α::AbstractVector{<:Real},
	β::AbstractVector{<:Real})
end