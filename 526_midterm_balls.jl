### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# ╔═╡ 44cfed04-51aa-4a07-9903-a25ab5d9f914
function select_balls(num)
	balls = ["r", "r", "r", "b", "b", "b", "w", "w", "w", "w"]
	num_red = 0
	num_black = 0
	num_white = 0
	for i in 1:num
		index = rand(1:length(balls))
		ball = popat!(balls, index)
		if ball == "r"
			num_red += 1
		elseif ball == "b"
			num_black += 1
		elseif ball == "w"
			num_white += 1
		end
	end
	return num_red, num_black, num_white
end

# ╔═╡ 5e18451f-c503-4d67-9965-769b14f575f4
function count_events(num_trials)
	events = 0
	for i in 1:num_trials
		red, black, white = select_balls(5)
		if red >= 1 && black >= 1 && white >= 1
			events += 1
		end
	end
	return events / num_trials
end

# ╔═╡ 06650077-59b3-4e2b-8767-aa4f050d0c46
count_events(10000000)

# ╔═╡ Cell order:
# ╠═44cfed04-51aa-4a07-9903-a25ab5d9f914
# ╠═5e18451f-c503-4d67-9965-769b14f575f4
# ╠═06650077-59b3-4e2b-8767-aa4f050d0c46
