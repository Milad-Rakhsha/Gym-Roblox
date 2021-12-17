local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
RunService.PhysicsPaused = true
local DT=0.2
local port = 8810

local NUM_ENV=Vector3.new(1,1,1)
local Dx=100
local Dy=0
local Dz=100
local Dr= Vector3.new(Dx,Dy,Dz)
local Collision_margin=3


local Env
local sampleAgent
local Agents_list={}
local Envs_list={}
local Envs_center={}


local p_norm = 1
local success_goal_reward= -0.1
local eps_rew=0
local max_steps=100

bool_to_number={ [true]=1, [false]=0 }

DEFAULT_REWARD = 0
BOUNDARY_SIDE_LENGTH = 80
BOUNDARY_HEIGHT = 20
VELOCITY_LIMITS = { 10, 10 }
ANGULARVELOCITY_LIMITS = { -5, 5 }

_G.XLimits = { -BOUNDARY_SIDE_LENGTH/2, BOUNDARY_SIDE_LENGTH/2 }
_G.ZLimits = { -BOUNDARY_SIDE_LENGTH/2, BOUNDARY_SIDE_LENGTH/2 }
_G.EpisodeReward = 0

local adversary_color=BrickColor.White()
local agent_color=BrickColor.Blue()
local leaderColor = BrickColor.new("Navy blue")


local function distance(entityA, entityB)

	return (entityA.Position - entityB.Position).Magnitude
end

local function setAgentColor(agent, bodyColor)
	agent["Body Colors"].HeadColor=bodyColor
	agent["Body Colors"].LeftArmColor=bodyColor
	agent["Body Colors"].LeftLegColor=bodyColor
	agent["Body Colors"].RightArmColor=bodyColor
	agent["Body Colors"].RightLegColor=bodyColor
	agent["Body Colors"].TorsoColor=bodyColor
end


local function make_object(env, center, obj, name, parent, height)
	local obj=obj:Clone()
	obj.Parent= parent
	obj.Name= name
	local pos=Vector3.new(
		math.random(center.X+_G.XLimits[1], center.X+_G.XLimits[2]),
		height,
		math.random(center.Z+_G.ZLimits[1], center.Z+_G.ZLimits[2])
	)
	--obj:SetPrimaryPartCFrame (CFrame.new(pos))
	obj.CFrame=CFrame.new(pos)
	obj:SetAttribute("Collectable", true)

	obj.Anchored=true
	obj.CanCollide=false
end

local function make_agent(env, center, blind, silent, u_range, comm, adversary, name, leader)
	local agent=ReplicatedStorage:WaitForChild('NPC'):Clone()
	agent.Humanoid.WalkSpeed=u_range
	agent.Humanoid.DisplayName=name
	agent:SetAttribute("Blind", blind)
	agent:SetAttribute("Silent", silent)
	agent:SetAttribute("Leader", leader)
	agent:SetAttribute("U_range", u_range)
	agent:SetAttribute("Comm", comm)
	agent:SetAttribute("Adversary", adversary)
	agent.Parent= env.Agents
	agent.Name="Adversary"
	if(adversary) then
		setAgentColor(agent,adversary_color)
		if(leader) then
			setAgentColor(agent,leaderColor)
			agent["Body Colors"].TorsoColor=BrickColor.DarkGray()
		end
		agent.Name="Agent"
	end

end

local function make_envs(NUM_ENV)
	Env=ReplicatedStorage:WaitForChild('Model')
	local tree=ReplicatedStorage:WaitForChild('Tree'):Clone()
	local coin=ReplicatedStorage:WaitForChild('Coin'):Clone()

	local p1=Env.Wall1.Position
	local p2=Env.Wall2.Position
	local p3=Env.Wall3.Position
	local p4=Env.Wall4.Position
	local Pos=0.25*(p1+p2+p3+p4)

	Env.PrimaryPart=Env.center

	for i = 1, NUM_ENV.X do
		for j = 1, NUM_ENV.Y do
			for k = 1, NUM_ENV.Z do
				local i_env=Env:Clone()
				i_env.Parent=workspace
				i_env.PrimaryPart=i_env.center
				i_env:SetPrimaryPartCFrame(CFrame.new(i*Dx+Pos.X, j*Dy+Pos.Y, k*Dz+Pos.Z))
				i_env.Name=string.format("Env_%d_%d_%d", i,j,k)
				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,20.0,Vector3.new(0,0,0),true, "Adv1", false)

				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,20.0,Vector3.new(0,0,0),true, "Adv2", false)

				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,20.0,Vector3.new(0,0,0),true, "Adv3", false)

				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,20.0,Vector3.new(0,0,0),true, "Adv4", true)


				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,30.0,Vector3.new(0,0,0),false, "Agent1",false)

				make_agent(i_env, Env.center.Position,
					--blind,silent,u_range,comm,adversary
					false,true,30.0,Vector3.new(0,0,0),false, "Agent2",false)

				--make_object(i_env, Env.center.Position, tree, "Tree1", i_env.Trees, 23)
				--make_object(i_env, Env.center.Position, tree, "Tree2", i_env.Trees, 23)
				make_object(i_env, Env.center.Position, coin, "Coin1", i_env.Coins, 2)
				make_object(i_env, Env.center.Position, coin, "Coin2", i_env.Coins, 2)

				table.insert(Envs_list, i_env)
			end
		end
	end
	Env:Destroy()
end


local function costum_step()
	RunService.PhysicsPaused = false
	RunService:DoPhysicsStep(DT)
	RunService.PhysicsPaused = true
	--wait(DT)

end

local function get_ob_single_agent(env, agent, agent_idx)
	local obs={}
	local center=env.center.Position
	table.insert(obs, agent.HumanoidRootPart.Position.X-center.X)
	table.insert(obs, agent.HumanoidRootPart.Position.Z-center.Z)
	table.insert(obs, agent.HumanoidRootPart.AssemblyLinearVelocity.X)
	table.insert(obs, agent.HumanoidRootPart.AssemblyLinearVelocity.Z)


	--for _,tree in ipairs(env.Trees:GetChildren()) do
	--	table.insert(obs, agent.HumanoidRootPart.Position.X-tree.Position.X)
	--end
	for _,coin in ipairs(env.Coins:GetChildren()) do
		table.insert(obs, agent.HumanoidRootPart.Position.X-coin.Position.X)
		table.insert(obs, agent.HumanoidRootPart.Position.Z-coin.Position.Z)
		table.insert(obs, bool_to_number[coin:GetAttribute("Collectable")])
	end


	for idx, other_agent in ipairs(env.Agents:GetChildren()) do
		if idx == agent_idx then
			continue
		end
		--table.insert(obs, other_agent:GetAttribute("Comm").X)
		--table.insert(obs, other_agent:GetAttribute("Comm").Z)
		table.insert(obs,
			agent.HumanoidRootPart.Position.X - other_agent.HumanoidRootPart.Position.X)
		table.insert(obs,
			agent.HumanoidRootPart.Position.Z - other_agent.HumanoidRootPart.Position.Z)


		--if not other_agent:GetAttribute("Adversary") then
			table.insert(obs,
					agent.HumanoidRootPart.AssemblyLinearVelocity.X - other_agent.HumanoidRootPart.AssemblyLinearVelocity.X)
			table.insert(obs,
					agent.HumanoidRootPart.AssemblyLinearVelocity.Z - other_agent.HumanoidRootPart.AssemblyLinearVelocity.Z)
		--end

	end

	return obs
end

local function get_ob_single_env(env)
	local obsAll ={}
	for idx,agent in ipairs(env.Agents:GetChildren()) do
		table.insert(obsAll, get_ob_single_agent(env, agent, idx))
	end
	return obsAll
end

local function get_ob(idx)
	if(idx == nil) then
		local obs={}
		for _,env in ipairs(Envs_list) do
			local env_obs=get_ob_single_env(env)
			table.insert(obs, env_obs)
		end
		return obs
	else
		local env = Envs_list[idx+1]
		return get_ob_single_env(env)
	end
end


local function reset_single_agent(agent, center)
	local pos=Vector3.new(
		math.random(center.X+_G.XLimits[1], center.X+_G.XLimits[2]),
		3.5,
		math.random(center.Z+_G.ZLimits[1], center.Z+_G.ZLimits[2])
	)
	--agent:SetPrimaryPartCFrame (CFrame.new(pos))
	agent.HumanoidRootPart.CFrame=CFrame.new(pos)
	if(not agent:GetAttribute("Adversary")) then
		setAgentColor(agent, agent_color)
	else
		setAgentColor(agent, adversary_color)
	end
end

local function reset_single_object(object, center, collecable)
	local pos=Vector3.new(
		math.random(center.X+_G.XLimits[1], center.X+_G.XLimits[2]),
		3.5,
		math.random(center.Z+_G.ZLimits[1], center.Z+_G.ZLimits[2])
	)
	object:SetAttribute("Collectable", true)
	object.CFrame=CFrame.new(pos)
	object.Transparency=0.0

end

local function reset_single_env(env)
	local center=env.center.Position
	for idx,agent in ipairs(env.Agents:GetChildren()) do
		reset_single_agent( agent, center)
	end
	for idx,tree in ipairs(env.Trees:GetChildren()) do
		reset_single_object( tree, center)
	end
	for idx,coin in ipairs(env.Coins:GetChildren()) do
		reset_single_object( coin, center)
	end
	return get_ob_single_env(env)
end

local function reset(idx)
	if(idx == nil) then
		for _,env in ipairs(Envs_list) do
			reset_single_env(env)
		end
		return get_ob()
	else
		return reset_single_env(Envs_list[idx+1])
	end
end

local function Agent_action(env,agent,action_u, action_c)
	--
	--print(action_u, action_c)
	agent.Humanoid:Move(Vector3.new(action_u[1], 0, action_u[2]), false)
	--agent:SetAttribute("Comm", Vector3.new(action_c[1], 0, action_c[2]))

end

local function make_action(action_variables)
	for Idx,env in ipairs(Envs_list) do
		for idx, agent in ipairs(env.Agents:GetChildren()) do
			Agent_action(env,agent,action_variables["u"][idx],action_variables["c"][idx])
		end
	end
end

local function agent_reward(env, agent)
	-- Agents are negatively rewarded if caught by adversaries
	local reward=0
	local done=false

	for idx,other_agent in ipairs(env.Agents:GetChildren()) do
		--competition
		if(other_agent:GetAttribute("Adversary")) then
			local d=distance(other_agent.HumanoidRootPart , agent.HumanoidRootPart)
			reward+= 0.01* d
			if (d<Collision_margin) then
				reward-=50
				done=true
			end
		else
			--cooporation
			reward+=0
		end
	end

	for idx,coin in ipairs(env.Coins:GetChildren()) do
		local d=distance(coin , agent.HumanoidRootPart)
		-- some small reward to encourage collecting coins
		reward+= 0.002* d
		if (d<Collision_margin and coin:GetAttribute("Collectable")) then
			reward+=20
			setAgentColor(agent, BrickColor.Green())
			coin:SetAttribute("Collectable", false)
			coin.Transparency=1.0
		end

	end

	return reward,done

end

local function adversary_reward(env, agent)
	local reward=0
	local done=false
	for idx,adversary_agent in ipairs(env.Agents:GetChildren()) do
		--competition
		if(adversary_agent:GetAttribute("Adversary")) then
			local temp_reward=100000000
			for idx,good_agent in ipairs(env.Agents:GetChildren()) do
				if(not good_agent:GetAttribute("Adversary")) then
					local d=distance(adversary_agent.HumanoidRootPart , good_agent.HumanoidRootPart)
					temp_reward= math.min(temp_reward, d)
					if (d<Collision_margin) then
						reward+=50
						done=true
						setAgentColor(adversary_agent, BrickColor.Green())
						setAgentColor(good_agent, BrickColor.Red())
					end
				end

			end
			reward+= -0.01* temp_reward
		else --cooporation
			reward+=0
		end
	end




	return reward,done
end

local function compute_rewards()
	local rewardsAll={}
	local is_doneAll={}
	local infoAll={}

	for E,env in ipairs(Envs_list) do
		local rewards={}
		local is_done={}
		local info={}
		for idx,agent in ipairs(env.Agents:GetChildren()) do
			local reward,done
			if(agent:GetAttribute("Adversary")) then
				reward,done= adversary_reward(env, agent)
			else
				reward,done=agent_reward(env, agent)
			end
			table.insert(rewards,reward)
			table.insert(is_done,done)
			table.insert(info, {[""]=""})
		end

		table.insert(rewardsAll, rewards)
		table.insert(is_doneAll, is_done)
		table.insert(infoAll, info)

	end
	return rewardsAll, is_doneAll, infoAll
end


local function initialize_env(env)
	local blind ={}
	local silent = {}
	local u_range ={}
	local obs_dim ={}
	local adv ={}
	local comm ={}

	local obs= reset_single_env(env)
	for idx,agent in ipairs(env.Agents:GetChildren()) do
		table.insert(blind, agent:GetAttribute("Blind"))
		table.insert(silent, agent:GetAttribute("Silent"))
		table.insert(u_range, agent:GetAttribute("U_range"))
		table.insert(adv, agent:GetAttribute("Adversary"))
		--table.insert(comm, agent:GetAttribute("Comm"))
		table.insert(obs_dim, table.getn(obs[idx]))

	end
	return blind, silent, u_range, obs_dim, obs, adv, comm
end

MLRpcService = game:GetService("MLRpcService")

local function close()
	if(Env) then
		Env:Destroy()
	end
end

MLRpcService:RegisterMethodHandler("initialize", function()
	close()
	make_envs(NUM_ENV)
	local blind, silent, u_range, obs_dim, obs, adv, commAll = initialize_env(Envs_list[1])
	local response = {
		["info"] = {
			["dim_p"]=2,
			["dim_c"]=2,
			["discrete_action"]= false,
			["collaborative"]= false
		};
		["policy_agents"] = {
			["num_agents"]= #Envs_list[1].Agents:GetChildren(),
			["blind"]=blind,
			["silent"]=silent,
			["u_range"]=u_range,
			["comm"]=commAll,
			["obs_dim"]= obs_dim,
			["adversary"]= adv,

		};
		["maxSteps"]= max_steps;
		["success_goal_reward"]= success_goal_reward,
		["p_norm"]= p_norm,
		["num_envs"]= #Envs_list,
	}
	return response
end)

MLRpcService:RegisterMethodHandler("reset", function(idx)
	reset(idx)
	local states=get_ob(idx)
	local response={
		["observations"] = states;
	}
	return response
end)


MLRpcService:RegisterMethodHandler("get_ob", function()
	local states=get_ob()
	local response={
		["observations"] = states;
	}
	return response
end)

MLRpcService:RegisterMethodHandler("step", function(action_variables)
	make_action(action_variables)
	costum_step()
	local observations=get_ob()
	local rewards,is_done, info=compute_rewards()
	local response={
		["observations"] = observations;
		["rewards"]= rewards;
		["is_done"]= is_done;
		["info"]= info
	}
	return response
end)



--make_envs(NUM_ENV)
MLRpcService:Start(port)
