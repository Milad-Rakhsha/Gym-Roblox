local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RunService = game:GetService("RunService")
RunService.PhysicsPaused = true


local DT=1/60

local Dx=5
local Dy=5
local Dz=10
local Dr= Vector3.new(Dx,Dy,Dz)

local Env=ReplicatedStorage:WaitForChild('Model')
local Envs_list={}
local Envs_center={}
local Envs_random={}

local function make_envs(nEnvs)
	local P=Env.Base.Position
	for i = 1, nEnvs.X do
		for j = 1, nEnvs.Y do
			for k = 1, nEnvs.Z do
				local d= Vector3.new(Dx*i,Dy*j,Dz*k)
				local i_env=Env:Clone()
				i_env.Parent = workspace
				i_env.PrimaryPart =i_env.Base
				i_env:SetPrimaryPartCFrame(CFrame.new(i*Dx+P.X, j*Dy+P.Y, k*Dz+P.Z))
				i_env.Name=string.format("Env-%d-%d-%d", i,j,k)
				table.insert(Envs_list, i_env)
				table.insert(Envs_random, Random.new(i))
			end
		end
	end
end

local function reset_single_env(idx, env)
	local p: Vector3 = env.Base.Position

	local random = Envs_random[idx]:NextNumber(-5, 5)
	env.Pend.Orientation=Vector3.new(random, 0, 0)
	env.Pend.Position=Vector3.new(p.X, p.Y + 1.5, p.Z)
	env.Pend.AssemblyLinearVelocity=Vector3.new(0, 0, 0)
	env.Pend.AssemblyAngularVelocity=Vector3.new(0, 0, 0)
	env.Pend.CylindricalConstraint.Velocity=0
	env.Pend.VectorForce.Force=Vector3.new(0,0,0)

	if env.Step.Value > env.StepMax.Value then
		env.StepMax.Value = env.Step.Value
	end
	env.Step.Value = 0
end
local function reset()
	for i,env in ipairs(Envs_list) do
		reset_single_env(i, env)
	end
end

local function costum_step()
	RunService:DoPhysicsStep(DT)
end

local function get_ob_single_env(env)
	local v = env.Pend.AssemblyLinearVelocity.Z
	local w = env.Pend.AssemblyAngularVelocity.X
	local r = env.Pend.Attachment.WorldPosition- env.Pend.Position
	local env_s = {
		env.Pend.Orientation.X,
		env.Pend.AssemblyAngularVelocity.X,
		env.Pend.Attachment.WorldPosition.Z - env.Base.Position.Z,
		v + w * r.Magnitude,
		--env.Pend.AssemblyVelocity.X,
	}
	return env_s
end

local function get_ob()
	local states={}
	for _,env in ipairs(Envs_list) do
		local env_s=get_ob_single_env(env)
		table.insert(states, env_s)
	end
	return states
end

local function make_action(action_variables)
	for idx,env in ipairs(Envs_list) do
		env.Pend.CylindricalConstraint.Velocity=action_variables[idx][1]
		--env.Pend.VectorForce.Force=Vector3.new(0,action_variables[idx][1],0)

		env.Step.Value += 1
	end
end

local function compute_rewards(states)
	local rewards={}
	local is_done={}

	for i,env in ipairs(Envs_list) do
		local done= math.abs(states[i][3])>2 or math.abs(states[i][1])>15
		local reward=1.0/math.max(math.abs(states[i][1]),1.0);
		reward = done and -1 or 1
		table.insert(rewards, reward)
		table.insert(is_done, done or env.Step.Value >= 200)
	end

	return rewards, is_done
end

local MLRpcService = game:GetService("MLRpcService")

local function close()
	for _,env in ipairs(Envs_list) do
		env:Destroy()
	end

	Envs_list={}
	Envs_center={}
	Envs_random={}
end

MLRpcService:RegisterMethodHandler("initialize", function(num_envs)
	close()

	print('initialize:', num_envs)
	local side = math.sqrt(num_envs)
	local numEnvsVec = Vector3.new(math.floor(side), 1, math.ceil(side))
	make_envs(numEnvsVec)

	local response = {
		["obs_info"] = {
			["low"] = {-40,-20,-40,-40};
			["high"]= { 40, 20, 40, 40};
		};
		["ac_info"] = {
			["low"] = {-10};
			["high"] ={ 10};
		};
		["maxSteps"]= 200;
		["num_envs"]= #Envs_list;

	}
	print('initialized', numEnvsVec, response)
	return response
end)

MLRpcService:RegisterMethodHandler("close", close)
MLRpcService:RegisterMethodHandler("seed", function(seed: number)
	local result = {}
	for i=1,#Envs_random,1 do
		local envSeed = seed+i
		Envs_random[i] = Random.new(envSeed)

		result[i] = envSeed
	end

	return result
end)

MLRpcService:RegisterMethodHandler("reset", function()
	reset()
	local states=get_ob()
	return {
		["observations"] = states;
	}
end)

MLRpcService:RegisterMethodHandler("get_ob", function()
	local states=get_ob()
	local PostReq={
		["observations"] = states;
	}
end)

MLRpcService:RegisterMethodHandler("step",  function(action_variables)
	local stepStart = os.clock ()
	make_action(action_variables)
	costum_step()

	local states=get_ob()
	local rewards,is_done=compute_rewards(states)
	-- define whether we are done or not according to the states
	local infos={}
	for i,env in ipairs(Envs_list) do
		local info = {}
		if is_done[i] then
			info.terminal_observation = states[i]
			reset_single_env(i, env)
			states[i] = get_ob_single_env(env)
		else
			info[""] = "" -- turn it into object table
		end
		table.insert(infos, info)
	end

	return {
		["observations"] = states;
		["rewards"]= rewards;
		["is_done"]= is_done;
		["info"]= infos
	}
end)

MLRpcService:Start()
