local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local port = 8070
RunService.PhysicsPaused = true
local DT=0.2


local NUM_ENV=Vector3.new(10,1,10)
local Dx=100
local Dy=0
local Dz=100
local Dr= Vector3.new(Dx,Dy,Dz)


local Env
local Envs_list={}

local LidarL=50
local LidarN=10
local offset=5
local CyliderScale=1000
bool_to_number={ [true]=1, [false]=0 }

local weight={1, 1,  0.0,  0.05, 0.05}
local scale= {100, 100, 180.0, 20,  20}

local p_norm = 2
local success_goal_reward= -0.01
local eps_rew=0
local max_steps=200


local function convert360(A,B)
	--local h= math.floor((angle+360)/360)
	--local r= angle+360 - h*360
	--return r
	local d = A - B
	if d > 180 then d -= 360 end
	if d < -180 then d += 360 end
	return math.abs(d)
end


local function makeLidar(Env)
	local agent=Env.AGENT.VehicleSeat
	local lidar= Instance.new("Part")
	lidar.Name="Lidar"
	lidar.Parent=agent
	lidar.Size=Vector3.new(1.0,1.0,1.0)
	lidar.Position=agent.Position+Vector3.new(0,offset,0)
	local weld=Instance.new("WeldConstraint")
	weld.Part0=lidar
	weld.Part1=agent
	weld.Parent=lidar
	lidar.CanCollide=false
	lidar.CanTouch=false

	for i=0,LidarN-1 do
		local part= Instance.new("Part")
		local alpha_d=360/LidarN*i
		local alpha=alpha_d*math.pi/180

		part.Shape = Enum.PartType.Cylinder
		part.Size=Vector3.new(LidarL,LidarL/CyliderScale,LidarL/CyliderScale)
		--part.Transparency=0.2
		part.Name=string.format("L%d",i)
		part.Parent=agent
		part.Orientation=lidar.CFrame:VectorToWorldSpace(Vector3.new(0,alpha_d,0))
		part.BrickColor=BrickColor.Green()
		part.CanCollide=false
		part.CanTouch=false
		part.Position=agent.Position+
			lidar.CFrame:VectorToWorldSpace(Vector3.new(LidarL/2*math.cos(alpha),offset,-LidarL/2*math.sin(alpha)))
		local weld=Instance.new("WeldConstraint")
		weld.Part0=part
		weld.Part1=lidar
		weld.Parent=lidar
		local density = .001
		local friction = .1
		local elasticity = 1
		local frictionWeight = 1
		local elasticityWeight = 1

		-- Construct new PhysicalProperties and set
		local physProperties = PhysicalProperties.new(density, friction, elasticity, frictionWeight, elasticityWeight)
		part.CustomPhysicalProperties = physProperties


	end
end



local function lidarRayCast(lidar,agent)

	local lidarStates = {}
	for i=0,LidarN-1 do
		local alpha_d=360/LidarN*i
		local alpha=alpha_d*math.pi/180

		local rayOrigin = lidar.Position
		local rayDirection = LidarL* lidar.CFrame:VectorToWorldSpace(Vector3.new(math.cos(alpha),0,-math.sin(alpha)))

		--- Build a "RaycastParams" object and cast the ray
		local raycastParams = RaycastParams.new()
		raycastParams.FilterDescendantsInstances = {agent.Parent}
		raycastParams.FilterType = Enum.RaycastFilterType.Blacklist
		local raycastResult = workspace:Raycast(rayOrigin, rayDirection, raycastParams)
		local L = agent:FindFirstChild(string.format("L%d",i))
		--table.insert(lidarStates, alpha)

		if raycastResult then
			local distance=(raycastResult.Position-lidar.Position).Magnitude
			L.Position=agent.Position+
				lidar.CFrame:VectorToWorldSpace(Vector3.new(distance/2*math.cos(alpha),offset,-distance/2*math.sin(alpha)))
			L.Size=Vector3.new(distance,distance/CyliderScale,distance/CyliderScale)
			L.BrickColor=BrickColor.Red()
			table.insert(lidarStates, distance)
		else
			L.Position=agent.Position+
				lidar.CFrame:VectorToWorldSpace(Vector3.new(LidarL/2*math.cos(alpha),offset,-LidarL/2*math.sin(alpha)))
			L.Size=Vector3.new(LidarL,LidarL/CyliderScale,LidarL/CyliderScale)
			L.BrickColor=BrickColor.Green()
			table.insert(lidarStates, LidarL)
		end

	end
	return lidarStates
end



local function make_envs(NUM_ENV)
	--local P=Env.Base.Position
	Env=ReplicatedStorage:WaitForChild('ATV')

	local p1=Env.Wall1.Position
	local p2=Env.Wall2.Position
	local p3=Env.Wall3.Position
	local p4=Env.Wall4.Position
	local Pos=0.25*(p1+p2+p3+p4)

	Env.PrimaryPart=Env.center
	local PrimaryPosition = Env.PrimaryPart.Position

	for i = 1, NUM_ENV.X do
		for j = 1, NUM_ENV.Y do
			for k = 1, NUM_ENV.Z do
				local i_env=Env:Clone()
				i_env.Parent=game.Workspace
				makeLidar(i_env)
				i_env.PrimaryPart=i_env.center
				i_env:SetPrimaryPartCFrame(CFrame.new(i*Dx+Pos.X, j*Dy+Pos.Y, k*Dz+Pos.Z))
				i_env.Name=string.format("Env_%d_%d_%d", i,j,k)
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
end

local function get_ob_single_env(env)
	local agent=env.AGENT.VehicleSeat
	local goal=env.Goal
	local lidar=agent.Lidar

	local agentV=agent.CFrame:VectorToObjectSpace(agent.AssemblyLinearVelocity)
	local agentW=agent.CFrame:VectorToObjectSpace(agent.AssemblyAngularVelocity)
	local env_s = {
		(agent.Position.X- goal.Position.X)/scale[1],
		(agent.Position.Z- goal.Position.Z)/scale[2],
		math.abs(convert360(agent.Orientation.Y, goal.Orientation.Y))/scale[3],
		(agentV.X)/scale[4],
		(agentV.Z)/scale[5],
	}
	for _,l in ipairs(lidarRayCast(lidar,agent)) do
		table.insert(env_s, l/LidarL)
	end
	return env_s
end

local function get_ob(idx)
	if(idx == nil) then
		local states={}
		for _,env in ipairs(Envs_list) do
			local env_s=get_ob_single_env(env)
			table.insert(states, env_s)
		end
		return states
	else
		local env = Envs_list[idx+1]
		return get_ob_single_env(env)
	end
end

local function reset_single_env(env)
	local center=env.center.Position
	local agent=env.AGENT.VehicleSeat
	local goal=env.Goal
	env.PrimaryPart=env.center
	env:SetPrimaryPartCFrame(
		CFrame.new(center.X,center.Y,center.Z) *
			CFrame.Angles(0,0,0)
	)
	--agent.Position=Vector3.new(center.X, 1, center.Z)
	--agent.Lidar.Position=Vector3.new(center.X, offset+1, center.Z)
	--env.Base.Position=Vector3.new(center.X, 1, center.Z)

	--goal.Position=Vector3.new(center.X+ math.random(-45, 45), 2, center.Z+ math.random(-45, 45))
	--goal.Orientation=Vector3.new(0, 0,0)


	env.AGENT.PrimaryPart=env.AGENT.VehicleSeat
	env.AGENT:SetPrimaryPartCFrame(
		CFrame.new(center.X, 2, center.Z) *
			CFrame.Angles(0,0,0)
	)
	agent.ThrottleFloat=0
	agent.SteerFloat=0
	goal.Position=Vector3.new(center.X+ -35, 2, center.Z+ math.random(-35, 35))
	goal.Orientation=Vector3.new(0,90,0)

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

local function make_action(action_variables)
	for Idx,env in ipairs(Envs_list) do

		local actions=action_variables[Idx]
		if(#Envs_list==1) then
			actions=action_variables
		end
		local agent=env.AGENT.VehicleSeat
		agent.SteerFloat=actions[1]
		agent.ThrottleFloat=actions[2]

	end
end

local function compute_rewards(achieved_goal)
	local rewards={}
	local is_done={}

	for E,env in ipairs(Envs_list) do
		local agent=env.AGENT.VehicleSeat
		local lidar=agent.Lidar

		local done= false
		local tmp=0
		for i = 1 , #scale, 1 do
			tmp+= (1-math.abs(achieved_goal[E][i])) * weight[i]
		end

		local d=math.sqrt(math.pow(achieved_goal[E][1],2)+ math.pow(achieved_goal[E][2],2))
		local angle=1 - math.abs(achieved_goal[E][3])

		local heading_reward=math.abs(angle*math.pow(1-d, 3))

		local rew= math.pow( tmp+heading_reward, p_norm)
		local hit_reward=0
		for index = 1, 10 do
			local d=achieved_goal[E][index+5]
			if(d*LidarL<10) then
				--hit_reward=-500
				done = true
			end
		end
		--rew+=hit_reward

		if(hit_reward<0) then
			lidar.BrickColor=BrickColor.Red()
		end

		if(rew > 8) then
			lidar.BrickColor=BrickColor.Green()
		else
			lidar.BrickColor=BrickColor.Gray()
		end

		table.insert(rewards, rew)
		table.insert(is_done, done)
	end

	return rewards, is_done
end


MLRpcService = game:GetService("MLRpcService")

local function close()
	if(Env) then
		Env:Destroy()
	end
end

MLRpcService:RegisterMethodHandler("reset", function(idx)
	reset(idx)
	local states=get_ob(idx)
	local response={
		["observations"] = states;
	}
	return response
end)


MLRpcService:RegisterMethodHandler("initialize", function()
	close()
	make_envs(NUM_ENV)
	local env=workspace.Env_1_1_1
	local agent=env.AGENT.VehicleSeat
	local goal=env.Goal
	local lidar=agent.Lidar

	local response={
		["obs_info"] = {
			["low"] = {-1,-1,-1, -1,-1};
			["high"]= {1,1,1,1,1};
		};

		["ac_info"] = {
			["low"] = {-1,-1};
			["high"] ={ 1, 1};
		};

		["maxSteps"]= max_steps;
		["success_goal_reward"]= success_goal_reward,
		["reward_weights"]= weight,
		["scale"]= scale,
		["p_norm"]= p_norm,
		["num_envs"]= #Envs_list,
	}

	for _,l in ipairs(lidarRayCast(lidar,agent)) do
		--table.insert(PostReq["obs_info"]["states"]["low"], 0)
		--table.insert(PostReq["obs_info"]["states"]["high"], 1)
		table.insert(response["obs_info"]["low"], 0)
		table.insert(response["obs_info"]["high"], 1)
	end
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
	local states=get_ob()
	local rewards,is_done=compute_rewards(states)
	local info={}
	for _,env in ipairs(Envs_list) do
		table.insert(info, {[""]=""})
	end

	local response={
		["observations"] = states;
		["rewards"]= rewards;
		["is_done"]= is_done;
		["info"]= info
	}

	if(#Envs_list==1) then
		response["observations"] = states[1];
		response["reward"]= rewards[1];
		response["is_done"]= is_done[1];
	end
	return response
end)

MLRpcService:Start(port)
