local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RunService = game:GetService("RunService")
RunService.PhysicsPaused = true

local DT=0.05
--RunService:DoPhysicsStep(DT)
local Env

local function reset()
	Env.Pend.Position=Vector3.new(0,4,0)
	Env.Pend.Orientation=Vector3.new(0.02, 0, 0)
	Env.Pend.AssemblyLinearVelocity=Vector3.new(0, 0, 0)
	Env.Pend.AssemblyAngularVelocity=Vector3.new(0, 0, 0)
	Env.Pend.CylindricalConstraint.Velocity=0
	Env.Pend.VectorForce.Force=Vector3.new(0,0,0)

end

local function costum_step()
	RunService:DoPhysicsStep(DT)
end

local function get_ob()
	local v=Env.Pend.AssemblyLinearVelocity.Z
	local w=Env.Pend.AssemblyAngularVelocity.X
	local r=Env.Pend.Attachment.WorldPosition- Env.Pend.Position

	local states = {
		Env.Pend.Orientation.X,
		Env.Pend.AssemblyAngularVelocity.X,
		Env.Pend.Attachment.WorldPosition.Z,
		v+w*r.Magnitude
	}
	return states
end

local function make_action(action_variables)
	Env.Pend.CylindricalConstraint.Velocity=action_variables[1]

end
local function close()
	if(Env) then
		Env:Destroy()
	end
end


local MLRpcService = game:GetService("MLRpcService")

MLRpcService:RegisterMethodHandler("reset", function()
	reset()
	print('reset')
	local states=get_ob()
	local response = {
		["Response"] = "resetCompleted";
		["observations"] = states;
	}
	return response

end)

MLRpcService:RegisterMethodHandler("initialize", function()
	close()
	print('initialize')
	Env=ReplicatedStorage:WaitForChild('Model'):Clone()
	Env.Parent=workspace
	Env.PrimaryPart =Env.Base

	local response = {
		["Response"] = "initializeCompleted";
		["obs_info"] = {
			["low"] = {-40,-40,-40,-40};
			["high"]= { 40, 40, 40, 40};
		};
		["ac_info"] = {
			["low"] = {-10};
			["high"] ={ 10};
		};
		["maxSteps"]= 100;
	}
	return response

end)

MLRpcService:RegisterMethodHandler("get_ob", function()

	local states=get_ob()
	local response = {
		["observations"] = states;
	}
	return response

end)

MLRpcService:RegisterMethodHandler("step",  function(action_variables)

	make_action(action_variables)
	costum_step()
	local states=get_ob()
	local is_done= math.abs(states[3])>2.4 or math.abs(states[1])>15
	local reward=1.0/math.max(math.abs(states[1]),1.0);
	-- define whether we are done or not according to the states
	local response = {
		["observations"] = states;
		["reward"]= reward;
		["is_done"]= is_done
	}
	if(is_done) then
		--print("episode reward: "..reward)
	end
	return response
end)


MLRpcService:Start(8844)
