# Gym-Roblox

```bash
cd gym-roblox
pip install -e .
```

# Example
Training the ant environment using the PPOalgorithm in [OpenAI Baselines](https://github.com/openai/baselines)
```bash
python -m baselines.run --alg=ppo2 --env=gym_roblox.envs:ContinuousActions_ContinuousStates-v0 --network=mlp --num_timesteps=2e6 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy --save_path=~/models/Pendulum --log_path=~/logs/Pendulum
```
