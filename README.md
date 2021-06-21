# Gym-Roblox

```bash
cd gym-roblox
pip install -e .
```

# Example
Training the ant environment using the PPOalgorithm in [OpenAI Baselines](https://github.com/openai/baselines)
```bash
python -m baselines.run --alg=ppo2 --env=gym_roblox.envs:roblox_pendulum-v0 --network=mlp --num_timesteps=100 --ent_coef=0.1 --num_hidden=8 --num_layers=3 --value_network=copy
```
