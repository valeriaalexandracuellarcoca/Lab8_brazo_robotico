import gymnasium as gym
from stable_baselines3 import PPO
from robotic_arm_env import RoboticArmEnv

# Crear el entorno
env = RoboticArmEnv()

# Crear el modelo PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)

# Entrenar el modelo
model.learn(total_timesteps=20000)

# Guardar el modelo entrenado
model.save("ppo_robotic_arm2")

# Evaluar el modelo entrenado
obs, _ = env.reset()
done = False
truncated = False
total_reward = 0
while not (done or truncated):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if truncated:
        print("Episodio truncado")
print(f"Recompensa total: {total_reward}")