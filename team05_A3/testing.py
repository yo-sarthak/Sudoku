# import sudoku_gym
from team05_A3.sudoku_gym import env
from stable_baselines3.common.env_checker import check_env
print(check_env(env))
from team05_A3.sudoku_gym import PPO

# Save the trained model
model = PPO("CnnPolicy", env, verbose=1)
model.load("sudoku_rl_agent")


# Test the trained model
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    print("HELLO LOOK HERE")
    print("Action is: ", action)
    print("New action is: ", action)
    obs, rewards, done, truncated, info = env.step(action)
    print("Current game state is: ")
    print(env.game_state)
    if done:
        break

# Print final board state
print("Final Board:")
print(env.board)
print("Game state is: ")
print(env.game_state)
