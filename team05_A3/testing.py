import sudoku_gym
from sudoku_gym import env
from sudoku_gym import PPO

# Save the trained model
model = PPO("MlpPolicy", env, verbose=1)
model.load("sudoku_rl_agent")


# Test the trained model
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
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
