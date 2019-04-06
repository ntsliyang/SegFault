import sys
import gym


env_name = "Pendulum-v0"

if len(sys.argv) > 1:
    env_name = sys.argv[1]

env = gym.make(env_name)

print("Environment: ", env_name)
print("\tThe Observation space has size: %s" % env.observation_space)
print("\tThe action space has size: %s" % env.action_space)
