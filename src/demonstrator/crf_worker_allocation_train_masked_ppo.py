import gymnasium as gym
import sb3_contrib
import numpy as np
import stable_baselines3 as sb3

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True,
    normalize_observation_space=True,
    flat_observation_space=True,
    action_mode='task',  # alternative 'job'
)
env = sb3.common.monitor.Monitor(env)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


env = ActionMasker(env, mask_fn)

model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train the agent
log.info("training the model")
model.learn(total_timesteps=10_000)