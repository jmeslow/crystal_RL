from pokemoncrystal import crystalenv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank,seed=0):
    def _init():
        env = crystalenv()
        env.reset(seed=(seed+rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # All magic numbers taken from redexperiments
    ep_length = 2048 * 80
    num_cpus = 16
    env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])
    model = PPO("MultiInputPolicy", env, verbose = 1, n_steps = ep_length//64, batch_size=512, n_epochs=1,gamma=0.997, ent_coef=0.01)
    print(model.policy)
    model.learn(total_timesteps=ep_length*num_cpus*10000)
# %%