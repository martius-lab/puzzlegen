import numpy as np
import argparse
import os
import imageio

from .ice_slider import IceSlider
from .digit_jump import DigitJump

N_RECORDS = 3


def make_env(env_name, seed):
    if env_name == 'ice_slider':
        return IceSlider(seed=seed)
    if env_name == 'digit_jump':
        return DigitJump(seed=seed)
    raise NotImplementedError


def extract(record_dir, number_levels, start_level, max_steps, verbose, env_name, n_repeat, random, **kwargs):
    episodes_data = {}
    os.makedirs(record_dir, exist_ok=True)

    for n in range(n_repeat):
        for i in range(number_levels):
            if verbose:
                print(f'Running episode {i}.')
            observations, acts, rewards, dones, infos = [], [], [], [], []
            env = make_env(env_name=env_name, seed=start_level + i)
            obs = env.reset()
            observations.append(obs)
            if i < N_RECORDS:
                movie_writer = imageio.get_writer(os.path.join(record_dir, f"{i:03d}.mp4"), fps=15, quality=9)
                movie_writer.append_data(obs)

            actions = env.get_solution() if not random else [env.action_space.sample() for _ in range(max_steps)]
            for act in actions:
                obs, rew, done, info = env.step(act)
                observations.append(obs)
                acts.append(act)
                rewards.append(rew)
                dones.append(done)
                infos.append(info)
                if i < N_RECORDS:
                    movie_writer.append_data(obs)

            episodes_data[str(start_level+i) + (f'-{n}' if random else '')] = {'obs': np.asarray(observations),
                                                              'actions': np.asarray(acts),
                                                              'rewards': np.asarray(rewards),
                                                              'dones': np.asarray(dones),
                                                              'seed': start_level + i}
    filename = os.path.join(record_dir, f"{env_name}-{'random' if random else 'expert'}-max{max_steps}-{start_level}+{number_levels}x{n_repeat}.npy")
    np.save(filename, episodes_data, allow_pickle=True)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True, help="directory to record movies to")
    parser.add_argument("--env-name", default="ice_slider", help="name of game to create", choices=['ice_slider', 'digit_jump'])
    parser.add_argument("--start-level", default=0, type=int, help="select an individual level to use")
    parser.add_argument("--number-levels", default=10, type=int, help="number of levels to play (unrelated from seeds)")
    parser.add_argument("--max-steps", default=100, type=int, help="maximum steps per episode")
    parser.add_argument("--n-repeat", default=1, type=int, help="repeats the procedure n times")
    parser.add_argument("--random", default=1, type=int, help="uses random vs expert agent")
    parser.add_argument("--verbose", default=True, help="increase output verbosity")

    args = parser.parse_args()

    kwargs = dict(num_envs=1, **vars(args))
    extract(**kwargs)


if __name__ == "__main__":
    main()
