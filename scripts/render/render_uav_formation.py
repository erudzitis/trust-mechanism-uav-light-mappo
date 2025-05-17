#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import get_config
from envs.env_wrappers import DummyVecEnv

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # Use the same environment as in training
            from envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        print("Error: render only supports n_rollout_threads=1")
        exit(1)

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int, default=5, help="number of drones")
    # parser.add_argument('--render_episodes', type=int, default=3, help="number of episodes to render")
    # parser.add_argument('--save_gifs', action='store_true', default=True, help="save render as gif files")
    parser.add_argument('--gif_dir', type=str, default=None, help="directory to save gifs, will create if doesn't exist")
    # parser.add_argument('--ifi', type=float, default=0.1, help="gif frame interval")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Validate algorithm selection
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "Check recurrent policy!"
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), "Check recurrent policy!"
    else:
        raise NotImplementedError

    # Validate render settings
    assert all_args.use_render, "You need to set use_render to True"
    assert not (all_args.model_dir is None or all_args.model_dir == ""), "Set model_dir first"
    assert all_args.n_rollout_threads == 1, "Only support using 1 env to render"
    
    # Set up device
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Set up run directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Set up gif directory
    if all_args.save_gifs:
        if all_args.gif_dir is None:
            gif_dir = str(run_dir / 'gifs')
        else:
            gif_dir = all_args.gif_dir
        
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
        all_args.gif_dir = gif_dir

    # Set process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name)
    )

    # Set seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Initialize environment
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    # Set up config
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Create runner with appropriate policies 
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    # Create and run the runner
    runner = Runner(config)
    runner.render()
    
    # Clean up
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])