import time
import os
import numpy as np
from itertools import chain
import torch

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                # print('actions_env', actions_env, 'actions', actions)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # print('rewards', rewards)
                # print('dones', dones)
                # print('infos', infos)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                        train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            
            # Convert numpy arrays to tensors on the correct device
            share_obs_tensor = torch.FloatTensor(self.buffer[agent_id].share_obs[step]).to(self.device)
            obs_tensor = torch.FloatTensor(self.buffer[agent_id].obs[step]).to(self.device)
            rnn_states_tensor = torch.FloatTensor(self.buffer[agent_id].rnn_states[step]).to(self.device)
            rnn_states_critic_tensor = torch.FloatTensor(self.buffer[agent_id].rnn_states_critic[step]).to(self.device)
            masks_tensor = torch.FloatTensor(self.buffer[agent_id].masks[step]).to(self.device)
        
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                share_obs_tensor,
                obs_tensor,
                rnn_states_tensor,
                rnn_states_critic_tensor,
                masks_tensor
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        # print('dones', dones)
        # rnn_states[dones == True] = np.zeros(
        #     ((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #     dtype=np.float32,
        # )
        # rnn_states_critic[dones == True] = np.zeros(
        #     ((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #     dtype=np.float32,
        # )
        # masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Reshape dones properly to match rnn_states dimensions
        dones_reshaped = dones.reshape(self.n_rollout_threads, self.num_agents)
        
        # Reset RNN states for done agents
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                if dones_reshaped[i][j]:
                    rnn_states[i, j] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i, j] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                if dones_reshaped[i][j]:
                    masks[i, j] = np.zeros(1, dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Render episodes using trained policies"""
        print("Starting to render episodes...")
        
        # Import imageio if saving gifs
        if self.all_args.save_gifs:
            import imageio
            all_frames = []
        
        for episode in range(self.all_args.render_episodes):
            print(f"Rendering episode {episode+1}/{self.all_args.render_episodes}")
            episode_rewards = []
            
            # Reset environment
            obs = self.envs.reset()
            
            # Capture initial frame
            if self.all_args.save_gifs:
                try:
                    image = self.envs.render("rgb_array")[0]
                    all_frames.append(image)
                except Exception as e:
                    print(f"Error capturing frame: {e}")
                    continue
            
            # Initialize RNN states and masks
            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            # Run episode
            for step in range(self.episode_length):
                print(f"  Step {step+1}/{self.episode_length}", end="\r")
                
                # Collect actions from all agents
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    
                    # Get action from policy
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True
                    )
                    
                    # Process action based on action space type
                    action = _t2n(action)
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        action_env = np.zeros((self.n_rollout_threads, action.shape[-1]))
                        for i in range(self.envs.action_space[agent_id].shape):
                            action_env[:, i] = action[:, i]
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(action, axis=-1)
                    else:
                        # For continuous actions
                        action_env = action
                    
                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                
                # Prepare actions for environment step
                actions_env = []
                for i in range(self.n_rollout_threads):
                    agent_actions = []
                    for temp_action_env in temp_actions_env:
                        agent_actions.append(temp_action_env[i])
                    actions_env.append(agent_actions)
                
                # Step environment
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)
                
                # Update RNN states for done agents
                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                
                # Render current frame
                if self.all_args.save_gifs:
                    try:
                        image = self.envs.render("rgb_array")[0]
                        all_frames.append(image)
                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                else:
                    self.envs.render("human")
                
                # Break if all agents are done
                if np.all(dones):
                    break
            
            # Print episode results
            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print(f"  Agent {agent_id} average reward: {average_episode_rewards:.2f}")
        
        # Save gif if requested
        if self.all_args.save_gifs and all_frames:
            print(f"Saving {len(all_frames)} frames as gif...")
            gif_path = os.path.join(self.run_dir, 'gifs', f"episode_{episode}.gif")
            imageio.mimsave(gif_path, all_frames, duration=self.all_args.ifi)
            print(f"Gif saved to {gif_path}")
        
        print("Rendering complete!")
