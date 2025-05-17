from gymnasium import spaces
import numpy as np
from envs.env_core import EnvCore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(
                low=-1,
                high=+1,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        """
        Render the environment.
        
        Args:
            mode (str): 'human' for interactive visualization, 'rgb_array' for array output
            
        Returns:
            numpy array or None: If mode is 'rgb_array', returns an RGB image as a numpy array
        """
        if mode == "human":
            # Create interactive plot for human viewing
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            self._render_on_axes(ax)
            plt.draw()
            plt.pause(0.01)
            return None
            
        elif mode == "rgb_array":
            # Create a figure and axes
            fig = Figure(figsize=(10, 8), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, projection='3d')
            
            # Render the environment on the axes
            self._render_on_axes(ax)
            
            # Convert to RGB array
            canvas.draw()
            buffer = canvas.buffer_rgba()  # Get the RGBA buffer
            image = np.asarray(buffer).astype(np.uint8)  # Convert to numpy array
            image = image[:, :, :3]  # Drop the alpha channel
            
            # Clean up to avoid memory leaks
            plt.close(fig)
            
            return image
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported")
    
    def _render_on_axes(self, ax):
        """Helper method to render the environment on given matplotlib axes"""
        # Set axis limits
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 10])
        
        # Set labels
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.set_title('UAV Formation')
        
        # Get current drone positions from the state
        positions = self.env.state['x'].cpu().numpy()
        
        # Get current formation center target
        formation_center_target = self.env.formation_trajectory.update(self.env.current_time)['x'][0].cpu().numpy()
        
        # Calculate individual drone target positions using formation offsets
        formation_targets = np.array([formation_center_target + self.env.formation_offsets[i] 
                                     for i in range(self.num_agent)])
        
        # Plot each drone position
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        for i in range(self.num_agent):
            color = colors[i % len(colors)]
            pos = positions[i]
            target = formation_targets[i]
            
            # Plot drone position as a sphere
            ax.scatter(pos[0], pos[1], pos[2], color=color, s=100, label=f'Drone {i}')
            
            # Plot target position as a hollow circle
            ax.scatter(target[0], target[1], target[2], color=color, 
                      edgecolors='black', s=50, alpha=0.5)
            
            # Draw a line between drone and its target
            ax.plot([pos[0], target[0]], 
                   [pos[1], target[1]], 
                   [pos[2], target[2]], 
                   color=color, linestyle='--', alpha=0.5)
            
        # Calculate and display current formation error
        errors = [np.linalg.norm(positions[i] - formation_targets[i]) for i in range(self.num_agent)]
        avg_error = np.mean(errors)
        ax.text2D(0.05, 0.95, f"Avg Formation Error: {avg_error:.2f}m", 
                transform=ax.transAxes, fontsize=10)
        
        # Add a legend
        ax.legend()

    def seed(self, seed):
        pass
