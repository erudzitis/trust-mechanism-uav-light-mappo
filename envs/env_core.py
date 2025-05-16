import numpy as np
import torch
from copy import copy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rotorpy.world import World
from rotorpy.vehicles.multirotor import BatchedMultirotorParams, BatchedMultirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.wind.default_winds import BatchedNoWind
# from environment.utils.collision_detector import CollisionDetector
from trajectories.straight_line_traj import BatchedStraightLineTraj
from common.enums import FormationType

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        # Basic MAPPO settings
        self.agent_num = 5  # Number of drones
        
        # Observation space includes own state and relative positions to other drones, and the formation point:
        # Own position (3D): x, y, z coordinates
        # Own velocity (3D): vx, vy, vz
        # Relative positions to other drones: 3D vector to each other drone = 3 × (num_drones - 1)
        # Relative position to formation point (3D): vector to where the drone should be
        self.obs_dim = 3 + 3 + 3 * (self.agent_num - 1) + 3        
        self.action_dim = 3  # 3D velocity control (x, y, z)

        # Simulation parameters
        self.device = torch.device('cpu')
        self.sim_time = 10.0
        self.dt = 0.01
        self.formation_type = FormationType.V if 'FormationType' in globals() else 0  # Default to V formation if enum available
        self.max_steps = int(self.sim_time / self.dt)
        self.current_step = 0
        self.current_time = 0.0
        
        # Initialize simulation components if dependencies are available
        self._initialize_formation_env()
        
    def _initialize_formation_env(self):
        # Initialize world
        self.world = World({"bounds": {"extents": [-20, 20, -20, 20, 0, 10]}, "blocks": []})
        
        # Initial positions
        self.formation_offset_value = 1.0
        self.start_pos_no_offset = np.array([[-5, 0, 2]] * self.agent_num)
        self.end_pos_no_offset = np.array([[5, 0, 2]] * self.agent_num)
        self.start_pos = np.copy(self.start_pos_no_offset)
        self.end_pos = np.copy(self.end_pos_no_offset)
        
        # Set up formation
        self._setup_formation()
        
        # Collision detector
        # self.collision_detector = CollisionDetector(
        #     collision_distance=0.3,
        #     near_miss_distance=0.75,
        #     terminal_collision=True,
        #     terminal_severity=CollisionSeverity.COLLISION
        # )
        
        # Simulation objects
        self._setup_simulation()
        
        self.current_desired_velocities = np.zeros((self.agent_num, 3))
        self.max_agent_velocity = 2.0
        
        # Agent IDs
        self.possible_agents = [f"drone_{i}" for i in range(self.agent_num)]
        self.agents = copy(self.possible_agents)
        
        # Trajectory
        formation_center_speed = 1.0
        
        self.formation_trajectory = BatchedStraightLineTraj(
            start_positions=self.start_pos[0:1],  # Just for the formation center
            end_positions=self.end_pos[0:1],
            speeds=np.array([formation_center_speed]),
            hover_times=np.array([2.0]),
            device=self.device
        ) 

    def _setup_formation(self):
        """Set up the formation based on formation_type"""
        self.formation_offsets = np.zeros((self.agent_num, 3))
        
        if self.formation_type == getattr(FormationType, 'LINE', 0):
            # Line formation along x-axis
            for i in range(self.agent_num):
                self.formation_offsets[i, 0] = -i
                
        elif self.formation_type == getattr(FormationType, 'V', 1):
            # V formation with leader at the front
            if self.agent_num > 1:
                # Split drones between left and right wings
                left_count = (self.agent_num - 1) // 2 + ((self.agent_num - 1) % 2)
                right_count = (self.agent_num - 1) // 2
                
                # Left wing (positive y values)
                for i in range(left_count):
                    position = i + 1  # Position in the wing (1-indexed)
                    self.formation_offsets[i + 1, 0] = -position * self.formation_offset_value
                    self.formation_offsets[i + 1, 1] = position * self.formation_offset_value
                
                # Right wing (negative y values)
                for i in range(right_count):
                    position = i + 1  # Position in the wing (1-indexed)
                    self.formation_offsets[i + 1 + left_count, 0] = -position * self.formation_offset_value
                    self.formation_offsets[i + 1 + left_count, 1] = -position * self.formation_offset_value
        
        # Apply formation offsets to start and end positions
        self.start_pos = self.start_pos_no_offset + self.formation_offsets
        self.end_pos = self.end_pos_no_offset + self.formation_offsets           

    def _setup_simulation(self):
        """Initialize the simulation components"""
        # Set up vehicle parameters
        self.all_quad_params = [quad_params] * self.agent_num
        self.batch_params = BatchedMultirotorParams(self.all_quad_params, self.agent_num, self.device)
        
        # Hover rotor speed (rad/s)
        self.init_rotor_speed = 1788.53
        
        # Wind profile
        self.wind_profile = BatchedNoWind(self.agent_num)
        
    def _get_observations_as_list(self):
        """Convert observations dictionary to list format expected by light_mappo"""
        observations_dict = self._get_observations()
        return [observations_dict[f"drone_{i}"] for i in range(self.agent_num)]    
    
    def _get_observations(self):
        """Create observations for each agent"""
        observations = {}
        
        # Convert state tensors to numpy arrays
        positions_np = self.state['x'].cpu().numpy()
        velocities_np = self.state['v'].cpu().numpy()
        
        # Get the current target for the formation center
        current_formation_center_target = self.formation_trajectory.update(self.current_time)['x'].cpu().numpy().mean()
        
        for i, agent_id_str in enumerate(self.agents):
            own_pos = positions_np[i]
            own_vel = velocities_np[i]
            
            rel_positions_to_others = []
            for j in range(self.agent_num):
                if j != i:
                    rel_pos = positions_np[j] - own_pos
                    rel_positions_to_others.append(rel_pos)
            
            # Relative position to agent's desired formation point
            desired_agent_formation_pos = current_formation_center_target + self.formation_offsets[i]
            rel_to_formation_point = desired_agent_formation_pos - own_pos
            
            obs_components = [own_pos, own_vel] + rel_positions_to_others + [rel_to_formation_point]
            observations[agent_id_str] = np.concatenate(obs_components).astype(np.float32)
        
        return observations 
    
    def _calculate_rewards(self):
        """Calculate rewards for each agent"""
        rewards = {}
        
        current_positions_np = self.state['x'].cpu().numpy()
        current_velocities_np = self.state['v'].cpu().numpy()
        
        # Get current target for the formation center
        formation_center_target = self.formation_trajectory.update(self.current_time)['x'][0].cpu().numpy()
        
        # Calculate actual formation centroid (for cohesion rewards)
        actual_centroid = np.mean(current_positions_np, axis=0)
        
        # Get collision penalties from detector
        # collision_penalties = self.collision_detector.get_collision_penalties(self.agents)
        
        for i, agent_id_str in enumerate(self.agents):
            current_agent_pos = current_positions_np[i]
            current_agent_velocity = current_velocities_np[i]
            target_position_for_agent = formation_center_target + self.formation_offsets[i]
            
            # 1. Position Reward (Formation)
            error_to_target = np.linalg.norm(current_agent_pos - target_position_for_agent)
            position_reward = -min(error_to_target, 5.0)
            
            # 2. Velocity Reward (Formation)
            desired_direction = (target_position_for_agent - current_agent_pos) / (error_to_target + 1e-6)
            velocity_alignment = np.dot(current_agent_velocity, desired_direction)
            velocity_reward = max(velocity_alignment, 0)
            
            # 3. Cohesion Reward
            distance_to_centroid = np.linalg.norm(current_agent_pos - actual_centroid)
            cohesion_reward = -min(distance_to_centroid, 3.0)
            
            # 4. Collision avoidance penalty
            # collision_penalty = collision_penalties[agent_id_str]
            collision_penalty = 0
            
            # Combined Reward
            rewards[agent_id_str] = (
                0.5 * position_reward
                + velocity_reward
                + collision_penalty
                + cohesion_reward
            )
        
        return rewards    
    
    def _apply_actions(self, actions):
        """Convert agent actions (normalized velocity commands) to actual desired velocities."""
        # self.current_desired_velocities is a numpy array (num_drones, 3)
        
        for i, agent_id_str in enumerate(self.agents):
            if agent_id_str in actions:
                # actions[agent_name] is normalized [-1, 1]
                # Scale to a desired physical velocity
                normalized_action = np.clip(actions[agent_id_str], -1.0, 1.0)
                self.current_desired_velocities[i, :] = normalized_action * self.max_agent_velocity   
        
    def reset(self):
        """Reset the environment to initial state"""
        # Reset agents
        self.agents = copy(self.possible_agents)
        
        # Reset collision detector
        # self.collision_detector.reset(self.agents)
        
        # Reset step counter and time
        self.current_step = 0
        self.current_time = 0.0
        
        # Initialize drone states
        self.x0 = {
            'x': torch.tensor(self.start_pos, device=self.device).double(),
            'v': torch.zeros(self.agent_num, 3, device=self.device).double(),
            'q': torch.tensor([0, 0, 0, 1], device=self.device).repeat(self.agent_num, 1).double(),
            'w': torch.zeros(self.agent_num, 3, device=self.device).double(),
            'wind': torch.zeros(self.agent_num, 3, device=self.device).double(),
            'rotor_speeds': torch.tensor([self.init_rotor_speed] * 4, device=self.device).repeat(self.agent_num, 1).double()
        }
        
        # Create batched vehicle
        self.vehicle = BatchedMultirotor(
            self.batch_params,
            self.agent_num,
            self.x0,
            device=self.device,
            control_abstraction='cmd_vel',
            integrator='dopri5'
        )
        
        # Current state
        self.state = self.x0
        
        # Get observations
        return self._get_observations_as_list()

    def step(self, actions):
        """Take a step in the environment using the provided actions"""
        # Convert actions list to a format suitable for our environment
        actions_dict = {}
        for i, action in enumerate(actions):
            agent_id = f"drone_{i}"
            actions_dict[agent_id] = action
        
        # Apply actions to get desired velocities
        self._apply_actions(actions_dict)
        
        # Create control inputs for the vehicle
        control_input_dict_for_vehicle = {
            'cmd_v': torch.tensor(self.current_desired_velocities, dtype=torch.float64, device=self.device)
        }
        
        # Step the vehicle dynamics
        next_vehicle_state = self.vehicle.step(
            state=self.state,
            control=control_input_dict_for_vehicle,
            t_step=self.dt
        )
        
        # Update state
        self.state = next_vehicle_state
        
        # Increment step counter and time
        self.current_step += 1
        self.current_time = self.current_step * self.dt
        
        # Check for collisions
        current_positions = self.state['x'].cpu().numpy()
        current_velocities = self.state['v'].cpu().numpy()
        
        # collision_terminations = self.collision_detector.detect_collisions(
        #     positions=current_positions,
        #     agent_ids=self.agents,
        #     velocities=current_velocities
        # )
        
        # Get observations (in list format for MAPPO)
        observations = self._get_observations_as_list()
        
        # Get rewards
        rewards_dict = self._calculate_rewards()
        rewards = [rewards_dict[drone] for drone in self.agents]
        
        # Check termination & truncation
        # terminations = collision_terminations
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        
        # Combine terminations and truncations for MAPPO
        dones = np.array([terminations.get(d, False) or truncations.get(d, False) 
                for d in self.agents], dtype=np.bool_)
        
        # Additional info
        infos = [{
            # "collision_info": self.collision_detector.get_agent_collision_info(f"drone_{i}"),
            "formation_pos_error": np.linalg.norm(self.state['x'][i].cpu().numpy() - 
                                   (self.formation_trajectory.update(self.current_time)['x'][0].cpu().numpy() + 
                                    self.formation_offsets[i]))
        } for i in range(self.agent_num)]
        
        return [observations, rewards, dones, infos]
