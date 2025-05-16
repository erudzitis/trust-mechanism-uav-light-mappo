import numpy as np
import torch

# NOTE: This trajectory might not even be correct, because it expects speed?, however nowhere in all of the examples
# is speed used in trajectory calculation, only time...

class StraightLineTraj(object):
    """
    A trajectory that moves a drone in a straight line from start_pos to end_pos.
    """
    def __init__(self, start_pos=np.array([0, 0, 0]), end_pos=np.array([5, 0, 0]), 
                 speed=1.0, hover_time=1.0, yaw=0.0):
        """
        Initialize a straight line trajectory.
        
        Inputs:
            start_pos: starting position (x, y, z)
            end_pos: ending position (x, y, z)
            speed: desired speed in m/s
            hover_time: time to hover at the end position
            yaw: constant yaw angle during flight (radians)
        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.speed = speed
        self.hover_time = hover_time
        self.yaw = yaw
        
        # Calculate direction vector and distance
        self.direction = self.end_pos - self.start_pos
        self.distance = np.linalg.norm(self.direction)
        
        if self.distance > 0:
            self.direction = self.direction / self.distance  # Normalize
        else:
            self.direction = np.array([1, 0, 0])  # Default direction if start=end
            
        # Calculate flight time
        self.flight_time = self.distance / self.speed if self.speed > 0 else 0
        self.total_time = self.flight_time + self.hover_time
    
    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.
        
        Inputs:
            t: time in seconds
        Outputs:
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
                yaw_ddot  yaw acceleration, rad/s**2
        """
        # Handle different phases of flight
        if t <= 0:
            # Before flight starts
            x = self.start_pos
            x_dot = np.zeros(3)
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot = np.zeros(3)
        elif t < self.flight_time:
            # During flight
            x = self.start_pos + t * self.speed * self.direction
            x_dot = self.speed * self.direction
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot = np.zeros(3)
        else:
            # After reaching destination
            x = self.end_pos
            x_dot = np.zeros(3)
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot = np.zeros(3)
        
        # Constant yaw
        yaw = self.yaw
        yaw_dot = 0.0
        yaw_ddot = 0.0
        
        flat_output = {
            'x': x, 
            'x_dot': x_dot, 
            'x_ddot': x_ddot, 
            'x_dddot': x_dddot, 
            'x_ddddot': x_ddddot,
            'yaw': yaw, 
            'yaw_dot': yaw_dot, 
            'yaw_ddot': yaw_ddot
        }
        
        return flat_output


class BatchedStraightLineTraj(object):
    """
    A batched trajectory generator that moves multiple drones in straight lines from start positions 
    to end positions at specified speeds.
    
    This trajectory uses a speed-based parameterization approach, where the user specifies desired 
    speeds for each drone. The implementation then calculates flight times based on distance/speed
    and generates appropriate position and derivative references at each time step.
    
    Unlike time-based trajectory parameterizations (such as CircularTraj), this approach provides
    more intuitive control over drone movement speeds while still generating the reference values
    expected by the controller.
    
    Parameters
    ----------
    start_positions : array-like or torch.Tensor, shape (num_drones, 3)
        Starting positions for each drone in the batch [x, y, z]
    end_positions : array-like or torch.Tensor, shape (num_drones, 3)
        Target ending positions for each drone in the batch [x, y, z]
    speeds : array-like or torch.Tensor, shape (num_drones,), optional
        Desired constant speeds for each drone in m/s. Default is 1.0 for all drones.
    hover_times : array-like or torch.Tensor, shape (num_drones,), optional
        Time to hover at the end position for each drone in seconds. Default is 1.0 for all drones.
    yaws : array-like or torch.Tensor, shape (num_drones,), optional
        Constant yaw angle for each drone during flight in radians. Default is 0.0 for all drones.
    device : str, optional
        Device to store tensors on ('cpu' or 'cuda'). Default is 'cpu'.
        
    Attributes
    ----------
    num_drones : int
        Number of drones in the batch
    directions : torch.Tensor, shape (num_drones, 3)
        Normalized direction vectors from start to end positions
    distances : torch.Tensor, shape (num_drones,)
        Euclidean distances from start to end positions
    flight_times : torch.Tensor, shape (num_drones,)
        Calculated flight times based on distances and speeds
    total_times : torch.Tensor, shape (num_drones,)
        Total trajectory times including hover time at the end
        
    Notes !!!
    -----
    The trajectory generator provides reference values to the controller, which then
    computes appropriate motor commands to make the drones follow the trajectory (controllers try to follow these references).
    The actual drone movement is handled by the physics simulation.
    """
    def __init__(self, start_positions, end_positions, speeds=None, hover_times=None, 
                 yaws=None, device='cpu'):
        """
        Initialize batched straight line trajectories.
        
        Inputs:
            start_positions: tensor of shape (num_drones, 3) with starting positions
            end_positions: tensor of shape (num_drones, 3) with ending positions
            speeds: tensor of shape (num_drones,) with speeds (default: 1.0 for all)
            hover_times: tensor of shape (num_drones,) with hover times (default: 1.0 for all)
            yaws: tensor of shape (num_drones,) with yaw angles (default: 0.0 for all)
            device: device to store tensors on
        """
        self.num_drones = start_positions.shape[0]
        self.device = device
        
        # Convert inputs to tensors if they're not already
        self.start_positions = torch.tensor(start_positions, device=device, dtype=torch.float64)
        self.end_positions = torch.tensor(end_positions, device=device, dtype=torch.float64)
        
        if speeds is None:
            self.speeds = torch.ones(self.num_drones, device=device, dtype=torch.float64)
        else:
            self.speeds = torch.tensor(speeds, device=device, dtype=torch.float64)
            
        if hover_times is None:
            self.hover_times = torch.ones(self.num_drones, device=device, dtype=torch.float64)
        else:
            self.hover_times = torch.tensor(hover_times, device=device, dtype=torch.float64)
            
        if yaws is None:
            self.yaws = torch.zeros(self.num_drones, device=device, dtype=torch.float64)
        else:
            self.yaws = torch.tensor(yaws, device=device, dtype=torch.float64)
        
        # Calculate direction vectors and distances
        self.directions = self.end_positions - self.start_positions
        self.distances = torch.norm(self.directions, dim=1)
        
        # Normalize directions (handle zero distances)
        nonzero_mask = self.distances > 0
        self.directions[nonzero_mask] = self.directions[nonzero_mask] / self.distances[nonzero_mask].unsqueeze(1)
        self.directions[~nonzero_mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float64)
        
        # Calculate flight times
        self.flight_times = torch.zeros_like(self.distances)
        nonzero_speed = self.speeds > 0
        self.flight_times[nonzero_speed] = self.distances[nonzero_speed] / self.speeds[nonzero_speed]
        
        # Total time includes hover time
        self.total_times = self.flight_times + self.hover_times
    
    def update(self, t):
        """
        Given the present time, return the desired flat outputs for all drones.
        
        Inputs:
            t: time in seconds (can be scalar or tensor of shape (num_drones,))
        Outputs:
            flat_output: dict containing batched position and derivatives
        """
        # Convert scalar time to tensor if needed
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.float64).expand(self.num_drones)
        elif t.numel() == 1:
            t = t.expand(self.num_drones)
        
        # Initialize output tensors
        x = torch.zeros_like(self.start_positions)
        x_dot = torch.zeros_like(self.start_positions)
        x_ddot = torch.zeros_like(self.start_positions)
        x_dddot = torch.zeros_like(self.start_positions)
        x_ddddot = torch.zeros_like(self.start_positions)
        
        # Before flight starts
        before_start = t <= 0
        x[before_start] = self.start_positions[before_start]
        
        # During flight
        in_flight = (t > 0) & (t < self.flight_times)
        if torch.any(in_flight):
            flight_progress = t[in_flight].unsqueeze(1) * self.speeds[in_flight].unsqueeze(1)
            x[in_flight] = self.start_positions[in_flight] + flight_progress * self.directions[in_flight]
            x_dot[in_flight] = self.speeds[in_flight].unsqueeze(1) * self.directions[in_flight]
        
        # After reaching destination
        after_flight = t >= self.flight_times
        x[after_flight] = self.end_positions[after_flight]
        
        # Constant yaw for all drones
        yaw = self.yaws
        yaw_dot = torch.zeros_like(yaw)
        yaw_ddot = torch.zeros_like(yaw)
        
        flat_output = {
            'x': x,
            'x_dot': x_dot,
            'x_ddot': x_ddot,
            'x_dddot': x_dddot,
            'x_ddddot': x_ddddot,
            'yaw': yaw,
            'yaw_dot': yaw_dot,
            'yaw_ddot': yaw_ddot
        }
        
        return flat_output
