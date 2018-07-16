import numpy as np
from physics_sim import PhysicsSim
from tensorflow import set_random_seed

np.random.seed(0)
set_random_seed(0)

class SamsTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        reward = 1 - .1*( 
            abs(self.sim.pose[2] - self.target_pos[2]) 
        ).sum()
        
        if reward > 1:
            reward = 1
        if reward <= 0:
            reward = 0
            
        if self.sim.v[2] > 5: #is travelling upward
           reward += 1
    
        
        return reward
   
    
#You can add z-axis velocity in reward function to encourage quadcopter to fly towards the target.
#You can subtract angular velocity from the reward to make sure quadcopter flies straight up.
#You can subtract the sum of x and y-axis from the target position to make sure quadcopter goes straight up.
#You can include some large bonus and penalty rewards also. Such as a bonus on achieving the target height and a penalty on #crashing.
#Clip your final reward between (-1, 1). It will definitely help in better performance.
    
    def get_position(self):
        return self.sim.pose[:3]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if done:
                if(self.sim.pose[2] <= 0): # crash
                    reward -= 60
                
                #if(abs(self.sim.pose[2] - self.target_pos[2]) < 5):
                #    reward += 60 # is close to target
    
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state