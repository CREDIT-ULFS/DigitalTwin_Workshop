import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from .Robot import *
from .Inverse_Kinematics import *
from tqdm import tqdm
import scipy as sp


class Trajectory:
    """Class for generating joint and cartesian trajectories. Currently, velocity and acceleration are not considered.
    """
    def __init__(self,
                robot,
                trajectory_type = 'joint',
                interpolation_type = 'linear',
                q_0 = None,
                q_f = None,
                x_0 = None,
                x_f = None,
                t_eval = None,
                generate = True,
                sol='000',
                tool_T = np.eye(4)):
        
        if trajectory_type not in ['joint', 'cartesian']:
            raise ValueError('Invalid trajectory type')
        

        if interpolation_type not in ['linear', 'fifth_order']:
            raise ValueError('Invalid interpolation type')
        
        
        self.robot = robot
        self.trajectory_type = trajectory_type
        self.tool_T = tool_T
        
        # Check if initial and final joint configurations are provided and are valid
        q_0 = np.array(q_0) if q_0 is not None else None
        q_f = np.array(q_f) if q_f is not None else None
        
        if isinstance(q_0, np.ndarray):
            if q_0.shape != (robot.n,):
                raise ValueError('Invalid initial joint configuration')
        self.q_0 = q_0
        
        if isinstance(q_f, np.ndarray):
            if q_f.shape != (robot.n,):
                raise ValueError('Invalid final joint configuration')
        self.q_f = q_f
        if not isinstance(t_eval, np.ndarray):
            raise ValueError('Invalid time array')
        
        
        self.x_0 = x_0
        self.x_f = x_f
        self.IK_fun = self.robot.IK_sol
        self.sol = sol
        self.meshes = None
        self.clim=None
        
    #------------------------------------------------------------------------------
        if t_eval is None:
            t_eval = np.linspace(0, 1, 50)
        self.t_eval = t_eval
        self.dt = t_eval[1] - t_eval[0]
        
        # Generate the trajectory if generate is True (default)
        if generate:
        # Joint trajectory    
            if self.trajectory_type == 'joint':
                self.q_traj = None
                self.dq_traj = None
                self.generate_trajectory()
        # Cartesian trajectory
            if self.trajectory_type == 'cartesian':
                self.x_traj = None
                self.dx_traj = None
                self.generate_trajectory()
        
    @property
    def t(self):
        """Gets the time array for the trajectory.
        """
        return self.t_eval
    
    @t.setter
    def t(self, value):
        """Sets the time array for the trajectory.

        Args:
            value (Numpy array): Time array

        Raises:
            ValueError: If the time array is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError('Invalid time array')
        self.t_eval = value
        self.dt = value[1] - value[0]
        
    @property
    def q(self):
            """
            Returns the trajectory of joint positions.

            Returns:
                list: A list of joint positions representing the trajectory.
            """
            return self.q_traj
    
    @q.setter
    def q(self, value):
        """
        Sets the joint trajectory.

        Args:
            value (numpy.ndarray): The joint trajectory values.

        Raises:
            ValueError: If the shape of the value array is not (len(self.t_eval), self.robot.n).

        """
        if value.shape != (len(self.t_eval), self.robot.n):
            raise ValueError('Invalid joint trajectory')
        self.q_traj = value
        self.dq_traj = np.gradient(value, self.t_eval, axis=0)
        self.ddq_traj = np.gradient(self.dq_traj, self.t_eval, axis=0)
        x_traj = np.zeros((1, 6))
        for q in self.q_traj:
            self.robot.q = q
            fkine = self.robot.fkine()
            euler = sp.spatial.transform.Rotation.from_matrix(fkine[:3,:3]).as_euler('ZYZ', degrees=True)
            x_val = np.concatenate((fkine[:3,3], euler))
            x_traj = np.vstack((x_traj, x_val))
        self.x_traj = x_traj[1:]
        

    
    @property    
    def x(self):
        """
        Returns the x trajectory.
        """
        return self.x_traj
    
    @x.setter
    def x(self, value):
        """
        Sets the Cartesian trajectory for the robot.

        Args:
            value (numpy.ndarray): The Cartesian trajectory as a numpy array of shape (len(self.t_eval), 6).

        Raises:
            ValueError: If the shape of the input trajectory is not (len(self.t_eval), 6).

        """
        if value.shape != (len(self.t_eval), 6):
            raise ValueError('Invalid Cartesian trajectory')
        self.x_traj = value
        q_traj = np.zeros((len(self.t_eval), self.robot.n))
        if self.IK_fun is None:
            print('No IK function provided. Using default function')
            inv = InverseKinematics(self.robot) # inverse kinematics object
            for i in range(len(self.t_eval)):
                q_traj[i] = inv.minimize_error(self.x_traj[i], q_0 = q_traj[i-1] if i > 0 else None)
        else:
            print('Using provided IK function')
            for i in range(len(self.t_eval)):
                q_traj[i] = self.IK_fun(self.x_traj[i],sol=self.sol)

        self.q_traj = q_traj
        self.dq_traj = np.gradient(q_traj,self.t_eval, axis=0)

        self.trajectory_polyline()

        
        

    
    def __add__(self, other):
        """
        Adds two Trajectory objects together and returns a new Trajectory object.

        Parameters:
        - other (Trajectory): The Trajectory object to be added.

        Returns:
        - new_traj (Trajectory): The new Trajectory object resulting from the addition.
        """

        new_traj = Trajectory(self.robot, trajectory_type=self.trajectory_type, t_eval=self.t_eval, generate=False)
        new_traj.t_eval = np.linspace(0, self.t_eval[-1] + other.t_eval[-1], len(self.t_eval) + len(other.t_eval))
        new_traj.dt = new_traj.t_eval[1] - new_traj.t_eval[0]
        new_traj.q_traj = np.concatenate((self.q_traj, other.q_traj))
        new_traj.dq_traj = np.concatenate((self.dq_traj, other.dq_traj))
        new_traj.q_0 = self.q_0
        new_traj.q_f = other.q_f
        new_traj.trajectory_type = self.trajectory_type
        new_traj.dq_traj = np.gradient(new_traj.q_traj, new_traj.t_eval, axis=0)
        if self.meshes is not None and other.meshes is not None:
            new_traj.meshes = np.concatenate((self.meshes, other.meshes))
        new_traj.trajectory_polyline()
        return new_traj
    
    def __eq__(self, other: object):
        """Plot the joint angles of two trajectories for comparison.

        Args:
            other (Trajectory object): The other trajectory object to be compared.
        """
        fig, ax = plt.subplots(1,1, dpi=300)
        for i in range(self.robot.n):
            j = i+1
            ax.plot(self.t_eval, self.q_traj[:,i], label=f'First $\\theta_{j}$')
        for i in range(self.robot.n):
            j = i+1
            ax.plot(other.t_eval, other.q_traj[:,i], label=f'Second $\\theta_{j}$')
        ax.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint Angle [deg]')

        plt.show()
        
    def __sub__(self, other):
        """Plot the joint angle error between two trajectories.

        Args:
            other (Trajectory object): The other trajectory object to be compared.
        """
        fig, ax = plt.subplots(1,1, dpi=300)
        for i in range(self.robot.n):
            ax.plot(self.t_eval, self.q_traj[:,i] - other.q_traj[:,i], label=f'$\\theta_{i+1}$ error')
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint Angle Error [deg]')
        plt.show()
        
        
    def generate_trajectory(self):
        """
        Generates the trajectory based on the specified trajectory type.

        If the trajectory type is 'joint', it generates a joint trajectory by interpolating between the initial and final joint positions.
        If the trajectory type is 'cartesian', it generates a Cartesian trajectory by interpolating between the initial and final Cartesian positions.

        The generated trajectory is stored in the `q_traj` and `dq_traj` attributes for joint trajectories,
        and in the `x_traj` and `dx_traj` attributes for Cartesian trajectories.

        Returns:
            None
        """
        if self.trajectory_type == 'joint':
            
            q_t = np.zeros((len(self.t_eval), self.robot.n))
            q_ = np.zeros((len(self.t_eval), self.robot.n))
            x_t = np.zeros((len(self.t_eval), 6))
            # scale = (-np.cos(np.linspace(0, np.pi, len(self.t_eval)))+1)/2 # smooth displacement profile (not physically accurate)
            
            coeffs = solve_fifth_order_polynomial(0, 0, 0, 0, 1, 1, 0, 0)
            def polynomial(x):
                return sum(c * x**i for i, c in enumerate(coeffs[::-1]))
            scale = polynomial(np.linspace(0, 1, len(self.t_eval)))
            if self.q_0 is None:
                self.q_0 = self.IK_fun(self.x_0, sol=self.sol)
            if self.q_f is None:
                self.q_f = self.IK_fun(self.x_f, sol=self.sol)
            for i in range(self.robot.n):
                q_t[:, i] = self.q_0[i] + (self.q_f[i] - self.q_0[i])*scale
            q_ = q_t
            
            self.q_traj = q_
            self.q_traj = q_t
            self.dq_traj = np.gradient(q_t,self.t_eval, axis=0)

        if self.trajectory_type == 'cartesian':
            x_t = np.zeros((len(self.t_eval), 6))
            # scale = (-np.cos(np.linspace(0, np.pi, len(self.t_eval)))+1)/2
            coeffs = solve_fifth_order_polynomial(0, 0, 0, 0, 1, 1, 0, 0)
            def polynomial(x):
                return sum(c * x**i for i, c in enumerate(coeffs[::-1]))
            scale = polynomial(np.linspace(0, 1, len(self.t_eval)))
            for i in range(6):
                x_t[:, i] = self.x_0[i] + (self.x_f[i] - self.x_0[i])*scale
            self.x_traj = x_t
            self.dx_traj = np.gradient(x_t,self.t_eval, axis=0)
            
            q_traj = np.zeros((len(self.t_eval), self.robot.n))
            if self.IK_fun is None:
                print('No IK function provided. Using default function')
                inv = InverseKinematics(self.robot) # inverse kinematics object
                for i in range(len(self.t_eval)):
                    q_traj[i] = inv.minimize_error(self.x_traj[i], q_0 = q_traj[i-1] if i > 0 else None)
            else:
                print('Using provided IK function')
                for i in range(len(self.t_eval)):
                    q_traj[i] = self.IK_fun(self.x_traj[i],sol=self.sol)
            
            self.q_traj = q_traj
            self.dq_traj = np.gradient(q_traj,self.t_eval, axis=0)
        
        self.trajectory_polyline()
            
            
        
    def trajectory_polyline(self):
        """Generates a polyline from the trajectory points for visualization.

        Returns:
            pyvista mesh: polyline
        """
        points = np.zeros((len(self.t_eval), 3))
        for i in range(len(self.t_eval)):
            self.robot.q = self.q_traj[i]
            points[i] = self.robot.fkine(tool_transform = self.tool_T)[:3,3]
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points)-1, 3), 2, dtype=int)
        cells[:, 1] = np.arange(0, len(points)-1, dtype=int)
        cells[:, 2] = np.arange(1, len(points), dtype=int)
        poly.lines = cells
        tube = poly.tube(radius=5)
        self.get_T_lists()
        return tube
    
    def get_T_lists(self):
        """Returns the list of transformation matrices for each joint configuration in the trajectory.

        Returns:
            list: List of transformation matrices
        """
        T_list = []
        for q in self.q_traj:
            self.robot.q = q
            
            T_list.append([self.robot.fkine(i).copy() for i in range(len(self.robot.links))])
        self.T_list = T_list
    
    def get_meshes(self, set_scalars = False, comparison_mesh = None, joint_frames = False):
        """Returns the list of meshes for each joint configuration in the trajectory.

        Returns:
            list: List of meshes
        """
        meshes = []
        print('Generating meshes...')
        try:
            T_list= self.T_list[0]
        except:
            self.get_T_lists()
        max_dist = 0
        
        if comparison_mesh is None and set_scalars:
            raise ValueError('Comparison mesh not provided')
        for i, q in tqdm(enumerate(self.q_traj),total=self.q_traj.shape[0]):
            mesh_ = pv.PolyData()
            T_list = self.T_list[i]
            for j, link in enumerate(self.robot.links):
                # mesh_ += link.stl.copy().transform(T_list[j], inplace=True)
                mesh_ = mesh_.merge(link.stl.copy().transform(T_list[j], inplace=False), merge_points=False)
                if joint_frames:
                    mesh_ += link.frame.copy().transform(T_list[j], inplace=True)
            if set_scalars:
                pos_i = mesh_.points
                pos_f = comparison_mesh.points
                dist = np.linalg.norm(pos_i - pos_f, axis=1)
                if dist.max() > max_dist:
                    max_dist = dist.max()
                mesh_['Scalars'] = dist


                self.clim = [0, max_dist]
            meshes.append(mesh_.copy())
        self.meshes = meshes
    
    def check_validity(self):
        """
        Checks the validity of the trajectory by verifying if the joint angles and speeds are within the specified limits.

        Returns:
            bool: True if the trajectory is valid, False otherwise.
        
        Raises:
            ValueError: If any joint angle or speed is out of range.
        """
        try:
            a = self.dq_traj
        except:
            self.dq_traj = np.gradient(self.q_traj, self.t_eval, axis=0)
        
        for i in range(self.robot.n):
            if np.deg2rad(self.q_traj[:,i].min()) < self.robot.theta_limits[i,0] or np.deg2rad(self.q_traj[:,i].max()) > self.robot.theta_limits[i,1]:
                raise ValueError(f'Trajectory not valid! Joint {i} angle out of range')
        
        for i in range(self.robot.n):
            if np.deg2rad(np.abs(self.dq_traj[:,i]).max()) > self.robot.theta_speed_limits[i]:
                raise ValueError(f'Trajectory not valid! Joint {i} speed out of range')
        
        return True
    
    def set_fps(self, target_fps):
        """
        Sets the frames per second (FPS) for the trajectory.

        Args:
            target_fps (int): The target frames per second.

        Returns:
            None
        """
        current_fps = 1/self.dt
        ratio = int(current_fps/target_fps)
        self.t = self.t_eval[::ratio]
        self.q = self.q_traj = self.q[::ratio]
        self.dt = self.t[1] - self.t[0]
        
        
    def set_tool(self, tool):
        """
        Sets the tool for the robot.

        Args:
            tool (Tool): The tool object to be set.

        Returns:
            None
        """
        self.tool = tool
        
        
        
    def crop_trajectory(self, start_time, end_time):
        """
        Crops the trajectory to the specified start and end times.

        Args:
            start_time (float): The start time for cropping.
            end_time (float): The end time for cropping.

        Returns:
            None
        """
        start_index = np.argmin(np.abs(self.t_eval - start_time))
        end_index = np.argmin(np.abs(self.t_eval - end_time))
        t = self.t_eval[start_index:end_index]
        q = self.q[start_index:end_index]
        new_traj = Trajectory(self.robot, trajectory_type=self.trajectory_type, t_eval=t, generate=False)
        new_traj.q = q
        return new_traj
    
    def plot_joint(self, title = 'Joint Trajectory'):
        """Plots the joint trajectory.
        
        Args:
            title (str, optional): Plot title. Defaults to 'Joint Trajectory'.
        """
        fig, ax = plt.subplots(1,1, dpi=300)
        for i in range(self.robot.n):
           ax.plot(self.t_eval, self.q_traj[:,i], label=f'$\\theta_{i+1}$')
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint Angle  [deg]')
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.show()
        
        
        
    def plot_cartesian(self, padding = 100, aspect = 'auto', title = 'Cartesian Trajectory'):
        """Plots the cartesian trajectory in 3D and 2D planes.

        Args:
            padding (float, optional): Padding around data. Defaults to 100.
            aspect (strin, optional): Axes aspect ratio. 'equal' or 'auto'. Defaults to 'auto'.
            title (str, optional): Plot title. Defaults to 'Cartesian Trajectory'.
        """        
        points = np.zeros((len(self.t_eval), 3))
        for i in range(len(self.t_eval)):
            try:
                self.robot.q = self.q_traj[i]
                points[i] = self.robot.fkine(tool_transform = self.tool_T)[:3,3]
            except:
                points[i] = self.x_traj[i][:3]
        
        x_lim = [points[:,0].min()-padding, points[:,0].max()+padding]
        y_lim = [points[:,1].min()-padding, points[:,1].max()+padding]
        z_lim = [points[:,2].min()-padding, points[:,2].max()+padding]
        min_lim = min([x_lim[0], y_lim[0], z_lim[0]])
        max_lim = max([x_lim[1], y_lim[1], z_lim[1]])
        with plt.style.context('default'):
            fig = plt.figure(figsize=(8,6),)
            ax1 = fig.add_subplot(221)
            ax1.plot(points[:,0], points[:,2])
            ax1.set_xlim(x_lim)
            ax1.set_ylim(z_lim)
            ax1.set_title('X-Z plane')
            ax1.set_aspect(aspect)
            ax1.grid()
            
            ax2 = fig.add_subplot(222, sharey=ax1)
            ax2.plot(points[:,1], points[:,2])
            ax2.set_xlim(y_lim)
            ax2.set_ylim(z_lim)
            ax2.set_title('Y-Z plane')
            ax2.set_aspect(aspect)
            ax2.grid()
            
            ax3 = fig.add_subplot(223, sharex=ax1)
            ax3.plot(points[:,0], points[:,1])
            ax3.set_xlim(x_lim)
            ax3.set_ylim(y_lim)
            ax3.set_title('X-Y plane')
            ax3.set_aspect(aspect)
            ax3.grid()
            
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.plot(points[:,0], points[:,1], points[:,2])
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.set_xlim(x_lim)
            ax4.set_ylim(y_lim)
            ax4.set_zlim(z_lim)
            ax4.set_aspect('equal', adjustable='box')
            
            plt.suptitle(title)
            plt.tight_layout()

            plt.show()


def solve_fifth_order_polynomial(a, Pa, dPa, ddPa, b, Pb, dPb, ddPb):
    """
    Solve a fifth-order polynomial equation based on given conditions.

    Parameters:
    a (float): The value of a.
    Pa (float): The value of P(a).
    dPa (float): The value of P'(a).
    ddPa (float): The value of P''(a).
    b (float): The value of b.
    Pb (float): The value of P(b).
    dPb (float): The value of P'(b).
    ddPb (float): The value of P''(b).

    Returns:
    list: A list of coefficients [c5, c4, c3, c2, c1, c0] of the fifth-order polynomial equation P(x) = c5 * x^5 + c4 * x^4 + c3 * x^3 + c2 * x^2 + c1 * x + c0.

    """
    
    # Coefficients: c5, c4, c3, c2, c1, c0
    # P(x) = c5 * x^5 + c4 * x^4 + c3 * x^3 + c2 * x^2 + c1 * x + c0
    # Equations based on given conditions:
    # P(a) = Pa
    # P(b) = Pb
    # P'(a) = 0
    # P'(b) = 0
    # P''(a) = 0
    # P''(b) = 0
    
    # Construct the matrix A and the vector B
    A = np.array([
        [a**5, a**4, a**3, a**2, a, 1],
        [b**5, b**4, b**3, b**2, b, 1],
        [5*a**4, 4*a**3, 3*a**2, 2*a, 1, 0],
        [5*b**4, 4*b**3, 3*b**2, 2*b, 1, 0],
        [20*a**3, 12*a**2, 6*a, 2, 0, 0],
        [20*b**3, 12*b**2, 6*b, 2, 0, 0]
    ])
    
    B = np.array([Pa, Pb, dPa, dPb, ddPa, ddPb])
    
    # Solve the system of linear equations
    coeffs = np.linalg.solve(A, B)
    
    return coeffs

def get_frame(p, T, scale=1.0, name_='frame', axes = 'xyz'):
    """Adds a frame to the pyvista plot.
    """
    scale *= 202
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]
    x_arrow = pv.Arrow(start=origin, direction=x_axis, scale=scale)
    y_arrow = pv.Arrow(start=origin, direction=y_axis, scale=scale)
    z_arrow = pv.Arrow(start=origin, direction=z_axis, scale=scale)
    if 'x' in axes:
        p.add_mesh(x_arrow, color='red', name=name_+'x_axis')
    if 'y' in axes:
        p.add_mesh(y_arrow, color='green', name=name_+'y_axis')
    if 'z' in axes:
        p.add_mesh(z_arrow, color='blue', name=name_+'z_axis')