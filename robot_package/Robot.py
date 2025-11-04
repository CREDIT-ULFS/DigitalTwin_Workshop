import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sym
import pyvista as pv
import pyvistaqt as pvqt
from tabulate import tabulate
from numpy import sin as s, cos as c, pi
import symengine
from tqdm import tqdm

from .Robot_plotter import RobotPlotter
from .Inverse_Kinematics import InverseKinematics

class Robot:
    def __init__(self,
                 links: list = None,
                 name = 'Robot',
                 media_dir = None):
        """Creates a Robot object.
        """
        
    #------------------------------------------------------------------------------
    
        n = 0
        for link in links:
            n += 1 if link.type != 'fixed' else 0
        self.n = n # number of joints(DoF)
        
        
        
        # Robot properties
        self.links = links
        self.name = name
        self.media_dir = media_dir
        self.IK_sol = None
        
        self._q = np.zeros(self.n)
        self.qd = np.zeros(self.n)
        self.qdd = np.zeros(self.n)
        
        self.q_lim = np.zeros((self.n, 2))
        
        q_lim = np.array([])
        q_speed_lim = np.array([])
        for link in self.links:
            if link.type != 'fixed':
                q_lim = np.append(q_lim, link.theta_limit)
                q_speed_lim = np.append(q_speed_lim, link.theta_speed_limit)
        q_lim = q_lim.reshape((self.n, 2))
        self.theta_limits = np.deg2rad(q_lim)
        self.theta_speed_limits = np.deg2rad(q_speed_lim)
        
        
    #------------------------------------------------------------------------------
        # Kinematic properties
        self.T_list = []
        self.T_list_sym = []
        
        self.sym_scale = 1
        
 
    #------------------------------------------------------------------------------
        # Visualizer properties
        self.meshes = []
        for link in self.links:
            self.meshes.append(link.stl)
            
        self.plotter = None
        
    #------------------------------------------------------------------------------
        # Connect the links
        for i, link in enumerate(links):
            if i == 0:
                link.parent = None
                link.type = 'fixed'
            else:
                link.parent = links[i-1]
            if i == len(links)-1:
                link.child = None
            else:
                link.child = links[i+1]
            link.index = i
            link.set_theta_sym()
            
        self.q_sym = [l.q_sym for l in links if l.type != 'fixed']
        self.dq_sym = [l.dq_sym for l in links if l.type != 'fixed']
        self.ddq_sym = [l.ddq_sym for l in links if l.type != 'fixed']
        self.theta_sym = [l.theta_sym for l in links if l.type != 'fixed']
        self.dtheta_sym = [l.dtheta_sym for l in links if l.type != 'fixed']
        self.ddtheta_sym = [l.ddtheta_sym for l in links if l.type != 'fixed']
        
        self.J_v_fun = None

        try:
            self.compute_joint_frames()
        except:
            print('Could not compute joint frames')
        # self.T_list_sym = self.compute_link_frames_sym(scale = sym.Rational(1,1000))
    #------------------------------------------------------------------------------
    @property
    def q(self):
        """"Getter for the joint angles in degrees."""
        return np.rad2deg(self._q)
    @q.setter
    def q(self, value):
        """Setter for the joint angles in degrees. Checks if the joint angles are within the limits and converts them to radians."""
        value = np.deg2rad(value)
        if len(value) != self.n:
            raise ValueError('Invalid number of joint angles')
        for i, q in enumerate(value):
            if q < self.theta_limits[i, 0] or q > self.theta_limits[i, 1]:
                print(f'Joint {i+1} angle out of range, setting to limit value')
                value[i] = np.clip(q, self.theta_limits[i, 0], self.theta_limits[i, 1])
            if self.links[i+1].type == 'prismatic':
                value[i] = np.rad2deg(value[i])
        self._q = value
        # if self.links[0].type == 'fixed':
        #     for i, link in enumerate(self.links[1:]) if self.links[-1].type != 'fixed' else enumerate(self.links[1:-1]):
        #         if link.type != 'fixed':
        #             link.theta = value[i]
        j=0
        for i, link in enumerate(self.links):
            if link.type != 'fixed':
                link.theta = value[j]
                j += 1

        
    def __str__(self):
        """Pretty print the robot parameters."""
        basic_params = f'Robot: {self.name}, DoF = {self.n}\n' + tabulate([[link.index, link.name, link.type, f'{np.rad2deg(link.theta):.2f}', link.a, f'{np.rad2deg(link.alpha):.2f}', link.d, np.rad2deg(link.theta_offset), link.flip_rotation, link.theta_limit, link.theta_speed_limit] for link in self.links], 
                        headers=['Index', 'Name','Type','θ [°]','a', 'α [°]', 'd',  'θ offset [°]', 'flip rotation',  'Joint limits [°]', 'Joint speed limits [°/s]'], tablefmt='pretty')+'\n'  
        dynamic_params = f'Mass properties:\n' + tabulate([[link.index, link.name, link.mass, np.round(link.CM,3), (link.inertia_matrix[0,0]*1e3).round(5), (link.inertia_matrix[1,1]*1e3).round(5), (link.inertia_matrix[2,2]*1e3).round(5),(link.inertia_matrix[0,1]*1e3).round(5), (link.inertia_matrix[0,2]*1e3).round(5), (link.inertia_matrix[1,2]*1e3).round(5)] for link in self.links],
                        headers=['Index', 'Name', 'Mass [kg]', 'CoM [m]', 'Ixx  [10⁻³ kg.m²]', 'Iyy [10⁻³ kg.m²]', 'Izz [10⁻³ kg.m²]', 'Ixy [10⁻³ kg.m²]', 'Ixz [10⁻³ kg.m²]', 'Iyz [10⁻³ kg.m²]'], tablefmt='pretty')+'\n'
        return basic_params + dynamic_params
    
    def __repr__(self):
        return self.__str__()
    
    def __copy__(self):
        return Robot(links = self.links, name = self.name)
    
#--------------------------------------------------------------------------------
# kinematics
    
    def compute_joint_frames(self):
        """Computes the frames of all the joints of the robot for a given state of the joints.

        Args:
            q (array): 1D array of shape (n,) where each element represents the state of a joint in radians.

        Returns:
            list: List of 4x4 transformation matrices representing the frames of the joints.
        """
        T_list = []
        
        for i, link in enumerate(self.links):
            if i == 0:
                T_list.append(link.A())
            else:
                T = T_list[-1] @ link.A()
                T_list.append(T)
        if len(self.T_list_sym) == 0:
            self.compute_link_frames_sym()
        T_list = [symengine.lambdify([self.q_sym], T, real=True, cse=True) for T in self.T_list_sym]
        
        self.T_list = T_list
    
    def compute_link_frames_sym(self, scale = 1):
        """Computes the frames of all the links of the robot for a given state of the joints.

        Args:
            q (array): 1D array of shape (n,) where each element represents the state of a joint in radians.

        Returns:
            list: List of 4x4 transformation matrices representing the frames of the links.
        """
        self.T_list_sym = []
        print('Computing link frames')
        with tqdm(total = len(self.links)) as pbar:
            for i, link in enumerate(self.links):
                if i == 0:
                    self.T_list_sym.append(sym.nsimplify(link.A_sym(evalf=False, scale = scale), tolerance=1e-10))
                else:
                    # T = sym.simplify(sym.nsimplify(self.T_list_sym[-1] * link.A_sym(evalf=False, scale = scale), tolerance=1e-10))
                    T = sym.nsimplify(self.T_list_sym[-1] * link.A_sym(evalf=False, scale = scale), tolerance=1e-10)
                    
                    self.T_list_sym.append(T)
                pbar.update(1)
                # print(f'Link {i} done', end = '\r')
        self.sym_scale = scale
        return self.T_list_sym
    
    def fkine(self, joint=None, tool_transform = None):
        """Computes the transformation matrix of the end effector or a specific joint of the robot for a given state of the joints.

        Args:
            joint (int, optional): Index of desired joint forward kinematics. If None, the end effector transformation matrix is returned. Defaults to None.

        Returns:
            array: 4x4 transformation matrix
        """
        if joint is None:
            joint = len(self.links)-1
        if len(self.T_list) == 0:
            self.compute_joint_frames()
        # self.compute_joint_frames()
        if tool_transform is None:
            tool_transform = np.eye(4)
        return np.array(self.T_list[joint](*np.deg2rad(self.q))).reshape(4,4) @ tool_transform
        # return self.T_list[joint]
    
    def fkine_sym(self, joint=None, simplify = False, evalf = False, scale = 1):
        """Same as fkine but returns a symbolic transformation matrix.
        if simplify is True, the transformation matrix is simplified.
        if evalf is True, the transformation matrix is evaluated to a numerical value.
        """
        if (scale != self.sym_scale) or (len(self.T_list_sym) == 0):
            self.compute_link_frames_sym(scale = scale)
        if joint is None:
            joint = len(self.links)-1
        if evalf:
            T = self.T_list_sym[joint].subs({self.q_sym[i]: np.deg2rad(self.q[i]) for i in range(len(self.q_sym))})
            if simplify:
                return sym.nsimplify(sym.simplify(T), tolerance=1e-9)
            else:
                return sym.nsimplify(T, tolerance=1e-9)
                    
        if simplify:
            return self.T_list_sym[joint]
        else:
            return self.T_list_sym[joint]
    
#--------------------------------------------------------------------------------
# Differential kinematics

    # translational jacobian
    def jacobian_v_sym(self, simplify = False, joint = None, point = None, scale = 1):
        if joint is None:
            joint = self.n
            
        if point is None:
            point = np.zeros(3)
        point = sym.Matrix(point)
        T = self.fkine_sym(joint=joint, scale=scale)
        R = T[:3,:3].copy()
        p_add = R @ point
        x = T[0,3] + p_add[0]
        y = T[1,3] + p_add[1]
        z = T[2,3] + p_add[2]

        j_v = sym.Matrix([[x],[y],[z]])
        j_v = j_v.jacobian([self.q_sym])
        if simplify:
            return sym.simplify(sym.Matrix(j_v))
        else:
            return sym.Matrix(j_v)
    
    def jacobian_v_sym2(self, simplify = False, joint = None, point = None, scale = 1, evalf = False):
        if joint is None:
            joint = self.n
        if point is None:
            point = np.zeros(3)
        point = sym.Matrix(point)
        T = self.fkine_sym(joint=joint, scale=scale)
        # T = self.T_list_sym[joint].copy()
        R = T[:3,:3].copy()
        p_add = R @ point
        P = T[:3,3].copy()
        # print('P', P)
        # print('point', point)
        P = P + p_add
        
        J = sym.zeros(3, len(self.q_sym))
            
        for i in range(0, joint):
            T_prev = self.T_list_sym[i].copy()
            z_prev = T_prev[:3,2].copy()
            o_prev = T_prev[:3,3].copy()
            J[:,i] = z_prev.cross(P - o_prev)
        if simplify:
            if evalf:
                J = J.xreplace({self.q_sym[i]: np.deg2rad(self.q[i]) for i in range(len(self.q_sym))})
                return sym.Matrix(J)
            else:
                return sym.Matrix(J)
        else:
            if evalf:
                J = J.subs({self.q_sym[i]: np.deg2rad(self.q[i]) for i in range(len(self.q_sym))})
                return J
            else:
                return J
            
    def set_jacobian_v_fun(self):
        """Sets the jacobian function for the robot.
        """
        p_sym = sym.symbols(['px', 'py', 'pz'])
        Scale_sym = sym.Symbol('Scale')
        J_list = []
        for i in tqdm(range(len(self.q_sym)+1)):
            J_list.append(sym.lambdify([self.q_sym, p_sym, Scale_sym], self.jacobian_v_sym2(joint = i, point = p_sym, scale = Scale_sym), modules='numpy', cse=True))
        self.J_v_fun_list = J_list
        def J_v_fun(q, joint, point = None, scale = 1):
            J_v = self.J_v_fun_list[joint](np.deg2rad(q), point, scale)
            return J_v
        self.J_v_fun = J_v_fun
        
    # rotational jacobian
    def jacobian_w_sym(self, simplify = False, joint = None, scale = 1):
        if joint is None:
            joint = self.n
        J = sym.zeros(3, len(self.q_sym))
        for i in range(joint):
            T = self.fkine_sym(i, scale=scale)
            J_O_i = T[:3,2]
            J[:,i] = J_O_i
        if simplify:
            return sym.simplify(sym.Matrix(J))
        else:
            return sym.Matrix(J)
    
            
    
    def jacobian_sym(self, evalf = False, simplify = False, joint = None, scale = 1, point = None):
        """Analytically computes the jacobian of the robot end effector.
        if evalf is True, the jacobian is evaluated to a numerical value.
        if simplify is True, the jacobian is simplified.
        """
        j_v = self.jacobian_v_sym2(simplify = simplify, joint = joint, scale = scale, point = point)
        j_w = self.jacobian_w_sym(simplify = simplify, joint = joint, scale = scale)
        j = sym.Matrix.vstack(j_v, j_w)
        if evalf:
            j = j.subs({self.q_sym[i]: np.deg2rad(self.q[i]) for i in range(len(self.q_sym))})
            return j
        return sym.Matrix(j)
        
        
    
#--------------------------------------------------------------------------------
# plotter
    def plot(self, **kwargs):
        """Plots the robot in its current configuration.
        kvargs are passed to the RobotPlotter class and can include:
        joint_frames: bool, show the frames of the joints
        EE_frame: bool, show the frame of the end effector
        origin: bool, show the origin of the robot
        CoM: bool, show the center of mass of the links
        """
        self.plotter = RobotPlotter(robot = self, **kwargs)
        
    def update(self, q = None, x = None):
        """Updates the robot configuration and the plot.

        Args:
            q (list or array, optional): List of joint variables. Defaults to None.
            x (list or array, optional): List of desired end-effector position and orinetation. Defaults to None.
        If q is provided, the robot is updated to the given joint configuration.
        If x is provided, the robot is updated to the configuration that achieves the desired end-effector position and orientation.
        """
        if self.plotter is None:
            self.plot()
            
        if q is not None and x is not None:
            raise ValueError('Invalid arguments. Use either q or x, not both.')
        
        if q is not None: 
            if np.allclose(self.q, q) :
                print('Robot is already in the desired configuration')
                return       
            self.q = q
            self.plotter.add_meshes()
            
        if x is not None:
            if self.IK_sol is None:
                print('Using default numerical Inverse Kinematics solver')
                self.IK_sol_num = InverseKinematics(self)
                inv = InverseKinematics(self)
                self.q = inv.minimize_error(x,q_0 = self.q)
            else:
                print('Using supplied Inverse Kinematics solver')
                self.q = self.IK_sol(x)
            self.plotter.add_meshes()
        
        if self.plotter.traj_actor is not None:
            self.plotter.traj_actor.SetVisibility(False)
        self.plotter.plot.clear_slider_widgets()
            
        
    def animate(self, trajectory, **kwargs):
        """Animates the robot along the given trajectory.

        Args:
            trajectory (Instance of Trajectory): Trajectory to animate.
        if kwargs are provided, they are passed to the RobotPlotter.animate method and can include:
        plot_trajectory: bool, if True, the trajectory is plotted.
        save_video: bool, if True, the animation is saved as a video.
        video_file: str, name of the video file. Defaults to 'robot_animation.mp4'.
        time_slider: bool, if True, a time slider is added to the plot.
        FPS: int, frames per second of the video. Defaults to 30.
        print_time: bool, if True, the time is printed on the plot.
        """
        if self.plotter is None:
            self.plot()
        self.plotter.animate(trajectory, **kwargs)
    

    def add_point(self, point, **kwargs):
        """Adds a point to the plot.
        """
        if self.plotter is None:
            self.plot()
        sphere = pv.Sphere(radius = 20, center = point)
        self.plotter.add_mesh(sphere)
        
#--------------------------------------------------------------------------------
    
    def save_stl(self, filename = 'robot.stl'):
        """Returns the stl files of the robot links.
        """
        mesh =  self.plotter.add_meshes(return_stl = True)
        pv.save_meshio(filename, mesh)
        print('STL file saved at: ', filename)

