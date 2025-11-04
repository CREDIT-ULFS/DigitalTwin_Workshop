import numpy as np
import sympy as sym
import sympy.physics.mechanics as me
import pyvista as pv
import pyvistaqt as pvqt
import os
from numpy import sin as s, cos as c, pi
from tabulate import tabulate
import symengine

class Link:
    def __init__(self,
                 name,
                 type = 'revolute',
                 a = 0, 
                 alpha = 0, 
                 d = 0, 
                 theta = 0, 
                 theta_offset = 0,
                 flip_rotation = False, 
                 mass = 0, 
                 CM = np.zeros(3), 
                 inertia_matrix = np.zeros(6),
                 stl_file_path = None,
                 theta_limit = np.array([-np.inf, np.inf]),
                 theta_speed_limit = np.inf,
                 dh_convention = 'standard',
                 decimate_mesh = 0.95):
        
        
            
    #------------------------------------------------------------------------------
      
        # Link properties
        self.name = name # name of the link
        if type not in ['revolute' , 'fixed', 'prismatic']:
            raise ValueError('Invalid joint type')
        self.type = type # type of joint (revolute, fixed(base))
        self.child = None # child link
        self.parent = None # parent link
        self.index = None # index of the joint in the robot
        t = sym.symbols('t')
        self.theta_sym = None
        if dh_convention not in ['standard', 'modified']:
            raise ValueError('Invalid DH convention')
        self.dh_convention = dh_convention
  
        
    #------------------------------------------------------------------------------
        
        
        # Kinematic parameters
        self.a = a # link length
        self.alpha = alpha # link twist
        self.d = d # link offset
        self._theta = theta # joint angle
        self.theta_offset = theta_offset # joint angle offset
        self.flip_rotation = flip_rotation # if True, the joint rotates in the opposite direction
        self.theta_limit = theta_limit # joint angle limits
        self.theta_speed_limit = theta_speed_limit # joint speed limit in deg/s
        

        
    #------------------------------------------------------------------------------
        
        self.link_length = a
    #------------------------------------------------------------------------------
        

        
        #Physical link properties
        self.mass = mass # mass of the link
        self.CM = CM # position of the center of mass with respect to the link frame
        Ixx, Iyy, Izz, Ixy, Iyz, Ixz = inertia_matrix
        self.inertia_matrix = np.array([[Ixx, Ixy, Ixz],
                                        [Ixy, Iyy, Iyz],
                                        [Ixz, Iyz, Izz]]) # inertia matrix of the link
        
    #------------------------------------------------------------------------------
    
        # Visual properties
        # read stl file and create a pyvista mesh
        self.stl_file_path = stl_file_path
        if 'virtual' in name:
            self.stl_file_path = None
        if self.stl_file_path is not None:
            try:
                self.stl = pv.read(self.stl_file_path).decimate(decimate_mesh) # read stl file and decimate the mesh
                # select = self.stl.select_enclosed_points(self.stl)
                # self.stl = self.stl.extract_surface()
            except:
                raise ValueError('Invalid file path')
        else:
            # create a cylinder if no stl file is provided
            try:
                T = self.A()
            except:
                T  = np.eye(4)
            point = T[:3,3].copy()
            if np.linalg.norm(point) < 1e-6:
                self.stl = pv.Sphere(radius=20)
            else:
                line = pv.Line(point, [0,0,0]).transform(np.linalg.pinv(T))
                tube = line.tube(radius=20)
                self.stl = tube
                
            
    #------------------------------------------------------------------------------
    
    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, value):
        if value < self.theta_limit[0] or value > self.theta_limit[1]:
            raise ValueError('Joint value out of range')
        self._theta = value
        

        
    #------------------------------------------------------------------------------
    def __str__(self):
        """Pretty print the robot parameters.

        Returns:
            str: pretty printed string
        """
        return tabulate([[self.name,self.type,f'{np.rad2deg(self.theta):.2f}', self.a, f'{np.rad2deg(self.alpha):.2f}', self.d, np.rad2deg(self.theta_offset), self.flip_rotation, ]], 
                        headers=['Name','Type','θ [°]','a', 'α [°]', 'd',  'θ offset [°]', 'flip rotation','mass [kg]', 'CoM', 'Inertia'], tablefmt='pretty')+'\n'
    def __repr__(self):
        return self.__str__()
    
    def set_theta_sym(self):
        """Set the symbolic variable for the joint angle.
        # """
        # self.theta_sym = me.dynamicsymbols(f'theta{self.index}') #motor angle
        # self.dtheta_sym = me.dynamicsymbols(f'theta{self.index}',1)
        # self.ddtheta_sym = me.dynamicsymbols(f'theta{self.index}',2)
        # self.q_sym = me.dynamicsymbols(f'q{self.index}') #joint angle
        # self.dq_sym = me.dynamicsymbols(f'q{self.index}',1) #joint velocity
        # self.ddq_sym = me.dynamicsymbols(f'q{self.index}',2) #joint acceleration
        
        # normal symbols
        self.theta_sym = symengine.symbols(f'theta{self.index}') #motor angle
        self.dtheta_sym = symengine.symbols(f'dtheta{self.index}')
        self.ddtheta_sym = symengine.symbols(f'ddtheta{self.index}')
        self.q_sym = symengine.symbols(f'q{self.index}')
        self.dq_sym = symengine.symbols(f'dq{self.index}')
        self.ddq_sym = symengine.symbols(f'ddq{self.index}')
    
    def A(self):
        """Computes the transformation matrix of the link.
        """
        alpha = self.alpha
        a = self.a
        d = self.d
        theta = self.theta
        theta = -theta if self.flip_rotation else theta
        
        if self.type == 'fixed':
            theta = 0
            T = np.eye(4)
            
        if self.type == 'prismatic':
            d += np.rad2deg(self.theta)
            theta = 0
        theta +=  self.theta_offset
    
        if self.dh_convention == 'modified':
            T= np.array([[c(theta), -s(theta), 0, a],
                    [s(theta)*c(alpha), c(theta)*c(alpha), -s(alpha), -s(alpha)*d],
                    [s(theta)*s(alpha), c(theta)*s(alpha), c(alpha), c(alpha)*d],
                    [0, 0, 0, 1]])
        else:
            T= np.array([[c(theta), -s(theta)*c(alpha), s(theta)*s(alpha), a*c(theta)],
                    [s(theta), c(theta)*c(alpha), -c(theta)*s(alpha), a*s(theta)],
                    [0, s(alpha), c(alpha), d],
                    [0, 0, 0, 1]])
        return T
    
    def A_sym(self, evalf = False, scale = 1):
        """Computes the symbolic transformation matrix of the link.

        Args:
            evalf (bool, optional): If true, numerical symbolic values are substituted with numerical. Defaults to False.

        Returns:
            sympy.Matrix: symbolic transformation matrix
        """
        t = sym.symbols('t')
        theta = self.q_sym
        theta = -theta if self.flip_rotation else theta
        if self.type == 'fixed':
            theta = 0
        # th_offset = convert_to_symbolic_pi(self.theta_offset)
        # alpha = convert_to_symbolic_pi(self.alpha)
        th_offset = self.theta_offset
        alpha = self.alpha
        a = self.a * scale
        d = self.d * scale
        if self.type == 'prismatic':
            d += self.q_sym
            theta = 0
        
        if self.dh_convention == 'modified':
            T = sym.Matrix([[sym.cos(theta+th_offset),-sym.sin(theta+th_offset),0,a],
                    [sym.sin(theta+th_offset)*sym.cos(alpha),sym.cos(theta+th_offset)*sym.cos(alpha),-sym.sin(alpha),-sym.sin(alpha)*d],
                    [sym.sin(theta+th_offset)*sym.sin(alpha),sym.cos(theta+th_offset)*sym.sin(alpha),sym.cos(alpha),sym.cos(alpha)*d],
                    [0,0,0,1]])
        else:
            T = sym.Matrix([[sym.cos(theta+th_offset),-sym.sin(theta+th_offset)*sym.cos(alpha),sym.sin(theta+th_offset)*sym.sin(alpha),a*sym.cos(theta+th_offset)],
                        [sym.sin(theta+th_offset),sym.cos(theta+th_offset)*sym.cos(alpha),-sym.cos(theta+th_offset)*sym.sin(alpha),a*sym.sin(theta+th_offset)],
                        [0,sym.sin(alpha),sym.cos(alpha),d],
                        [0,0,0,1]])
        if evalf:
            T = T.subs({self.q_sym: self.theta if self.type=='revolute' else np.rad2deg(theta)}).evalf()
        return T
    
            
    
    
    

def create_links(names,
                 types,
                 dh_params,
                 flip_rotation = None, 
                 mass = None, 
                 CM = None, 
                 inertia_matrix = None, 
                 stl_paths = None,
                 theta_limits = None,
                 theta_speed_limits = None,
                 dh_convention = 'standard'):
    """Create a list of links from the given parameters.

    Returns:
        list: list of links
    """
    

    links = []
    for i, name in enumerate(names):
        link = Link(name = name,
                    type = types[i] if types is not None else 'revolute',
                    a = dh_params[i,0],
                    alpha = dh_params[i,1],
                    d = dh_params[i,2],
                    theta_offset = dh_params[i,3],
                    flip_rotation = flip_rotation[i] if flip_rotation is not None else False,
                    mass = mass[i] if mass is not None else 0,
                    CM = CM[i] if CM is not None else np.zeros(3),
                    inertia_matrix = inertia_matrix[i] if inertia_matrix is not None else np.zeros(6),
                    stl_file_path = stl_paths[i] if stl_paths is not None else None,
                    theta_limit = theta_limits[i] if theta_limits is not None else np.array([-np.inf, np.inf]),
                    theta_speed_limit = theta_speed_limits[i] if theta_speed_limits is not None else np.inf,
                    dh_convention = dh_convention)
        links.append(link)
    
    return links



def convert_to_symbolic_pi(number):
    """Converts a numerical value to a symbolic value if it is close to pi or -pi.

    """
    if type(number) == sym.Symbol:
        return number
    if np.isclose(number, np.pi):
        return sym.pi
    if np.isclose(number, -np.pi):
        return -sym.pi
    if np.isclose(number, np.pi/2):
        return sym.pi/2
    if np.isclose(number, -np.pi/2):
        return -sym.pi/2
    else:
        return number
            
        
        