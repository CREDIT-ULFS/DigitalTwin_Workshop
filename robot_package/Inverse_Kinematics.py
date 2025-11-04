import numpy as np
import sympy as sym
import scipy as sp


class InverseKinematics:
    """Class for solving the inverse kinematics of a robot (simple implementation)
    """
    def __init__(self, 
                 robot):
        self.robot = robot
        
        
    def error_function(self, q, x_d):
        """Returns the error between the desired and actual end-effector pose for use in the optimization algorithm.
        """
        self.robot.q = q
        T = self.robot.fkine()
        p = T[0:3,3]
        r = T[0:3,0:3]
        alpha, beta, gamma = euler_from_matrix(r)
        x = np.concatenate((p, np.array([alpha, beta, gamma])))
        return x - x_d
        
    def minimize_error(self, x_d, q_0 = None):
        """Minimizes the error between the desired and actual end-effector pose.
        """
        if q_0 is None:
            q_0 = np.zeros(self.robot.n)
        q = sp.optimize.least_squares(self.error_function, q_0, args = (x_d,))['x']
        return q
    
        
    
    
    
def euler_from_matrix(R):
    """Return Euler angles from rotation matrix for specified axis sequence."""
    phi = np.arctan2(R[1,0], R[0,0])
    theta = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    psi = np.arctan2(R[2,1], R[2,2])
    return np.array([psi, theta, phi])
    
