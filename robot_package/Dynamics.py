import numpy as np
import scipy as sp
import sympy as sym
import sympy.physics.mechanics as me
from .Trajectory import Trajectory
from sympy.utilities.autowrap import ufuncify, autowrap
import matplotlib.pyplot as plt
import dill
from tqdm import tqdm


class RobotDynamics:
    def __init__(self, 
                robot,
                units = 'mm',
                K = None,
                C = None,
                g = np.array([0,0,9.81]),
                verbose=False,
                generate_dynamics = True):
        
        self.robot = robot
        self.theta_sym  = robot.theta_sym
        self.dtheta_sym = robot.dtheta_sym
        self.q_sym      = robot.q_sym
        self.dq_sym     = robot.dq_sym
        if units not in ['mm','m']:
            raise ValueError('Invalid units. Choose between "mm" and "m"')
        self.units = units
        self.verbose = verbose
        self.g0 = sym.Matrix(np.append(g,0))
        
        if K is None:
            K = np.ones(robot.n)
        if len(K) != robot.n:
            raise ValueError('Number of stiffness values must be equal to the number of joints')
        self.K = np.diag(K)
        self.K_sym = sym.symbols('K1:%d'%(robot.n+1))
        
        if C is None:
            C = np.zeros(robot.n)
        if len(C) != robot.n:
            raise ValueError('Number of damping values must be equal to the number of joints')
        self.C = np.array(C)
        
        self.old_q = np.ones(robot.n)
        self.old_K = np.ones(robot.n)
        self.EOM_fun = None
        self.M_correction = np.zeros((robot.n, robot.n))
        
        if generate_dynamics:
            self.get_lagrangian()
    
        
    def import_model(self, M_f, M_inv_f, C_f, g_f, params, stiffness_fun = None):
        self.params = params
        def M_fun(q):
            return np.array(M_f(*q, *self.params)).reshape((len(q),len(q))) + self.M_correction
        def M_inv_fun(q):
            return np.array(M_inv_f(*q, *self.params)).reshape((len(q),len(q)))
        def C_fun(q,dq):
            return np.array(C_f(*q,*dq, *self.params)).reshape((len(q),len(q)))
        def g_fun(q):
            return np.array(g_f(*q, *self.params)).reshape((len(q),1))
        self.mass_matrix_fun = M_fun
        self.mass_matrix_inv_fun = M_inv_fun
        self.C_fun = C_fun
        self.g_fun = g_fun
        self.stiffness_fun = stiffness_fun
        def rhs_fun(q,dq,theta,dtheta,K,tau=None):
            if tau is None:
                tau = np.zeros((len(q),0))
            q_v = q.reshape((len(q),1))
            dq_v = dq.reshape((len(q),1))
            theta_v = theta.reshape((len(q),1))
            dtheta_v = dtheta.reshape((len(q),1))
            # print(K[5])
            P1 = (-self.mass_matrix_inv_fun(q)@(-self.C_fun(q,dq)+np.diag(self.C)))@dq_v + (self.mass_matrix_inv_fun(q)@np.diag(self.C))@dtheta_v +self.mass_matrix_inv_fun(q)@self.g_fun(q)
            P2 = -(self.mass_matrix_inv_fun(q)) @ K @ (q_v-theta_v)  

            full = np.vstack((dq_v,P1+P2))
            return full
        self.rhs_lambda = rhs_fun
        
        def torque_fun(q,dq,ddq):
            q_v = q.reshape((len(q),1))
            dq_v = dq.reshape((len(q),1))
            ddq_v = ddq.reshape((len(q),1))

            return self.mass_matrix_fun(q)@ddq_v*0 + self.C_fun(q,dq)@dq_v + self.g_fun(q)
        self.torque_lambda = torque_fun
            
        
        
    def get_kinetic_energy(self):
        # Kinetic energy
        T = 0
        dq = sym.Matrix(self.robot.dq_sym)
        for i in range(1,len(self.robot.links)):
            p_com = self.robot.links[i].CM * sym.Rational(1,1000) if self.units == 'mm' else self.robot.links[i].CM
            if self.robot.links[i].type in ['revolute','prismatic']:
                # print(1)
                Jp = self.robot.jacobian_v_sym2(simplify=False,joint=i, point = p_com, scale = sym.Rational(1,1000) if self.units == 'mm' else 1)
                # print(2)
                Jw = self.robot.jacobian_w_sym(joint=i, simplify=False, scale = sym.Rational(1,1000) if self.units == 'mm' else 1)
                # print(3)
            if self.robot.links[i].type == 'fixed':
                Jp = self.robot.jacobian_v_sym(simplify=True,joint=i, point = p_com, scale = sym.Rational(1,1000) if self.units == 'mm' else 1)
                Jw = self.robot.jacobian_w_sym(joint=i-1, simplify=True, scale = sym.Rational(1,1000) if self.units == 'mm' else 1)
            m = self.robot.links[i].mass
            T += sym.nsimplify((sym.Rational(1,2)*m*dq.T@Jp.T@Jp@dq)[0], tolerance = 1e-5) # translational kinetic energy of center of mass
            # print(4)
            I = sym.Matrix(self.robot.links[i].inertia_matrix)
            FK = self.robot.fkine_sym(joint=i, scale = sym.Rational(1,1000) if self.units == 'mm' else 1).copy()
            # print(5)
            R = FK[:3,:3]
            T += sym.nsimplify((sym.Rational(1,2)*dq.T@Jw.T@R@I@R.T@Jw@dq)[0],tolerance=1e-5) # rotational kinetic energy
            # print(6)
            if self.verbose:
                print(f'Link{i}:')
                print(f'Mass: {m:.2f}')
                print(f'CoM: {p_com}')
                
            print(f'Link {i} done.')
            
            
        # return sym.simplify(T)
        return T
    
    # def get_potential_energy(self):
    #     U = 0
    #     K = self.K_sym
    #     for i in range(1,len(self.robot.links)):
    #         link = self.robot.links[i]
    #         q = link.q_sym
    #         m = link.mass
    #         p_com = np.array(np.ones(4),ndmin=2, dtype=object).T
    #         p_com[:3,0] = link.CM * sym.Rational(1,1000) if self.units == 'mm' else link.CM
    #         FK = self.robot.fkine_sym(joint=i).copy()
    #         FK[:3,3] = FK[:3,3]*sym.Rational(1,1000) if self.units == 'mm' else FK[:3,3]
    #         p = FK@p_com*sym.Rational(1,1000) if self.units == 'mm' else FK@p_com
    #         U += sym.nsimplify((m * self.g0.T @ p)[0],tolerance=10**-14)
    #         if link.type != 'fixed':
    #             U += sym.nsimplify((sym.Rational(1,2)*K[i-1]*(q-self.theta_sym[i-1])**2),tolerance=10**-14)
    #             if self.verbose:
    #                 print(f'Link {i-1}')
    #                 print(f'K: {K[i-1]}')
    #     return U
    
    def get_potential_energy(self):
        U = 0
        K = self.K_sym
        for i in range(1,len(self.robot.links)):
            link = self.robot.links[i]
            q = link.q_sym
            m = link.mass
            p_com = np.array(np.ones(4),ndmin=2, dtype=object).T
            p_com[:3,0] = link.CM * sym.Rational(1,1000) if self.units == 'mm' else link.CM
            FK = self.robot.fkine_sym(joint=i, scale = sym.Rational(1,1000) if self.units == 'mm' else 1).copy()
            
            
            p = FK@p_com
            U += sym.nsimplify((-m * self.g0.T @ p)[0],tolerance=10**-14)
            if link.type != 'fixed':
                U += sym.nsimplify((sym.Rational(1,2)*K[i-1]*(q-self.theta_sym[i-1])**2),tolerance=10**-14)
                if self.verbose:
                    print(f'Link {i-1}')
                    print(f'K: {K[i-1]}')
        return U
    
    def get_lagrangian(self):
        
        print('Calculating Lagrangian...')
        print('This may take a while for large robots.')
        print('Getting kinetic energy...')
        T = self.get_kinetic_energy()
        print('Getting potential energy...')
        U = self.get_potential_energy()
        print('Calculating Lagrangian...')
        # L = sym.simplify(T-U)
        self.L = T-U
        self.LM = me.LagrangesMethod(self.L, self.q_sym)
        print('Getting equations of motion...')
        self.LM.form_lagranges_equations()
        print('Getting mass matrix...')
        self.mass_matrix = self.LM.mass_matrix
        print('Creating numerical functions...')
        self.mass_matrix_fun = sym.lambdify(args=self.q_sym, expr=self.mass_matrix, cse=True, modules='numpy')
        print('Done.')
        return self.L
    
    
    def modal_analysis(self, q=None, verbose=True):
        """
        Perform modal analysis on the robot's dynamics.

        Args:
            q (numpy.ndarray, optional): Joint angles. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Eigenvalues and eigenvectors.

        Raises:
            ValueError: If the number of joint angles is not equal to the number of joints.
        """
        if q is None:
            q = np.zeros(self.robot.n)
            print('No joint angles provided. Defaulting to zero angles...')
        if len(q) != self.robot.n:
            raise ValueError('Number of joint angles must be equal to the number of joints')
        if np.all(q == self.old_q) and np.all(self.K == self.old_K) and self.eigvals is not None:
            if verbose:
                print('Using cached eigenvalues and eigenvectors...')
            return self.eigvals, self.eigvecs
        else:
            if verbose:
                print('New mass or stiffness matrix detected. Recalculating modes...')
        M_fun = self.mass_matrix_fun
        q = np.deg2rad(q).copy()
        try:
            M_inv = self.mass_matrix_inv_fun(q)
        except:
            M_inv = np.linalg.inv(M_fun(*q))
        M_inv = np.array(M_inv).astype(np.float64)
        K = self.K
        A = M_inv@K
        eigvals, eigvecs = np.linalg.eig(A)
        ind = np.argsort(eigvals)
        self.eigvals = eigvals[ind]
        self.eigvecs = eigvecs[:,ind]
        self.old_q = q.copy()
        self.old_K = self.K.copy()
        return self.eigvals, self.eigvecs
    
    def get_natural_frequency_map(self, j1=0, j2=1, q=None):
        """
        Calculates the natural frequency map for the given joint angles.

        Args:
            j1 (int): Index of the first joint angle (default is 0).
            j2 (int): Index of the second joint angle (default is 1).
            q (ndarray, optional): Joint angles (default is None).

        Returns:
            tuple: A tuple containing the following elements:
                - eigen_map (ndarray): Array of natural frequencies for each joint angle combination.
                - th_list_1 (ndarray): Array of joint angles for the first joint.
                - th_list_2 (ndarray): Array of joint angles for the second joint.
        """
        q_lim_1 = np.rad2deg(self.robot.theta_limits[j1].copy())
        q_lim_2 = np.rad2deg(self.robot.theta_limits[j2].copy())
        if np.any(np.abs(q_lim_1) == np.inf) or np.any(np.abs(q_lim_2) == np.inf):
            raise ValueError('Joint limits must be finite')
        if q is None:
            q = np.zeros(self.robot.n)
        q_ = q.copy()
        q_start = q.copy()
        if len(q_) != self.robot.n:
            raise ValueError('Number of joint angles must be equal to the number of joints')
        th_list_1 = np.arange(q_lim_1[0], q_lim_1[1] + 1, 5)
        th_list_2 = np.arange(q_lim_2[0], q_lim_2[1] + 1, 5)
        eigen_map = np.zeros((self.robot.n, len(th_list_1), len(th_list_2)))
        for i in range(len(th_list_1)):
            for j in range(len(th_list_2)):
                q_[j1] = th_list_1[i]
                q_[j2] = th_list_2[j]
                eigvals, _ = self.modal_analysis(q=q_, verbose=False)
                eigen_map[:, i, j] = np.sqrt(eigvals) / 2 / np.pi
        self.robot.q = q_start

        return eigen_map, th_list_1, th_list_2
    
    def plot_natural_frequency_map(self, j1=0, j2=1, q=None, mode=0):
        """
        Plots the natural frequency map for the given joint indices and mode.

        Parameters:
            j1 (int): Index of the first joint (default is 0).
            j2 (int): Index of the second joint (default is 1).
            q (array-like): Joint configuration (default is None).
            mode (int): Mode index (default is 0).

        Returns:
            None
        """
        eigen_map, th_list_1, th_list_2 = self.get_natural_frequency_map(j1=j1, j2=j2, q=q)
        fig, ax = plt.subplots(dpi=300)
        a = ax.contourf(th_list_1, th_list_2, eigen_map[mode].T, cmap='jet', levels=8)
        ax.set_xlabel(f'$\\theta_{j1+1}$ [°]')
        ax.set_ylabel(f'$\\theta_{j2+1}$ [°]')
        ax.set_title(f'{mode+1}. Natural Frequency [Hz]')
        ax.set_aspect('equal')
        ax.grid(alpha=0.4)
        plt.colorbar(a, ax=ax)
        plt.show()
        
        
        
    def create_modeshape(self, q=None, mode=0, scale=1, phase=1):
        """
        Creates a mode shape trajectory based on the given joint angles.

        Args:
            q (numpy.ndarray, optional): Joint angles. If not provided, default to an array of zeros with the same length as the number of joints.
            mode (int, optional): Mode shape index. Defaults to 0.
            scale (float, optional): Scaling factor for the mode shape. Defaults to 1.

        Returns:
            traj (Trajectory): Mode shape trajectory object.

        Raises:
            ValueError: If the number of joint angles is not equal to the number of joints.
        """
        if q is None:
            q = np.zeros(self.robot.n)
        if len(q) != self.robot.n:
            raise ValueError('Number of joint angles must be equal to the number of joints')
        self.robot.q = q
        eigvals, eigvecs = self.modal_analysis(q=q)
        self.old_q = q
        self.old_K = self.K
        mode_shape = eigvecs[:,mode]*phase
        t = np.linspace(0, 2*np.pi, 60)
        traj = Trajectory(self.robot, t_eval=t, generate=False)
        q_traj = np.zeros((len(t), self.robot.n))
        for i in range(len(t)):
            q_traj[i] = q + mode_shape*scale*np.cos(t[i])
        traj.q_traj = q_traj
        ref_mesh = self.robot.plotter.add_meshes(return_stl = True)
        traj.get_meshes(set_scalars=True, comparison_mesh=ref_mesh)
        
        return traj
    
    def get_rhs(self):
        """
        Creates the right hand side function for the dynamics.

        This method calculates the mass matrix and forcing terms, and creates numerical functions
        for the mass matrix and forcing. It also defines the right hand side function `rhs_fun`
        that calculates the dynamics of the system.

        Args:
            None

        Returns:
            None
        """
        print('Creating right hand side function...')
        mass_matrix = self.LM.mass_matrix_full
        forcing = self.LM.forcing_full
        print('Making numerical functions...')
        self.mass_matrix_full_fun = sym.lambdify(args=self.q_sym, expr=mass_matrix, cse=True, modules='numpy')
        forcing_dummy, dummys = dummify_matrix(forcing, [*self.q_sym,*self.dq_sym,*self.theta_sym, *self.K_sym])
        self.forcing_full_fun = sym.lambdify(args=dummys, expr=forcing_dummy, cse=True, modules='numpy')
        def rhs_fun(q,dq,theta,K):
            return np.linalg.inv(self.mass_matrix_full_fun(*q)) @ self.forcing_full_fun(*q,*dq,*theta,*K)
        self.rhs_lambda = rhs_fun
        print('Done.')
        
    def simulate_trajectory(self, traj, EE_force=None, adaptive_stiffness = False):
        """
        Simulates the trajectory of the robot.

        Args:
            traj (Trajectory): The trajectory object containing the desired joint positions and velocities.
            EE_force (ndarray, optional): The end effector forces. Defaults to None.

        Returns:
            Trajectory: The simulated trajectory object.

        Raises:
            ValueError: If the number of time points is not equal to the number of end effector forces.
        """
        if adaptive_stiffness:
            try:
                a = self.stiffness_fun(np.zeros(self.robot.n))
            except:
                raise ValueError('Stiffness function must be defined for adaptive stiffness simulation')
        q_0 = np.deg2rad(traj.q_traj[0])
        dq_0 = np.zeros(self.robot.n)
        t_traj = traj.t_eval
        if EE_force is None:
            EE_force = np.zeros((len(t_traj), 6))
            def J_zero(*args):
                return np.zeros((6, self.robot.n))
            J_fun = J_zero
        else:
            print('Creating Jacobian function...')
            J_fun = sym.lambdify(args=self.q_sym, expr=(self.robot.jacobian_sym()), cse=True)
        if EE_force.shape[0] != len(t_traj):
            raise ValueError('Number of time points must be equal to the number of end effector forces')
        th_interp = sp.interpolate.CubicSpline(traj.t_eval, np.deg2rad(traj.q_traj), axis=0, bc_type='not-a-knot', extrapolate=None)
        dth_interp = sp.interpolate.CubicSpline(traj.t_eval, np.deg2rad(traj.dq_traj), axis=0, bc_type='not-a-knot', extrapolate=None)
        EE_force_interp = sp.interpolate.CubicSpline(traj.t_eval, EE_force, axis=0, bc_type='not-a-knot', extrapolate=None)
        C = np.diag(self.C)
        # def dSdt(t, S):
        #     """
        #     Calculates the derivative of the state vector.

        #     Args:
        #         t (float): The current time.
        #         S (ndarray): The state vector. The first n elements are the joint angles and the next n elements are the joint velocities.

        #     Returns:
        #         ndarray: The derivative of the state vector.
        #     """
        #     # Print t every 1000 iterations
        #     if np.isclose(t % 0.1, 0, atol=1e-3):
        #         print(f'{t / t_traj[-1] * 100:.2f}%', end='\r')
        #     th_t = th_interp(t)
        #     q = S[:self.robot.n]
        #     dq = S[self.robot.n:]
        #     dS = self.rhs_lambda(q, dq, th_t, self.K).T
        #     M_inv = np.linalg.inv(self.mass_matrix_fun(*q))
        #     J = J_fun(*q)
        #     J[:3, :] = J[:3, :] * 1e-3 if self.units == 'mm' else J[:3, :]
        #     Tau = J.T @ EE_force_interp(t)
        #     torque = M_inv @ ((C @ (dq - dth_interp(t))) - Tau)
        #     return dS.flatten() - np.array([*np.zeros(self.robot.n), *torque])
        self.prev_ddq = np.zeros(self.robot.n)
        def dSdt(t, S):
            """
            Calculates the derivative of the state vector.

            Args:
                t (float): The current time.
                S (ndarray): The state vector. The first n elements are the joint angles and the next n elements are the joint velocities.

            Returns:
                ndarray: The derivative of the state vector.
            """
            # Print t every 1000 iterations
            if np.isclose(t % 0.1, 0, atol=1e-2):
                print(f'{t / t_traj[-1] * 100:.2f}%', end='\r')
            th_t = th_interp(t)
            q = S[:self.robot.n]
            dq = S[self.robot.n:]
            if adaptive_stiffness:
                K = self.stiffness_fun(self.torque_lambda(q, dq, self.prev_ddq).flatten()).copy()
                # print(np.max(K), '\t', np.max(self.torque_lambda(q, dq, self.prev_ddq).flatten()))
            else:
                K = self.K.copy()
            
            J = J_fun(*q)
            J[:3, :] = J[:3, :] * 1e-3 if self.units == 'mm' else J[:3, :]
            Tau = J.T @ EE_force_interp(t)
            dS = self.rhs_lambda(q, dq, th_t, dth_interp(t), K, tau=Tau).T
            # M_inv = np.linalg.inv(self.mass_matrix_fun(*q))
            # torque = M_inv @ (-Tau)
            self.prev_ddq = dS.flatten()[self.robot.n:]
            # print(self.prev_ddq)
            return dS.flatten()
        # t_eval = np.arange(t_traj[0], t_traj[-1], 0.001)
        # print('Newtime')
        t_eval = traj.t
        print(f'Last time: {t_eval[-1]:.2f} s')
        sol = sp.integrate.solve_ivp(dSdt, [t_traj[0], t_traj[-1]], [*q_0, *dq_0], t_eval=t_eval, method='RK45')
        
        print('Done.  ', end='\r')
        traj_sim = Trajectory(self.robot, t_eval=sol.t, generate=False, tool_T=traj.tool_T)
        traj_sim.q = np.rad2deg(sol.y[:self.robot.n].T).copy()
        return traj_sim
    
    def get_torques(self, traj, EE_force=None):
        if self.EOM_fun is None:
            print('Creating EOM function... (only need to do this once)')
            EOM = self.LM.eom.copy().subs({self.K_sym[i]:0 for i in range(len(self.K_sym))})
            dummy_EOM, dummys = dummify_matrix(matrix = EOM, args = [*self.robot.q_sym,*self.robot.dq_sym,*self.robot.ddq_sym])
            self.EOM_fun = sym.lambdify(args=dummys, expr=dummy_EOM, cse=True, modules='numpy')
        else:
            print('Using cached EOM function...')
        if EE_force is None:
            EE_force = np.zeros((len(traj.t_eval), 6))
        J_fun = sym.lambdify(args=self.q_sym, expr=(self.robot.jacobian_sym()), cse=True)
        
        q = np.deg2rad(traj.q_traj)
        dq = np.gradient(q,traj.t, axis=0)
        ddq = np.gradient(dq,traj.t, axis=0)
        torques = np.zeros_like(q)
        for i in range(len(q)):
            J = J_fun(*q[i])
            J[:3, :] = J[:3, :] * 1e-3 if self.units == 'mm' else J[:3, :]
            Tau = J.T @ EE_force[i]
            torques[i] = self.EOM_fun(*q[i],*dq[i], *ddq[i]).flatten() - Tau.flatten()
        return torques
    
    def save_model(self, path):
        dill.dump(self, open(path, 'wb'))
        
    
    def get_joint_space_Y(self, q, freq, use_M = True, use_K = True, use_C=True):
        M = self.mass_matrix_fun(np.deg2rad(q))
        K = self.K
        C = np.diag(self.C)
        Y_full = np.zeros((freq.shape[0], M.shape[0], M.shape[0]), dtype=complex)
        use_M = 1 if use_M else 0
        use_K = 1 if use_K else 0
        use_C = 1 if use_C else 0
        for i in range(freq.shape[0]):
            w = 2*np.pi*freq[i]
            if np.isclose(w, 0):
                print(f'Warning: Frequency {freq[i]} is close to zero, using small value to avoid division by zero.')
                w = 1e-1
            Y_full[i] = np.linalg.inv(-w**2*M*use_M + 1j*w*C*use_C + K*use_K) * -1*w**2
        return Y_full
    
        
def dummify_matrix(matrix, args):
    dummys = [sym.Symbol(f'a{i}') for i in range(len(args))]
    sub_dict = {args[i]:dummys[i] for i in range(len(args))}
    matrix = matrix.xreplace(sub_dict)
    return matrix, dummys



        
        
def MAC(A,B):
    return (np.abs(np.dot(A.T.conj(),B))**2/(np.dot(A.T.conj(),A)*np.dot(B.T.conj(),B))).real
    
    
def add_noise(Y,n1 = 2e-2, n2 = 2e-1, n3 = 2e-1 ,n4 = 5e-2):
    """
    Additive noise to synthesized FRFs by random values as per standard normal distribution with defined scaling factors.

    :param n1: amplitude of real part shift scalied with FRF absolute amplitude
    :type n1: float
    :param n2: amplitude of imag part shift scalied with FRF absolute amplitude
    :type n2: float
    :param n3: amplitude of real part shift
    :type n3: float
    :param n4: amplitude of real part shift
    :type n4: float
    """
    rand1 = n1 * np.random.randn(*Y.shape)
    rand2 = n2 * np.random.randn(*Y.shape) * 1j
    rand3 = n3 * np.random.randn(*Y.shape)
    rand4 = n4 * np.random.randn(*Y.shape) * 1j

    noise = np.einsum("ijk,ijk->ijk", np.abs(Y), rand1) + np.einsum("ijk,ijk->ijk", np.abs(Y), rand2) + rand3 + rand4

    Y_noise = Y + noise
    return Y_noise