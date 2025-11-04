import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from PyQt5 import  QtGui
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QAction
from time import time, sleep


class RobotPlotter:
    def __init__(self, robot, 
                 joint_frames = False, 
                 EE_frame = False, 
                 origin = False,
                 CoM = False,
                 J_det = False,
                 pos = None,
                 joint_axes = False,
                 floor = [0, 0, 1, 500, 500],
                 robot_opacity = 1.0,
                 GCS = False):
        
    #------------------------------------------------------------------------------
        self.robot = robot
        self.joint_frames = joint_frames
        self.EE_frame = EE_frame
        self.origin = origin
        self.CoM = CoM
        self.J_det = J_det
        self.print_time = False
        self.joint_axes = joint_axes
        self.floor = floor
        self.robot_opacity = robot_opacity
            
        
    #------------------------------------------------------------------------------
        # Initialize the plotter


        self.plot = pvqt.BackgroundPlotter(window_size=(1000, 1000))
        
        self.plot.enable_anti_aliasing()

        self.plot.auto_update = 0.01
        self.plot.app_window.setWindowTitle('Robot Visualizer')
        self.plot.background_color = "#FFFFFF"
        if GCS:
            self.plot.add_axes(line_width=5, labels_off=True)
        self.plot.enable_eye_dome_lighting()
        light = pv.Light((2000, 2000, 2000), (0, 0, 0), 'white',intensity=0.7)
        self.plot.add_light(light)
        light = pv.Light((-2000, -2000, 2000), (0, 0, 0), 'white',intensity=0.7)
        self.plot.add_light(light)
        
        # Add floor
        
        # floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=500, j_size=500)
        floor = pv.Plane(center=(0, 0, 0), direction=(self.floor[:3]), i_size=self.floor[3], j_size=self.floor[4])
        self.plot.add_mesh(floor, color='gray', show_edges=True)
        self.add_meshes()
        self.traj_actor = None
        self.plot.view_isometric()
        self.plot.parallel_projection = True
        pos = self.plot.camera_position if pos is None else pos
    
        self.plot.camera_position = pos
        
        
        self.animate_toolbar = self.plot.app_window.addToolBar('Animate Modeshape')
        self.animation_traj = None
    #------------------------------------------------------------------------------    
        if self.origin:
            add_frame(self.plot, np.eye(4), scale=3, name_='world')
        
        
    
    
    #------------------------------------------------------------------------------
    
    def add_meshes(self, return_stl = False, T_list = None, mesh = None, clim = None):
        if T_list is None and mesh is None:
            T_list = self.robot.T_list
        stl = pv.PolyData()
        colour =  '#b2babb'
        if mesh is None:
            for i, link in enumerate(self.robot.links):
                T_link = self.robot.fkine(i)
                # stl += (link.stl.copy()).transform(T_link, inplace=False)
                stl = stl.merge((link.stl.copy()).transform(T_link, inplace=False), merge_points=False)
                
                if self.joint_frames:
                    add_frame(self.plot, T_link, scale=1, name_=link.name)
                if self.joint_axes:
                    add_joint_axis(self.plot, T_link, scale=1, name_=link.name)
        else:
            stl = mesh
        if return_stl:
            return stl
        
        try:
            self.plot.add_mesh(stl, name='robot', color = colour, scalars = stl['Scalars'], clim=clim, cmap = "jet",reset_camera=False)
            self.plot.remove_scalar_bar()
        except:
            self.plot.add_mesh(stl, name='robot', color = colour, reset_camera=False, opacity = self.robot_opacity)
        
        if self.EE_frame:
            add_frame(self.plot, self.robot.fkine(-1), scale=1, name_='EE')
            
        if self.CoM:
            for i, link in enumerate(self.robot.links):
                T_link = self.robot.fkine(i)
                CoM_pos = link.CM
                if link.mass == 0:
                    continue
                else:
                    self.plot.add_mesh(pv.Sphere(radius=20, center=CoM_pos).transform(T_link, inplace=False), color='red', name=link.name+'_CoM', reset_camera=False)
        
        if self.J_det:
            J_det = float(self.robot.jacobian_sym(evalf=True).det().evalf())
            color = plt.cm.RdYlGn(np.abs(J_det/300000000))
            self.plot.add_text(f"|J|: {J_det:.3f}", position='upper_left', font_size=30, color=color, name='J_det')
        

            
    def animate(self, 
                trajectory,
                plot_trajectory = False,
                save_video = False, 
                video_file = 'robot_animation.mp4',
                time_slider = False,
                print_time = False,
                FPS = None,
                add_note = None):
        """Animate the robot along the trajectory. The trajectory can be joint or cartesian
        """
        self.print_time = print_time
        if time_slider and save_video:
            raise Exception('Cannot save video and use time slider at the same time')
        
        if FPS is None:
            FPS = int(1/trajectory.dt)
        
        if self.robot.media_dir is not None:
            video_file = os.path.join(self.robot.media_dir, video_file)
        
        def slider_callback(value):
            ind = np.argmin(np.abs(trajectory.t - value))
            self.robot.q = trajectory.q_traj[ind]
            self.add_meshes()
        
        if time_slider:
            self.plot.add_slider_widget(callback=slider_callback, rng=[0, trajectory.t[-1]], value=0, title='time', interaction_event='always', style='modern', pointa=(0.1, 0.1), pointb=(0.9, 0.1))
        
        
        if self.joint_frames:
            print('To improve the performance, please set joint_frames to False')
        
        if save_video:
            self.plot.open_movie(video_file, framerate=FPS)
            print(f'Saving video at {FPS} FPS')
            
        if plot_trajectory:
            tube = trajectory.trajectory_polyline()
            self.traj_actor = self.plot.add_mesh(tube, color='red', name='trajectory')
            self.traj_actor.SetVisibility(True)
        else:
            if self.traj_actor is not None:
                self.traj_actor.SetVisibility(False)
                
        if trajectory.meshes is None:
            trajectory.meshes = [None]*len(trajectory.t)
        frameperiod = trajectory.dt
        now = time()
        nextframe = now + frameperiod
        if not time_slider:    
            for i in tqdm(range(len(trajectory.t))):
                # self.robot.q = trajectory.q_traj[i]
                self.robot.q = trajectory.q_traj[i]
                self.add_meshes(T_list=trajectory.T_list[i], mesh=trajectory.meshes[i], clim=trajectory.clim)
                if print_time:
                    self.plot.add_text(f"t: {trajectory.t[i]:.2f}", position='upper_left', font_size=12, color='black', name='time')
                if add_note is not None:
                    self.plot.add_text(add_note, position='upper_right', font_size=12, color='black', name='note')
                if save_video:
                    self.plot.write_frame()
                while now < nextframe:
                    sleep(int(nextframe - now))
                    now = time()
                nextframe += frameperiod
        
        
        if save_video:
            self.plot.mwriter.close()

        if not save_video and not time_slider and self.animation_traj is None:
            self.add_action(self.animate_toolbar, "Animate", self.animate_traj)
        self.animation_traj = trajectory 
    
    def animate_traj(self):
        trajectory = self.animation_traj
        frameperiod = trajectory.dt
        now = time()
        nextframe = now + frameperiod
        for i in range(len(trajectory.t)):
            self.robot.q = trajectory.q_traj[i]
            self.add_meshes(T_list=trajectory.T_list[i], mesh=trajectory.meshes[i], clim=trajectory.clim)
            if self.print_time:
                self.plot.add_text(f"t: {trajectory.t[i]:.2f}", position='upper_left', font_size=12, color='black', name='time')
            while now < nextframe:
                sleep(int(nextframe - now))
                now = time()
            nextframe += frameperiod
        
    def add_action(self,toolbar, key, function):
        """
        Connects a toolbar button with a certain function

        :param toolbar: Toolbar parameter
        :param key: Name of the toolbar
        :param function: Function to connect the action to
        """

        action = QAction(key, self.plot.app_window)
        action.triggered.connect(function)
        toolbar.addAction(action)
            
            
                
def add_frame(p, T, scale=1.0, name_='frame', axes = 'xyz'):
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
    
def add_joint_axis(p, T, scale = 1.0, name_ = 'axis', color = 'blue', radius = 10):
    scale *= 400
    origin = T[:3, 3]
    z_axis = T[:3, 2]
    z_axis = pv.Cylinder(center=origin, direction=z_axis, height=scale, radius=radius)
    p.add_mesh(z_axis, color=color, name=name_+'joint_axis')
    