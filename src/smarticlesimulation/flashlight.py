import numpy as np
import pybullet as p

from pdb import set_trace as bp

class Flashlight(object):
    """docstring for Flashlight."""

    def __init__(self,urdf_path, basePosition, yaw=0, beam_width=np.pi/14,\
                ray_count=90, ray_length=1.2, debug=0):
        self.debug = debug
        self.x = np.array(basePosition).astype(np.double)
        self.polar = np.array(self.x2polar(*self.x[:2]))
        self.yaw = np.mod(yaw,2*np.pi)
        self.beam_width = beam_width
        self.ray_len = ray_length
        self.ray_count = ray_count
        self.ray_from = self.x*np.ones([self.ray_count,1])
        self.d_theta = beam_width/(ray_count-1)
        self.rays = -self.beam_width/2.\
                    +np.arange(0,self.beam_width+self.d_theta/2.,self.d_theta)
        self.ray_hit_color = [1,0.49,0]
        self.ray_miss_color = [1,0.75,0]

        self.id= p.loadURDF(urdf_path,basePosition=basePosition,\
                                baseOrientation=p.getQuaternionFromEuler([0,0,self.yaw]))
    @staticmethod
    def polar2x(r, theta):
        z = r * np.exp(1j*theta)
        return np.array([np.real(z),np.imag(z)])

    @staticmethod
    def x2polar(x,y):
        r = np.sqrt(x**2+y**2)
        theta = np.mod(np.arctan2(y,x),2*np.pi)
        return r, theta


    def set_position(self,x, yaw=None):
        '''
        set position of flashlight using x - [x y z] and yaw or angle in the 2d plane
        '''
        self.x = np.array(x)
        self.ray_from = self.x*np.ones([self.ray_count,1])
        if yaw is not None:
            self.yaw = np.mod(yaw,2*np.pi)
        p.resetBasePositionAndOrientation(self.id,x,p.getQuaternionFromEuler([0,0,self.yaw]))



    def set_polar_position(self,origin, r=None,th=None, z = None):
        '''
        set position of flashlight using polar coordinates
        '''
        if r is not None:
            self.polar[0] = r
        if th is not None:
            self.polar[1] = th
        if z is not None:
            self.x[2] = z
        xy = self.polar2x(self.polar[0],self.polar[1])+origin[:2]
        yaw = np.arctan2(xy[1]-origin[1],xy[0]-origin[0])
        self.set_position([xy[0],xy[1],self.x[2]],yaw=yaw+np.pi)


    def draw_rays(self):
        '''
        draws rays to be visualized in sim
        '''
        ray_to = np.zeros([self.ray_count,3])
        ray_ids = np.zeros(self.ray_count)
        ray_angles = np.mod(self.yaw+self.rays,2*np.pi)
        for ii,th in enumerate(ray_angles):
            ray_to[ii] = self.x+ np.array([self.ray_len*np.cos(th),\
                                self.ray_len*np.sin(th),0])

            if self.debug:
                p.addUserDebugLine(self.x, ray_to[ii], self.ray_miss_color)
        return p.rayTestBatch(self.ray_from, ray_to)

    def ray_check(self, smarticles):
        '''
        updates position of smarticles and checks if any are hit by the light rays
        '''
        p.removeAllUserDebugItems()
        results = self.draw_rays()
        smart_ids = [x.id for x in smarticles]
        for smart in smarticles:
            smart.update_position()
            smart.set_plank(0)
        for ray in results:
            if ray[0]>=smart_ids[0] and ray[1]==-1:
                index = smart_ids.index(ray[0])
                if smarticles[index].light_plank(ray[3],self.yaw):
                        p.addUserDebugLine(self.x, ray[3], self.ray_hit_color)
