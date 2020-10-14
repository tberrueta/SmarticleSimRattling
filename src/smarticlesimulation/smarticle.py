# smarticle_simulation_object
import numpy as np
import pybullet as p

class Smarticle(object):
    MAX_FORCE = 0.1
    MAX_VEL = 10
    MAX_VEL_RANGE = 0.2
    MAX_ANGLE_OFFSET = 0.2
    MAX_HIT_ANGLE = 0.9
    EPS = 5e-3
    PR_LOC = 1e-3*np.array([[16,11.5,10],[-16,16,10]])

    def __init__(self,urdf_path, basePosition=None, baseOrientation= None, max_r=5e-3, debug = 0):

        self.x = np.zeros(3)
        self.pr_loc_global = np.zeros([2,3])
        self.hit = 0
        self.plank = 0
        self.debug = debug

        # load basePosition and baseOrientation
        if basePosition is None:
            basePosition = [0,0,0]
        elif basePosition == 'random':
            r = max_r*np.random.rand()
            th = 2*np.pi*np.random.rand()
            c,s = np.cos(th), np.sin(th)
            basePosition = [r*c,r*s,0]
        if baseOrientation is None:
            baseOrientation = [0,0,0]
        elif baseOrientation == 'random':
            th = 2*np.pi*np.random.rand()
            baseOrientation = [0,0,th]
        baseOrientation = p.getQuaternionFromEuler(baseOrientation)

        # loadURDF
        self.initial_pos = basePosition
        self.initial_orn = baseOrientation
        self.id = p.loadURDF(urdf_path, basePosition = basePosition,\
                                        baseOrientation = baseOrientation)

      # add randomness to smarticle params
        self.maxvel = self.MAX_VEL + self.MAX_VEL_RANGE*(2.*np.random.rand(2)-1.)
        self.control = p.POSITION_CONTROL
        self.arm_offset = self.MAX_ANGLE_OFFSET*(2.*np.random.rand(2)-1.)

        # set friction of arms to zero
        for joint in range(2):
            p.changeDynamics(self.id,joint,lateralFriction = 0.1)
        self.update_position()

    def reset_pose(self, pos=None, orn=None, joint_state=None):
        '''
        reset smarticles to initial positions
        '''
        if pos == None:
            pos = self.initial_pos
        if orn == None:
            orn = self.initial_orn
        if joint_state == None:
            joint_state = [0,0]
        p.resetBasePositionAndOrientation(self.id,pos,orn)
        p.resetJointState(self.id,0,joint_state[0])
        p.resetJointState(self.id,1,joint_state[1])
        self.update_position()

    def load_gait(self, gait, gait_dt, rand_init = False, rand_phase=False):
        '''
        load gait of [gaitL, gaitR], where gaitL and gaitR are lists for smarticles to execute
        '''
        self.n = gait.shape[-1]
        self.gait_period = gait_dt
        self.gaitL = gait[0]
        self.gaitR = gait[1]

        # phase drift
        if rand_phase:
            self.gait_phase = int(self.gait_period*np.random.rand())
        else:
            self.gait_phase = 0

        # phase init
        if rand_init:
            self.gait_index = np.random.randint(0,self.n)
        else:
            self.gait_index = 0

    def move_arms(self,posL, posR):
        '''
        move arms to specified angles
        '''
        p.setJointMotorControl2(self.id,0, self.control,\
            targetPosition=posL+self.arm_offset[0],\
            maxVelocity = self.maxvel[0],\
            force = self.MAX_FORCE)
        p.setJointMotorControl2(self.id,1, self.control,\
            targetPosition=posR+self.arm_offset[1],\
            maxVelocity = self.maxvel[0],\
            force = self.MAX_FORCE)

    def move_random_corners(self):
        '''
        move arms to random corners
        '''
        posL = np.random.choice([-np.pi/2, np.pi/2])
        posR = np.random.choice([-np.pi/2, np.pi/2])
        self.move_arms(posL, posR)

    def set_plank(self, state):
        '''
        enable/disable plank state of smarticles
        '''
        if state==1:
            self.plank = 1
            self.move_arms(0,0)
            if self.debug:
                p.changeVisualShape(self.id,-1,rgbaColor=[0,1,0,1])
        else:
            self.plank = 0
            if self.debug:
                p.changeVisualShape(self.id,-1,rgbaColor=[0.3,0.3,0.3,1])

    def motor_step(self):
        '''
        execute next step in gait
        '''
        if self.plank == 0:
            posL = self.gaitL[self.gait_index]
            posR = self.gaitR[self.gait_index]
            self.move_arms(posL, posR)
        else:
            posL = self.arm_offset[0]
            posR = self.arm_offset[1]
        self.gait_index = (self.gait_index+1)%self.n

    def update_position(self):
        '''
        update position of smarticles
        '''
        pos, orient = p.getBasePositionAndOrientation(self.id)
        self.x[0:2]=np.array(pos[0:2])
        self.x[2]=np.mod(p.getEulerFromQuaternion(orient)[2],2*np.pi)
        for ii in range(len(self.PR_LOC)):
            self.pr_loc_global[ii],_ = p.multiplyTransforms(pos, orient,\
                                                     self.PR_LOC[ii], [0,0,0,1])

    def light_plank(self,ray, light_yaw):
        '''
        check if light rays hit photoresistor positions on smarticles
        '''
        ray = np.array(ray)
        if self.plank == 1:
            return False
        else:
            err1 = np.linalg.norm(self.pr_loc_global[0][:2]-ray[:2])
            err2 = np.linalg.norm(self.pr_loc_global[1][:2]-ray[:2])
            angle_diff = np.abs(np.mod(light_yaw-self.x[2]+np.pi,2*np.pi)-np.pi)
            angle_diff = np.pi-angle_diff if (angle_diff>np.pi/2) else angle_diff

            if (err1 < self.EPS or err2 < self.EPS\
                and angle_diff< self.MAX_HIT_ANGLE):

                self.set_plank(1)
                return True
            else:
                return False
