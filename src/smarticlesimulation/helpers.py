import numpy as np
from .smarticle import Smarticle

def time_to_steps(time_s):
    return int(time_s*240)

def load_smarticles(num_smart,urdf_path,gait,gait_period,z,dx=0.032,th=np.pi/2, rand = False):
    smarticles = []
    offset = -dx*(num_smart//2)
    for ii in range(num_smart):
        x = offset+ii*dx
        if rand:
            smarticles.append(Smarticle(urdf_path, basePosition = 'random',\
                                        baseOrientation = 'random'))
        else:
            smarticles.append(Smarticle(urdf_path, basePosition = [0,x,z],\
                                        baseOrientation = [0,0,th]))

        if len(np.array(gait).shape) == 3:
            smarticles[-1].load_gait(np.array(gait[ii]),gait_period)
        else:
            smarticles[-1].load_gait(np.array(gait),gait_period)
    return smarticles

def get_state(smarticles,num=0):
    """
    Constructing state vector that we query from simulation 
    at each point in time. Structure as of now is the following:
    [x_1 ,y_1, th_1, ... (Ns) gait_index_1, ... (Ns), special index to mark events]
    """
    vec = []
    vec2 = []
    for s in smarticles:
        s.update_position()
        vec.append(s.x)
        vec2.append(s.gait_index)
    vec2.append(num)
    return np.hstack([np.array(vec).flatten(),np.array(vec2)])

def get_ring_state(ring):
    return np.array(p.getBasePositionAndOrientation(ring)[0][:2])
