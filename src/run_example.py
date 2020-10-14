import pybullet as p
import time
import pybullet_data
import smarticlesimulation as sim
import numpy as np
import keyboard

# URDF paths
smarticle_path = '../urdf/smarticle.urdf'
ring_path = '../urdf/ring.urdf'
data_path = '../data/'

# Parameters
viz = False
rand_len = 2*60 # random gait time
sim_len = 2*60  # sim time
rand_steps = sim.time_to_steps(rand_len)
sim_steps = sim.time_to_steps(sim_len)
Ns = 3 # number of smarticles
max_vel = 20 # max arm velocity
width = 0.032 # placement spacing
th = np.pi/2 # inital orientation

# Set up environment
if viz:
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
ringID = p.loadURDF(ring_path, basePosition = [0,0,0])
ringConstraintID = p.createConstraint(ringID,-1,0,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])

# Set up smarticles
R = [-np.pi/2,-np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi/2,-np.pi/2,-np.pi/2]
L = [np.pi/2,np.pi/2,np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2,-np.pi/2]
gaits = [[L,R],[L,R],[L,R]]
gait_period = sim.time_to_steps(0.2)
smarticles = sim.load_smarticles(Ns, smarticle_path, gaits, gait_period, 0, width, th)

# Randomize
for i in range(rand_steps):
    p.stepSimulation()
    for s in smarticles:
        if i % gait_period == s.gait_phase:
            s.move_random_corners()
        if keyboard.is_pressed('q'):
            break
    if keyboard.is_pressed('q'):
        break

# Simulation loop
data = np.zeros((sim_steps,3*Ns))
for i in range(sim_steps):
    if viz:
        time.sleep(1/240.)
    data[i] = sim.get_state(smarticles)
    p.stepSimulation()
    for s in smarticles:
        if i % gait_period == s.gait_phase:
            s.motor_step()
        if keyboard.is_pressed('q'):
            break
    if keyboard.is_pressed('q'):
        break

# Save data
np.savetxt(data_path+"foo.csv", data, delimiter=",")
p.disconnect()
