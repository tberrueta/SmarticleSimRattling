import pickle
import numpy as np
import scipy.linalg as sci
from scipy import signal
from sklearn.neighbors import NearestNeighbors

# Rotations
def wrap2Pi(x):
	xm = np.mod(x+np.pi,(2.0*np.pi))
	return xm-np.pi

def Rot(x):
	return np.array([[np.cos(x),-np.sin(x)],[np.sin(x),np.cos(x)]])

def RotVec(x_vec, rot_vec):
	rvec = np.array([np.dot(x_vec[i,:-1],Rot(rot_vec[i])) for i in range(x_vec.shape[0])])
	return np.hstack([rvec, x_vec[:,2].reshape(x_vec.shape[0],1)])

# Rattling
def diffusion_vel(x, dt):
    t_vec = np.expand_dims(np.sqrt(np.arange(1,x.shape[0]+1)*dt),axis=1)
    vec = np.divide(x-x[0],t_vec)
    return vec

def rattling(x, dt, noRat = False, diffVel=True):
    if diffVel:
        vec = diffusion_vel(x, dt)
    else:
        vec = np.copy(x)
    C = np.cov(vec.T)
    if noRat:
        R = None
    else:
        if len(np.shape(C)) == 0:
            R = 0.5*np.log(C)
        else:
            R = 0.5*np.log(np.linalg.det(C))
    return R, C

def rattling_windows(mat, dt, window_sz, overlap,noRat=False,diffVel=True):
    cov_list = []
    rat_list = []
    ind_list = window_inds(mat,window_sz,overlap)
    for inds in ind_list:
        R, C = rattling(mat[inds[0]:inds[1],:],dt,noRat, diffVel)
        cov_list.append(C)
        rat_list.append(R)
    return rat_list, cov_list, ind_list

# Rectangular windowing
def window_inds(dataset, window_sz, overlap, offset=0):
    """
    Helper function that applies a rectangular window to the dataset
    given some overlap percentage, s.t. ov \in [0,1)
    """
    data_len = dataset.shape[0]
    assert window_sz < data_len
    ind1 = offset
    ind2 = offset+window_sz-1
    ind_list = []
    ov_ind_diff = int(np.ceil(np.abs(overlap*window_sz)))
    if ov_ind_diff == window_sz:
        ov_ind_diff += -1
    while ind2 < data_len+offset:
        ind_list.append((ind1,ind2))
        ind1 += window_sz-ov_ind_diff
        ind2 += window_sz-ov_ind_diff
    return ind_list

# Filtering
def moving_average(x,N):
    return np.convolve(x,np.ones((N,))/float(N),mode='valid')

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandstop(cutoffs, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoffs = cutoffs / nyq
    b, a = signal.butter(order, normal_cutoffs, btype='bandstop', analog=False)
    return b, a

def butter_bandstop_filter(data, cutoffs, fs, order=5):
    b, a = butter_bandstop(cutoffs, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Save/Load
def store_data(fname, rlist, clist, elist):
    db = {}
    db['rlist'] = rlist
    db['clist'] = clist
    db['elist'] = elist
    dbfile = open(fname, 'ab')
    pickle.dump(db, dbfile)
    dbfile.close()

def load_data(fname):
    # for reading also binary mode is important
    dbfile = open(fname, 'rb')
    db = pickle.load(dbfile)
    rlist = db['rlist']
    clist = db['clist']
    elist = db['elist']
    dbfile.close()
    return rlist, clist, elist

# Observable preprocessing
def preprocess(data):
	"""
	Here we have take in a dataset of smarticle coordinates of the following
	shape: (SampleNum, SmarticleNum*3), where each of the 3 coordinates are
	coords = [x,y,theta,L_arm_theta,R_arm_theta]. We output a 7 dimensional
	vector of the following format: [<mx_1>,<mx_2>,<mx_3>,<my_1>,<my_2>,<my_3>,e_theta]
	"""

	# Take in (x,y,theta) of each smarticle
	S1_coords = data[:,0:3]
	S2_coords = data[:,3:6]
	S3_coords = data[:,6:9]

	#########################
	# Rotational invariance #
	#########################
	# Get CoM from the frame of each smarticle
	CoM = np.mean([S1_coords,S2_coords,S3_coords],axis=0)
	CoM_S1 = CoM-S1_coords
	CoM_S2 = CoM-S2_coords
	CoM_S3 = CoM-S3_coords

	# Wrap angles
	CoM_S1[:,2] = wrap2Pi(CoM_S1[:,2])
	CoM_S2[:,2] = wrap2Pi(CoM_S2[:,2])
	CoM_S3[:,2] = wrap2Pi(CoM_S3[:,2])

	# Rotate coordinates so they're relative to the previous timestep
	relCoM_S1 = RotVec(CoM_S1, S1_coords[:,2])
	relCoM_S2 = RotVec(CoM_S2, S2_coords[:,2])
	relCoM_S3 = RotVec(CoM_S3, S3_coords[:,2])

	# Result Matrix
	resMat = np.vstack([relCoM_S1[:,0],relCoM_S2[:,0],relCoM_S3[:,0],
						relCoM_S1[:,1],relCoM_S2[:,1],relCoM_S3[:,1]]).T

	# For theta:
	pTheta = np.abs(np.mean(np.exp(1j*np.vstack([S1_coords[:,2],S2_coords[:,2],S3_coords[:,2]]).T),axis=1)).reshape(data.shape[0],1)
	return np.hstack([resMat,pTheta])

# Estimating steady-state
def estimate_pss(qseed,qss=None,thresh=5,num_neighb=None,returnModel=False):
    # Make N >> d
    d = np.min(qseed.shape)
    L = np.max(qseed.shape)
    N = 3*d if num_neighb is None else num_neighb
    qss_vec = np.copy(qseed) if qss is None else qss

    # Instantiate NN model
    knn = NearestNeighbors(n_neighbors=N)
    knn.fit(qseed)
    neighborhoods = knn.kneighbors(qseed,N,return_distance=False) # get neighbors

    # Calculate neighborhood volume
    var_tensors = [np.cov(qseed[neighborhoods[i]].T) for i in range(L)] # get variance tensor over neighborhoods
    vol_list = np.array([np.sqrt(np.linalg.det(var_tensors[i])) for i in range(L)])

    # Calculate inner-product over variance tensor
    var_prods = []
    for i in range(L):
        temp = []
        for q_p in qss_vec:
            temp.append((q_p-qseed[i]).dot(np.linalg.inv(var_tensors[i])).dot((q_p-qseed[i]).T))
        var_prods.append(temp)

    # Estimate probability given by counting fraction of points within some radius
    prob_list = np.array([len(np.where(np.array(v)<thresh)[0])/float(len(v)) for v in var_prods])

    # Output probability density
    if returnModel:
        return prob_list/vol_list, knn
    else:
        return prob_list/vol_list
