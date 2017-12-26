import math
import numpy as np
import EMAN2

def calc_points(vert_ang_inc, max_lat=math.pi, max_long=2 * math.pi, S=True):
    ''' vert_ang_inc - define how fine the sphere is sampled, max_lat - maximum latitutde angle
        max_long - max longtitude angle '''
    points = []
    ball_dia     = math.sqrt(2 - 2 * math.cos(vert_ang_inc))
    num_circles  = round(max_lat / vert_ang_inc)
    horz_stagger = 0
    for circle in range(int(num_circles)):
        vert_ang = circle * vert_ang_inc
        rad = math.sin(vert_ang)
        try:
            num_balls = math.floor(max_long / math.acos((2 * rad * rad - ball_dia * ball_dia) / (2 * rad * rad)))
        except:
            num_balls = 1
        # print ">>"+str(horz_stagger)
        horz_ang_inc = max_long / num_balls
        for n in range(int(num_balls)):
            horz_ang = n * horz_ang_inc + horz_stagger
            points.append((rad * math.sin(horz_ang), math.cos(vert_ang), rad * math.cos(horz_ang)))

        if S:
            horz_stagger = horz_ang_inc * (circle % 2) / 2
    return points

def emrot2mat(rots,symtype='c1',n=0):
    nrots = len(rots)
    mats  = np.zeros([nrots,9],dtype='float32')
    for k in range(nrots):
        m = rots[k].get_sym(symtype,n).get_matrix()
        mats[k,:] = (np.reshape(m,[3,4])[...,:3]).flatten()
    return mats

def gen_orient_balanced(delta, symtype):
    sym = EMAN2.parsesym(symtype)
    drad = np.pi * delta / 180.0
    # generate uniform orientations on the sphere
    vecs = np.float32(calc_points(drad))
    nvec = vecs.shape[0]

    # # apply random rotation to all vectors
    # R = utils.rand_rot_mat((3, 3))
    # vecs = np.dot(vecs, R)

    # convert unit vectors to azimuth and altitude
    az  = 180.0 * np.arctan2(vecs[:, 0], vecs[:, 2]) / np.pi + 180.0
    alt = 180.0 * np.arcsin(vecs[:, 1]) / np.pi + 90.0

    # zero out azimuth on poles
    aloc = np.logical_or(alt == 0.0, alt == 180.0)
    az[aloc] = 0.0

    # select orientations insde the asymmetric unit
    isin = np.array([sym.is_in_asym_unit(float(az[v]), float(alt[v]), True) \
                     for v in range(nvec)])
    azin = az[isin]
    altin = alt[isin]
    ors = [EMAN2.Transform({"type": "eman", "az": float(a), "alt": float(al), "phi": 0.0}) for a, al in zip(azin, altin)]

    return ors