import math
import numpy as np
import EMAN2

class Symmetry(object):
    def __init__(self,symtype):
        super(Symmetry, self).__init__()
        self._symtype = symtype

    @staticmethod
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
    
    def orients_one_unit(self,delta_deg):
        sym = EMAN2.parsesym(self._symtype)
        drad = np.pi * delta_deg / 180.0
        # generate uniform orientations on the sphere
        vecs = np.float32(Symmetry.calc_points(drad))
        nvec = vecs.shape[0]

        # convert unit vectors to azimuth and altitude
        az  = 180.0 * np.arctan2(vecs[:, 0], vecs[:, 2]) / np.pi + 180.0
        alt = 180.0 * np.arcsin(vecs[:, 1]) / np.pi + 90.0

        # zero out azimuth on poles
        aloc = np.logical_or(alt == 0.0, alt == 180.0)
        az[aloc] = 0.0

        # select orientations inside the asymmetric unit, including mirrors (True)
        isin  = np.array([sym.is_in_asym_unit(float(az[v]), float(alt[v]), True) for v in range(nvec)])
        azin  = az[isin]
        altin = alt[isin]
        ors   = [EMAN2.Transform({"type": "eman", "az": float(a), "alt": float(al), "phi": 0.0}) for a, al in zip(azin, altin)]
        return ors

    def orients2mats_all_units(self,orients):
        ''' returns rotation matrices corresponding to orientations in each symmetric unit
            mats: nsym,nrots,3,3 '''
        nrots = len(orients)
        nsym  = EMAN2.parsesym(self._symtype).get_nsym()
        mats  = np.zeros([nsym,nrots,3,3], dtype='float32')
        for n in range(nsym):
            for k in range(nrots):
                m = orients[k].get_sym(self._symtype, n).get_matrix()
                mats[n,k] = np.reshape(m, [3, 4])[..., :3]
        return mats

    def orient_matrices(self,delta_deg):
        ''' Returns orientation matrices that cover rotations within assymetric unit '''
        ors = self.orients_one_unit(delta_deg)
        return self.orients2mats_all_units(ors)



######## GARBAGE #############
   # def orients2allunits(self,orients):
    #     ''' Generates arrays of orientation vectors corresponding to each symmetry unit '''
    #     sym     = EMAN2.parsesym(self._symtype)
    #     nsym    = sym.get_nsym()
    #     nvecs   = len(orients)
    #     vecs    = np.zeros((nsym,nvecs,3))
    #     sors    = []
    #     for s in range(nsym):
    #         # transform orientations to this symmetry unit
    #         sors.append([orient.get_sym(self._symtype,s) for orient in orients])
    #     return vecs