import os
from   fileio import filetools as ft
import numpy as np
from   star import star
from   ctf  import CTF

JOB2UPDIRS={'Select':2,
            'CtfFind':2}

def path2psize(path,jobname):
    ''' Read particle diameter from manual picking job.
        Assumes that path points to a star file in ctf job '''
    path    = os.path.realpath(path)
    jobfile = os.path.join(ft.updirs(path, JOB2UPDIRS[jobname]), '.gui_ctffindrun.job')
    line    = ft.get_line(jobfile, 'Magnified pixel size')
    return np.float32(line.split('==')[1])

def parse_particles_star(star_file):
    # here we also convert oringinal micrograph location to a phase flipped micrograph location
    # get path of phase flipped micrographs
    ajob   = os.path.dirname(os.path.realpath(star_file))
    recs   = star.starFromPath(star_file).readLines()
    micros = {}
    # obtain path to original micrograph
    root   = ft.updirs(ajob+'/',JOB2UPDIRS['Select']) #os.path.abspath(os.path.join(ajob,'../..'))
    for rec in recs:
        key = os.path.join(root,rec['MicrographName'])
        coord = [float(rec['CoordinateX']),float(rec['CoordinateY'])]
        if not key in micros:
            micros.update({key:{'coords':[coord],'ctf':CTF(**rec)}})
        else:
            micros[key]['coords'].append(coord)
    return micros

def parse_ctf_star(ctfstar):
    # ajob = os.path.dirname(os.path.realpath(ctfstar))
    recs = star.starFromPath(ctfstar).readLines()
    micros = {}
    root   = ft.updirs(ctfstar,JOB2UPDIRS['CtfFind'])
    for rec in recs:
        key = os.path.join(root,rec['MicrographName'])
        micros.update({key:{'ctf':CTF(**rec)}})
    return micros

def path2part_diameter(path):
    ''' Read particle diameter from manual picking job '''
    jobfile = os.path.join(ft.updirs(path, 2), '.gui_manualpickrun.job')
    line    = ft.get_line(jobfile, 'Particle diameter')
    return np.float32(line.split('==')[1])
    # return part_d_from_jobfile(jobfile)