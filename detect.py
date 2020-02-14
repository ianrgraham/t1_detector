"""
This guy is going to used to anaylze t1 events that pop up in events.

What we'll need to do is use Jason's phom framework (or something else)
to quickly produce the Delauney triangulations for these packings.

The simplest thing to do is to do this every quarter of a cycle,
and then look for changes in these sets of pairs

We may as well save birth death info on each quarter frame as well
"""

paths = ("/data1/shared/igraham/lib_persistent_homology/python_src:"
    "/data1/shared/igraham/new_lib_persistent_homology:"
    "/home1/igraham/Projects/quasilocalized_modes").split(':')

import gsd.hoomd
import gsd.fl
import numpy as np
from numba import njit
import pickle
import numpy.ma

import sys
import os
print(paths)
for p in paths:
    sys.path.insert(0,p)
    print(p)

pjoin = os.path.join

import phom
import triangulation as tri
import hessian_calc as hc

@njit
def _apply_inverse_box(pos, box_mat):
    new_pos = np.zeros_like(pos)
    inv_mat = np.linalg.inv(box_mat)
    for i in np.arange(len(new_pos)):
        new_pos[i] = np.dot(inv_mat, pos[i])
    return new_pos

def _get_configuration_hoomd(s, r_func=hc.get_hoomd_bidisperse_r, dim=2, remove_rattlers=False):

    assert(dim == 2)

    pos = s.particles.position[:,:dim].astype(np.float64)
    #pos = pos.view(np.ma.MaskedArray)
    NP = len(pos)
    rad2 = np.vectorize(r_func)(s.particles.typeid)**2

    #rad2 = rad2.view(np.ma.MaskedArray)
    box = s.configuration.box
    print(box)
    box_mat = np.array([[box[0],box[0]*box[3]],[0.0,box[1]]], dtype=np.float64)
    pos = _apply_inverse_box(pos, box_mat) # brings us to a -1/2 to 1/2 box in both dimensions
    #box_mat = box_mat.T
    print(box_mat)
    print(np.max(pos[:,0]), np.min(pos[:,0]))
    print(np.max(pos[:,1]), np.min(pos[:,1]))

    pos = pos.flatten() + 1 # since pos data is saved from -L/2 to L/2 originally, we want bring it to 0 to 1

    embed = phom.Embedding2D(NP, pos, np.asfortranarray(box_mat), True)
         
    comp = tri.construct_triangulation(embed, rad2)
    
    if remove_rattlers:
        (rattlers, comp, embed, rad2) = tri.remove_rattlers(comp, embed, rad2)
        return (comp, embed, rad2, rattlers)

    return (comp, embed, rad2)

def _get_births_deaths(comp, embed, rad2, dim=-1):
    """Returns the births and deaths (as two lists) of components in the alpha complex generated from the given jamming state file at the given record.
    
    Additionally, the dimension of the objects to be returned can be passed. 
    *** dim = -1 returns births/deaths of components of all dimension
    *** dim = n (>= 0) returns births/deaths of components of dimension n
    """
    
    if dim < -1:
        print("Improper dimension ({}) given".format(dim), flush=True)
        return
    
    r2norm = np.min(rad2)
    
    if embed.dim == 2:
        alpha_vals = phom.calc_alpha_vals_2D(comp, embed, rad2, alpha0=-r2norm)
    elif embed.dim == 3:
        alpha_vals = phom.calc_alpha_vals_3D(comp, embed, rad2, alpha0=-r2norm)
        
    filt = phom.construct_filtration(comp, alpha_vals, )

    pairs = phom.calc_persistence(filt, comp)

    bd = []

    print(len(pairs))
    
    for i, j in pairs:
        if dim == -1 or dim == comp.get_dim(i):
            bd.append(np.array([filt.get_func(i)/r2norm, filt.get_func(j)/r2norm]))

    return np.array(bd)

def _get_SSPq(gsd_file_root):
    """
    recording steps in each quarter cycle
    """
    prep = gsd_file_root.strip('/').split('/')[-1].split("_")
    recPer = 1
    maxStrain = None
    strainStep = None
    for p in prep:
        if "maxStrain" in p:
            maxStrain = float(p.replace("maxStrain",""))
        elif "strainStep" in p:
            strainStep = float(p.replace("strainStep",""))
        elif "recPer" in p:
            recPer = int(p.replace("recPer",""))
            
    SSPq = int(np.round(maxStrain/strainStep))//recPer
    return SSPq

def alpha_complex_producer(gsd_file_root, r_func=hc.get_hoomd_bidisperse_r, dim=2):
    # no need to make a class here, as procedure is pretty straight forward
    # we're only going to attempt to produce data on quarter cycles
    phom_dir = pjoin(gsd_file_root, "phom")
    print(phom_dir)
    os.makedirs(phom_dir, exist_ok=True)
    traj = pjoin(gsd_file_root, 'traj.gsd')
    with gsd.fl.open(name=traj, mode='rb') as f:
        nframes = f.nframes

    # infer SPP
    SSPq = _get_SSPq(gsd_file_root)

    with gsd.hoomd.open(name=traj, mode='rb') as f:
        for i in range(0,nframes, SSPq):
            s = f[i]
            #print(i)

            # construct cell complex
            comp, embed, rad2 = _get_configuration_hoomd(s)

            # save these three to disk
            out = pjoin(phom_dir, f"{i}_phom.pkl")
            # so it appears we can't save the complex object
            # I guess our best chance is to save all of the dealauney pairs at each quarter cycle frame
            #with open(out, 'wb') as f:
            #    pickle.dump({"comp":comp,"embed":embed,"rad2":rad2}, f)

            #if i % 2*SSPq == 0:
            # if at a strobascopic step, produce persistence data
            birthDeath = _get_births_deaths(comp, embed, rad2)
            out = pjoin(phom_dir, f"{i}_bd")
            #print(birthDeath.shape)
            #print(birthDeath)
            #sys.exit()
            np.savez_compressed(out, bd=birthDeath)
                
            #save birthDeath to disk


def t1_analyzer():
    # this guy will later go over all alpha complex frames and check 
    pass # TODO
    
