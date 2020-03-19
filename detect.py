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
import scipy.stats
from collections import deque
import netCDF4

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
    box_mat = np.array([[box[0],box[0]*box[3]],[0.0,box[1]]], dtype=np.float64)
    pos = _apply_inverse_box(pos, box_mat) # brings us to a -1/2 to 1/2 box in both dimensions
    #box_mat = box_mat.T


    pos = pos.flatten() + 1 # since pos data is saved from -L/2 to L/2 originally, we want bring it to 0 to 1

    embed = phom.Embedding2D(NP, pos, np.asfortranarray(box_mat), True)
         
    comp = tri.construct_triangulation(embed, rad2)
    
    if remove_rattlers:
        (rattlers, comp, embed, rad2) = tri.remove_rattlers(comp, embed, rad2)
        return (comp, embed, rad2, rattlers)

    return (comp, embed, rad2)

def _get_voro_hoomd(s, r_func=hc.get_hoomd_bidisperse_r, dim=2):

    assert(dim == 2)

    pos = s.particles.position[:,:dim].astype(np.float64)
    #pos = pos.view(np.ma.MaskedArray)
    NP = len(pos)
    rad2 = np.vectorize(r_func)(s.particles.typeid)**2

    #rad2 = rad2.view(np.ma.MaskedArray)
    box = s.configuration.box
    box_mat = np.array([[box[0],box[0]*box[3]],[0.0,box[1]]], dtype=np.float64)
    pos = _apply_inverse_box(pos, box_mat) # brings us to a -1/2 to 1/2 box in both dimensions
    #box_mat = box_mat.T

    pos = pos.flatten() + 1 # since pos data is saved from -L/2 to L/2 originally, we want bring it to 0 to 1
    
    voro = phom.Voronoi2D(NP, pos, rad2, np.asfortranarray(box_mat), True)
    return voro

def _get_voro_nc(state, index, dim=2):
    assert(dim == 2)

    NP = len(state.dimensions['NP'])

    pos = state.variables['pos'][index]

    rad2 = state.variables['rad'][index]**2

    box_mat = np.array(state.variables['BoxMatrix'][index].reshape((dim, dim))).T
    
    voro = phom.Voronoi2D(NP, pos, rad2, box_mat, True)
    return voro

def _get_centroid_diffs(voro, s):
    box = s.configuration.box
    cent = voro.get_cell_centroids()
    cent = cent.reshape(len(cent)//2, 2)
    embed = voro.get_embedding()
    #out = np.zeros(len(cent))
    out2 = np.zeros_like(cent)
    for idx, c in enumerate(cent):
        tmp = hc.PBC_LE(c - embed.get_pos(idx), box)
        #out[idx] = np.linalg.norm(tmp)
        out2[idx] = tmp
    return out2

def _get_centroid_diffs_box(voro, box):
    #box = s.configuration.box
    cent = voro.get_cell_centroids()
    cent = cent.reshape(len(cent)//2, 2)
    embed = voro.get_embedding()
    #out = np.zeros(len(cent))
    out2 = np.zeros_like(cent)
    for idx, c in enumerate(cent):
        tmp = hc.PBC_LE(c - embed.get_pos(idx), box)
        #out[idx] = np.linalg.norm(tmp)
        out2[idx] = tmp
    return out2

"""@njit 
def LE_adjust(diff, box):
    sub_dim = (diff > box[0]/2).astype(np.int64)
    add_dim = (diff < -box[0]/2).astype(np.int64)
    LE = (add_dim - sub_dim)[::-1]
    LE[1:] = 0
    diff += (add_dim - sub_dim + LE*box[3])*box[0]
    return diff"""

@njit
def _delauney_calc(cdiffs, verts, pos, box):
    # calculate area and divergence of vectors for each delauney triangle
    areas = np.zeros((len(verts)))
    divs = np.zeros((len(verts)))
    for i in range(len(verts)):
        v = verts[i]
        tpos = np.empty((3,2))
        tpos[0] = pos[v[0]]
        for j in range(2):
            diff = pos[v[j+1]] - pos[v[j]]
            if (np.abs(diff) > box[0]/2).any():
                # move particle j+1 closer to j
                tpos[j+1] = tpos[j] + hc.PBC_LE(diff, box)
            else:
                tpos[j+1] = pos[v[j+1]]
        # get area
        areas[i] = 0.5*np.abs(tpos[0,0]*(tpos[1,1]-tpos[2,1]) + tpos[1,0]*(tpos[2,1]-tpos[0,1]) + tpos[2,0]*(tpos[0,1]-tpos[1,1]))
        # get div
        mat_b = np.empty((2,3))
        mat_m = np.ones((3,3), dtype=np.float64)
        for j in range(3):
            mat_m[:2,j] = tpos[j]
            mat_b[:,j] = cdiffs[v[j]]
        mat_a = np.dot(mat_b,np.linalg.inv(mat_m))
        divs[i] = mat_a[0,0] + mat_a[1,1] # trace (ignoring last column)
    
    return areas, divs

def _get_Qks(voro, s):
    diffs = _get_centroid_diffs(voro, s)
    #embed = voro.embed
    comp = voro.comp
    
    # need to look at older notebooks for how I can loop over delauney triangles

    # loop over triangles, get areas, and calculate divergence of interpolated voro-centers

    # for divergence, just integrate on boundary

    #box_mat = embed.box_mat # need this?
    #L = np.diagonal(box_mat)
        
    range_info = comp.dcell_range[2]
    vertices = np.zeros((range_info[1]-range_info[0],3), dtype=np.int64)
    # voronoi tesselation will aways have the same number of edges
    
    for idx, c in enumerate(range(*range_info)):
        
        vertices[idx] = np.array(list(voro.comp.get_faces(c, 0)), dtype=np.int64)

    areas, divs = _delauney_calc(diffs, vertices, s.particles.position[:,:2], s.configuration.box)
    Qk = divs*areas/np.mean(areas)
    return Qk, vertices

def _get_Qks_nc(voro, s, index):
    pos = s.variables['pos'][index,:]
    pos = pos.reshape(len(pos)//2, 2)
    tmpbox = s.variables['BoxMatrix'][index,:]
    box = np.array([tmpbox[0], tmpbox[0], 1, tmpbox[1]/tmpbox[0]])
    diffs = _get_centroid_diffs_box(voro, box)
    #embed = voro.embed
    comp = voro.comp
    
    # need to look at older notebooks for how I can loop over delauney triangles

    # loop over triangles, get areas, and calculate divergence of interpolated voro-centers

    # for divergence, just integrate on boundary

    #box_mat = embed.box_mat # need this?
    #L = np.diagonal(box_mat)
        
    range_info = comp.dcell_range[2]
    vertices = np.zeros((range_info[1]-range_info[0],3), dtype=np.int64)
    # voronoi tesselation will aways have the same number of edges
    
    for idx, c in enumerate(range(*range_info)):
        
        vertices[idx] = np.array(list(voro.comp.get_faces(c, 0)), dtype=np.int64)

    areas, divs = _delauney_calc(diffs, vertices, pos, box)
    Qk = divs*areas/np.mean(areas)
    return Qk, vertices

def _get_delauney_edges(comp):
    range_info = comp.dcell_range[1]
    #edges = np.zeros((range_info[1]-range_info[0],2), dtype=np.int64)
    edges = []
    for idx, c in enumerate(range(*range_info)):
        #edges[idx] = np.array(list(comp.get_faces(c, 0)), dtype=np.int64)
        edges.append(frozenset(comp.get_faces(c, 0)))
    return set(edges)

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
        
    filt = phom.construct_filtration(comp, alpha_vals)

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
            comp, embed, rad2 = _get_configuration_hoomd(s, r_func=r_func)

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

            np.savez_compressed(out, bd=birthDeath)
                
            #save birthDeath to disk

def voro_producer(gsd_file_root, r_func=hc.get_hoomd_bidisperse_r, dim=2):
    # no need to make a class here, as procedure is pretty straight forward
    # we're only going to attempt to produce data on quarter cycles
    phom_dir = pjoin(gsd_file_root, "voro")
    print(phom_dir)
    os.makedirs(phom_dir, exist_ok=True)
    traj = pjoin(gsd_file_root, 'traj.gsd')
    with gsd.fl.open(name=traj, mode='rb') as f:
        nframes = f.nframes

    # infer SPP
    SSPq = _get_SSPq(gsd_file_root)

    with gsd.hoomd.open(name=traj, mode='rb') as f:
        for i in range(0,nframes//SSPq*SSPq, 10):
            s = f[i]
            print(i)
            
            out = pjoin(phom_dir, f"{i}_qks")

            if os.path.exists(out+'.npz'):
                continue

            # construct voronoi tesselation
            voro = _get_voro_hoomd(s, r_func=r_func)

            # obtain Q_k's
            qks, verts = _get_Qks(voro, s)

            print(np.mean(qks), np.std(qks), scipy.stats.skew(qks))

            np.savez_compressed(out, qks=qks, verts=verts)


def t1_producer(gsd_file_root, r_func=hc.get_hoomd_bidisperse_r, dim=2):
    # fetch complex
    t1_dir = pjoin(gsd_file_root, "t1")
    print(t1_dir)
    os.makedirs(t1_dir, exist_ok=True)
    traj = pjoin(gsd_file_root, 'traj.gsd')
    with gsd.fl.open(name=traj, mode='rb') as f:
        nframes = f.nframes

    # infer SPP
    SSPq = _get_SSPq(gsd_file_root)

    edge_deque = deque([], maxlen=5)
    
    with gsd.hoomd.open(name=traj, mode='rb') as f:
        for i in range(0,nframes, SSPq):
            s = f[i]
            comp, embed, rad2 = _get_configuration_hoomd(s, r_func=r_func)
            # iterate over "bonds"
            # save list to memory
            edges = _get_delauney_edges(comp)
            edge_deque.append(edges)

            idx = i - 4*SSPq
            out = pjoin(t1_dir, f"{idx}_t1s")

            if os.path.exists(out+'.npz'):
                continue

            if len(edge_deque) == 5:
                t1_events = []
                reversed = []
                irreversed = []
                first_edges = edge_deque[0]
                final_edges = edges
                for j in range(1,4):
                    comp_edges = edge_deque[j]
                    for edge in first_edges:
                        if edge not in comp_edges:
                            if edge not in t1_events:
                                t1_events.append(edge)
                for t1 in t1_events:
                    if t1 in final_edges:
                        reversed.append(t1)
                    else:
                        irreversed.append(t1)

                nparr_edges = numpy.array([list(c) for c in final_edges]) # converts set of frozen sets to nparray
                reversed = np.array([list(c) for c in reversed])
                irreversed = np.array([list(c) for c in irreversed])

                

                np.savez_compressed(out, rev=reversed, irrev=irreversed, edges=nparr_edges)

def _nc_good(data_root):
    d = netCDF4.Dataset(data_root, 'r')
    p2c = d.variables['periods_to_cycle'][:][0]
    cl = d.variables['cycle_length'][:][0]
    
    if p2c < 0 or cl != 1:
        d.close()
        return False
    else:
        d.close()
        return True
    

def t1_producer_nc(nc_state, dim=2):
    # fetch complex
    t1_dir = pjoin(nc_state.replace("_state.nc","/other"), "t1")
    if not _nc_good(nc_state.replace("_state.nc","_data.nc")):
        return
    print(t1_dir)
    os.makedirs(t1_dir, exist_ok=True)
    # init netcdf state
    # infer SPP
    #SSPq = _get_SSPq(gsd_file_root) SSPq will always be 25 for these packings
    SSPq = 25

    edge_deque = deque([], maxlen=5)
    
    s = netCDF4.Dataset(nc_state, 'r')
    nframes = len(s.dimensions['rec'])
    for i in range(0,nframes, SSPq):
        
        embed, rad2 = tri.get_configuration(s, i)
        comp =  tri.construct_triangulation(embed, rad2)
        # iterate over "bonds"
        # save list to memory
        edges = _get_delauney_edges(comp)
        edge_deque.append(edges)

        idx = i - 4*SSPq
        out = pjoin(t1_dir, f"{idx}_t1s")

        if os.path.exists(out+'.npz'):
            continue

        if len(edge_deque) == 5:
            t1_events = []
            reversed = []
            irreversed = []
            first_edges = edge_deque[0]
            final_edges = edges
            for j in range(1,4):
                comp_edges = edge_deque[j]
                for edge in first_edges:
                    if edge not in comp_edges:
                        if edge not in t1_events:
                            t1_events.append(edge)
            for t1 in t1_events:
                if t1 in final_edges:
                    reversed.append(t1)
                else:
                    irreversed.append(t1)

            nparr_edges = numpy.array([list(c) for c in final_edges]) # converts set of frozen sets to nparray
            reversed = np.array([list(c) for c in reversed])
            irreversed = np.array([list(c) for c in irreversed])

            

            np.savez_compressed(out, rev=reversed, irrev=irreversed, edges=nparr_edges)
    s.close()
            

def voro_producer_nc(nc_state, dim=2):
    # no need to make a class here, as procedure is pretty straight forward
    # we're only going to attempt to produce data on quarter cycles
    phom_dir = pjoin(nc_state.replace("_state.nc","/other"), "voro")
    if not _nc_good(nc_state.replace("_state.nc","_data.nc")):
        return
    os.makedirs(phom_dir, exist_ok=True)
    
    s = netCDF4.Dataset(nc_state, 'r')
    nframes = len(s.dimensions['rec'])
    #SSPq = 25

    #with gsd.hoomd.open(name=traj, mode='rb') as f:
    for i in range(0,nframes):
        #s = f[i]
        #print(i)
        
        out = pjoin(phom_dir, f"{i}_qks")

        if os.path.exists(out+'.npz'):
            continue

        # construct voronoi tesselation
        voro = _get_voro_nc(s, i)

        # obtain Q_k's
        qks, verts = _get_Qks_nc(voro, s, i)

        print(np.mean(qks), np.std(qks), scipy.stats.skew(qks))

        np.savez_compressed(out, qks=qks, verts=verts)
    
def alpha_complex_producer_nc(nc_state, dim=2):
    # no need to make a class here, as procedure is pretty straight forward
    # we're only going to attempt to produce data on quarter cycles
    phom_dir = pjoin(nc_state.replace("_state.nc","/other"), "phom")
    if not _nc_good(nc_state.replace("_state.nc","_data.nc")):
        return
    os.makedirs(phom_dir, exist_ok=True)
    
    s = netCDF4.Dataset(nc_state, 'r')
    nframes = len(s.dimensions['rec'])

    # infer SPP
    #SSPq = _get_SSPq(gsd_file_root)
    SSPq = 25

    
    for i in range(0,nframes, SSPq):
        embed, rad2 = tri.get_configuration(s, i)
        comp =  tri.construct_triangulation(embed, rad2)

        # save these three to disk
        # out = pjoin(phom_dir, f"{i}_phom.pkl")
        # so it appears we can't save the complex object
        # I guess our best chance is to save all of the dealauney pairs at each quarter cycle frame
        #with open(out, 'wb') as f:
        #    pickle.dump({"comp":comp,"embed":embed,"rad2":rad2}, f)

        #if i % 2*SSPq == 0:
        # if at a strobascopic step, produce persistence data
        birthDeath = _get_births_deaths(comp, embed, rad2)
        out = pjoin(phom_dir, f"{i}_bd")

        np.savez_compressed(out, bd=birthDeath)
            
        #save birthDeath to disk