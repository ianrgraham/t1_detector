"""
What do we need to do to detect voids?
We can easily draw the Delaunay triangulation
Categorize Delaunay edges as GAPS or BONDS (there will need to be a length scale introduced here)
Find all Delaunay cells with at least one GAP edge
Build a dictionary of cells to GAPS, and GAPS to cells
loop over all GAPS, building a list of voids as we go
for every GAP, look up cells that touch the gap, and then look at gaps that touch that cell
Add GAPS to a used set to check while we go
Time complexity should be roughly linear
"""

paths = ("/data1/shared/igraham/lib_persistent_homology/python_src:"
    "/data1/shared/igraham/new_lib_persistent_homology:"
    "/home1/igraham/Projects/quasilocalized_modes").split(':')

import sys
import os
import numpy as np
for p in paths:
    sys.path.insert(0,p)

pjoin = os.path.join

import phom
import triangulation as tri
import hessian_calc as hc
import detect as dt

from collections import defaultdict

#@njit
def is_gapped(comp, embed, rad, e):

    box_mat = embed.box_mat
    L = np.diagonal(box_mat)

    (vi, vj) = comp.get_facets(e)
        
    vposi = embed.get_vpos(vi)
    vposj = embed.get_vpos(vj)
    
    vbvec = embed.get_vdiff(vposi, vposj)
    bvec = box_mat.dot(vbvec)

    if np.linalg.norm(bvec) < rad[vi] + rad[vj]:
        return True
    else:
        return False


def get_void_tree(v, edges2cells, cells2edges, first=False, olde=[]):
    tmp_v = []
        
    edges = cells2edges[v]
    for e in edges:
        if e not in olde:
            olde.append(e)
            for v2 in edges2cells[e]:
                if v != v2:
                    tmp_v.append(v2)
                    tmp_v.extend(get_void_tree(v2, edges2cells, cells2edges, olde=olde))
    if first:
        tmp_v.append(v)
    return tmp_v

def find_voids(comp, embed, rad):

    edges2cells = defaultdict([])
    cells2edges = defaultdict([])
    # find gaps
    # loop over edges
    #   grab particles
    #   check if distance is greater than sum radii
    #       add edge id to gaps, value append cell id
    #       add cell id to voids, value append edge id
    edge_range = comp.dcell_range[1]
    for e in range(*edge_range):
        if is_gapped(comp, embed, rad, e):
            cells = comp.get_cofaces(e)
            edges2cells[e].extend(cells)
            for c in cells:
                cells2edges[c].append(e)


    # connect voids
    # loop over voids
    #   grab gaps of void
    #   for gaps in void
    #       find voids not equal to main void
    #       for cells in new void
    skip = []
    voids = []
    for v in cells2edges.keys:
        if v not in skip:
            voids.append(get_void_tree(v, cells2edges, edges2cells, first=True))
            skip.extend(voids[-1])

    # return list of voids (list of list of delauney cell ids)
    return voids

def void_qk(comp, embed, rad):

    # fetch Qk

    # fetch voids
    voids = find_voids(comp, embed, rad)


def hoomd_get_void_qk(gsd_file_root, i, r_func=hc.get_hoomd_bidisperse_r, dim=2):

    traj = pjoin(gsd_file_root, 'traj.gsd')
    # obtain Q_k's
    with gsd.hoomd.open(name=traj, mode='rb') as f:
        s = f[i]
        voro = dt._get_voro_hoomd(s, r_func=r_func)
        rad = np.vectorize(r_func)(s.particles.typeid)
        qks, _ = dt._get_Qks(voro, s)
        