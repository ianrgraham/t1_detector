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
for p in paths:
    sys.path.insert(0,p)

pjoin = os.path.join

import phom
import triangulation as tri
import hessian_calc as hc

from collections import defaultdict

@njit
def is_gapped(comp, embed, rad, ):
    pass

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
    for v in cells2edges.keys:
        if v not in skip:
            edges = cells2edges[v]
            # need to make this a recursive call
            for e in edges:
                for v in edges2cells[e]:

    # return list of voids (list of list of delauney cell ids)