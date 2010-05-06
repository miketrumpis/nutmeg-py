import scipy as sp
import scipy.io
import numpy as np
from nutmeg.core.tfbeam import tfbeam_from_file
from nutmeg.stats import beam_stats as bs

def compare_to_mat_beam(pybeam, matbeam, comp='diff', return_mat=False):
    if type(pybeam) == str:
        pybeam = tfbeam_from_file(pybeam)
    if type(matbeam) == str:
        matbeam = tfbeam_from_file(matbeam)

    inter_vox, vox_indices = bs.find_vox_intersection([pybeam.voxels,
                                                       matbeam.voxels])
    if len(inter_vox) != len(pybeam.voxels) or \
       len(pybeam.voxels) != len(matbeam.voxels):
        print 'mis-matched voxels between MATLAB and Python beams'
        # trim pybeam to the inter_vox
        s = np.array( [pybeam.s[pybeam.vox_lookup[tuple(v)]]
                       for v in inter_vox] )
        pybeam = pybeam.from_new_dataset(
            s, inter_vox, fixed_comparison=pybeam.uses
            )

    # need to re-order the matbeam signal since the voxels are in diff. order
    s = np.zeros_like(pybeam.s)
    for vsig, v in zip(matbeam.s, matbeam.voxels):
        try:
            i = pybeam.vox_lookup[tuple(v)]
            s[i] = vsig
        except KeyError:
            pass
    matbeam = matbeam.from_new_dataset(
        s, new_vox=pybeam.voxels.copy(), fixed_comparison=matbeam.uses
        )
    if comp=='diff':
        
        err = pybeam.from_new_dataset(
            matbeam.s - pybeam.s,
            new_vox=inter_vox,
            fixed_comparison='diff'
            )
    elif comp=='pctdiff':
        err = pybeam.from_new_dataset(
            100.*(matbeam.s-pybeam.s)/pybeam.s,
            new_vox=inter_vox,
            fixed_comparison='pctdiff'
            )
    elif comp=='sqrdiff':
        err = pybeam.from_new_dataset(
            (matbeam.s-pybeam.s)**2,
            new_vox=inter_vox,
            fixed_comparison='sqrdiff')
    if return_mat:
        return err, matbeam
    else:
        return err
