import numpy as np

def coordgrid(x_spacing, y_spacing, z_spacing):
    """ Given tuples for dim spacings (start, step, stop) in 3
    dimensions, return an nvoxel x ncoord matrix containing the coordinates
    corresponding to the specified grid
    """
    x_start, x_step, x_stop = map(float, x_spacing)
    y_start, y_step, y_stop = map(float, y_spacing)
    z_start, z_step, z_stop = map(float, z_spacing)
    nx = int((x_stop - x_start)/x_step + 1)
    ny = int((y_stop - y_start)/y_step + 1)
    nz = int((z_stop - z_start)/z_step + 1)
    x,y,z = np.mgrid[x_start:x_stop:1j*nx,
                     y_start:y_stop:1j*ny,
                     z_start:z_stop:1j*nz]
    cgrid = np.concatenate( (x.ravel()[:,np.newaxis],
                             y.ravel()[:,np.newaxis],
                             z.ravel()[:,np.newaxis]), axis=1)
    return cgrid

class RectilinearAnatomicalSpaceGrid (object):
    def __init__(self, anatomical_extents = [0,1,0,1,0,1]):
        self.left = anatomical_extents[0]
        self.right = anatomical_extents[1]
        self.post = anatomical_extents[2]
        self.ant = anatomical_extents[3]
        self.inf = anatomical_extents[4]
        self.sup = anatomical_extents[5]

    def index_list(self, dr):
        x_extent = int((self.right-self.left)/dr[0] + 0.5)
        y_extent = int((self.ant-self.post)/dr[1] + 0.5)
        z_extent = int((self.sup - self.inf)/dr[2] + 0.5)
        i_list = coordgrid( (0, 1, x_extent),
                            (0, 1, y_extent),
                            (0, 1, z_extent) )
        xform = self.index2vox_xform(dr)
        return i_list.astype('i'), xform

    def voxel_list(self, dr):
        v_list = coordgrid((self.left, dr[0], self.right),
                           (self.post, dr[1], self.ant),
                           (self.inf,  dr[2], self.sup))
        xform = self.index2vox_xform(dr)
        return v_list, xform
    
    def index2vox_xform(self, dr, affine=None):
        xform = np.zeros((4,4), 'd')
        for i in range(3):
            xform[i,i] = dr[i]
        xform[:,-1] = self.left, self.post, self.inf, 1
        # handle affine at some later date
        return xform

    def transform_voxels(self, voxels, dr):
        T = self.index2vox_xform(dr)
        nvx = voxels.shape[0]
        voxels_full = np.concatenate((voxels, np.ones((nvx,1))), axis=-1)
        indices = np.empty((nvx,3), 'i')
        indices[:] = (np.linalg.solve(T, voxels_full.T).T)[:,:3]
        return indices

    def transform_indices(self, indices, dr):
        T = self.index2vox_xform(dr)
        nvx = indices.shape[0]
        indices_full = np.concatenate((indices, np.ones((nvx,1))), axis=-1)
        
        voxels = np.empty((nvx,3), 'd')
        voxels[:] = np.dot(indices_full, T.T)[:,:3]
        return voxels

    def dim_sizes(self, dr):
        nx = int((self.right - self.left)/dr[0] + 1)
        ny = int((self.ant - self.post)/dr[1] + 1)
        nz = int((self.sup - self.inf)/dr[2] + 1)
        return (nx, ny, nz)
        
MNI_space = RectilinearAnatomicalSpaceGrid([-90.5, 90.5,  # left to right
                                            -108.5, 108.5, # post to ant
                                            -90.5, 90.5]) # inf to sup
##     BEAM_left = -90; BEAM_right = 90
##     BEAM_post = -125; BEAM_ant = 90
##     BEAM_inf = -70; BEAM_sup = 105
# these extents are rounded to the nearest 5mm ?
BEAM_space = RectilinearAnatomicalSpaceGrid([-90., 90.,
                                             -125., 90.,
                                             -70., 105.])
