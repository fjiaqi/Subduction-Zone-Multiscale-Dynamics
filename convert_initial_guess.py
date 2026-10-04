import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import h5py
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.debug(f'Started conversion')

dir_2d = '/home/x-jqfang/scratch/newnew_2607/fdynmid_3.00e-01'
meshfile_2d = f'{dir_2d}/data/mesh_0.h5'
vfile_2d = f'{dir_2d}/data/velocity_l19.h5'
pfile_2d = f'{dir_2d}/data/pressure_l19.h5'

dir_3d = '/home/x-jqfang/scratch/newnew_2607/3d_basic'
meshfile_3d = f'{dir_3d}/data/mesh_0.h5'

output_dir_3d = dir_2d
vfile_3d = f'{output_dir_3d}/data/velocity_n0.h5'
pfile_3d = f'{output_dir_3d}/data/pressure_n0.h5'

logger.debug(f'Started loading data')

with h5py.File(meshfile_2d, 'r') as h5:
    mesh_2d = h5['vertices'][...]
    map_2d = h5['en_map'][...]
logger.debug(f'2D mesh loaded')

submesh_2d = np.array([np.mean(mesh_2d[indices], axis=0) for indices in map_2d])
logger.debug(f'2D submesh calculated')

with h5py.File(vfile_2d, 'r') as h5:
    v_2d = h5['data'][...]
logger.debug(f'2D velocity loaded')

with h5py.File(pfile_2d, 'r') as h5:
    p_2d = h5['data'][...]
logger.debug(f'2D pressure loaded')

with h5py.File(f'{dir_3d}/data/mesh_0.h5', 'r') as h5:
    mesh_3d = h5['vertices'][...]
    map_3d = h5['en_map'][...]
logger.debug(f'3D mesh loaded')

v_func = NearestNDInterpolator(mesh_2d, v_2d)
# v_func = LinearNDInterpolator(mesh_2d, v_2d)
v_3d = np.zeros(mesh_3d.shape)
v_3d[:, ::2] = v_func(mesh_3d[:, ::2])
logger.debug(f'3D velocity calculated')

with h5py.File(vfile_3d, 'w') as h5:
    h5.create_dataset('data', data=v_3d)
logger.debug(f'3D velocity written')

submesh_3d = np.array([np.mean(mesh_3d[indices], axis=0) for indices in map_3d])
logger.debug(f'3D submesh calculated')

p_func = NearestNDInterpolator(submesh_2d, p_2d)
# p_func = LinearNDInterpolator(submesh_2d, p_2d)
p_3d = p_func(submesh_3d[:, ::2])
logger.debug(f'3D pressure calculated')

with h5py.File(pfile_3d, 'w') as h5:
    h5.create_dataset('data', data=p_3d)
logger.debug(f'3D pressure written')

logger.debug(f'Finished conversion')
