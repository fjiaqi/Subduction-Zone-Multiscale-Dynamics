import subprocess
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import logging
from pathlib import Path
from datetime import datetime

if(len(sys.argv) != 3):
    raise ValueError(f"Invalid number of arguments: {len(sys.argv)}. Must be 3.")

initial_guesses = [7, 9]
dt = 25.

param_idx = int(sys.argv[1])
param_value = float(sys.argv[2])
param_list = ['fdynmid', 'fdynup', 'fst', 'etamax', 'ymax', 
              'wdisl', 'Gm', 'Gsz', 'Cm', 'Csz',
              'wpa', 'epa', 'gfactor', 'Y', 'Z']
param_name = param_list[param_idx]
param_s = f'{param_name}_{param_value:.2e}'

output_dir_parent = '/home1/08524/jqfang/scratch/newnew_2607'

log_dir = f'{output_dir_parent}/{param_s}/'
Path(log_dir).mkdir(exist_ok=True, parents=True)
logger = logging.getLogger("optimization_logger")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(Path(log_dir) / f"optimization.log", mode='w')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


reference_length = 1.0e6
reference_viscosity = 1.0e19
gravity = 10.
reference_density = 3300.
reference_expansivity = 3e-5
reference_diffusivity = 1e-6
reference_T0 = 300.
reference_T1 = 1700.
reference_time = reference_length**2/reference_diffusivity
reference_velocity = reference_length/reference_time
reference_stress = reference_viscosity/reference_time
s_per_year = 3600.*24.*365.25
R_const = 8.314

Ra = (reference_expansivity*reference_density*gravity*(reference_T1-reference_T0)*reference_length**3)/(reference_diffusivity*reference_viscosity)
Rb = (reference_density*gravity*reference_length**3)/(reference_diffusivity*reference_viscosity)

def l_m(length):
    return length*reference_length

def l_km(length):
    return length*reference_length/1e3

def t_s(time):
    return time*reference_time

def t_yr(time):
    return time*reference_time/s_per_year

def v_m_s(velocity):
    return velocity*reference_velocity

def v_mm_yr(velocity):
    return velocity*reference_velocity*s_per_year*1e3

def v_cm_yr(velocity):
    return velocity*reference_velocity*s_per_year*1e2

def tau_Pa(stress):
    return stress*reference_stress

def tau_MPa(stress):
    return stress*reference_stress/1e6

def logeta_Pa_s(viscosity):
    return np.log10(viscosity*reference_viscosity)

def logeps_s(strain_rate):
    return np.log10(strain_rate/reference_time)

def load_tmesh(output_dir, yres):
    with h5py.File(f'{output_dir}/tracer_swarm_0.h5', 'r') as h5:
        mesh_loc = h5['data'][...]
        if(yres > 0):
            arg_mesh = np.lexsort((mesh_loc[:, 0], mesh_loc[:, 1], mesh_loc[:, 2]))
        else:
            arg_mesh = np.lexsort((mesh_loc[:, 0], mesh_loc[:, 1]))
        mesh_loc = mesh_loc[arg_mesh]

    ndims = 3 if yres > 0 else 2
    return l_km(mesh_loc).reshape((257, yres+1, 1025, ndims)), arg_mesh

def load_tvel(output_dir, step, yres, arg_mesh, flag='l'):
    with h5py.File(f'{output_dir}/tracer_velocity_{flag}{step}.h5', 'r') as h5:
        mesh_vel = h5['data'][...][arg_mesh]

    ndims = 3 if yres > 0 else 2
    return v_mm_yr(mesh_vel).reshape((257, yres+1, 1025, ndims))

def extract_surf_loc(mesh_loc):
    return mesh_loc[-1, 0, :, 0]

def extract_surf_vel(mesh_vel):
    return mesh_vel[-1, 0, :, 0]

def load_sz_args(output_dir, step, flag, sz_idx=3, basic_dir=None):
    if(basic_dir is None):
        basic_dir = output_dir

    with h5py.File(f'{basic_dir}/tracer_swarm_0.h5', 'r') as h5:
        mesh_loc = h5['data'][...]

    with h5py.File(f'{basic_dir}/tracer_material_0.h5', 'r') as h5:
        mesh_mat = h5['data'][...]
        arg_sz = np.where(mesh_mat == sz_idx)[0]

    with h5py.File(f'{output_dir}/tracer_velocity_{flag}{step}.h5', 'r') as h5:
        mesh_vel = h5['data'][...]

    return l_km(mesh_loc[arg_sz]), v_mm_yr(mesh_vel[arg_sz])

def extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins, plim=5):
    ndims = sz_loc.shape[-1]
    if(ndims == 2):
        sz_dv = np.zeros((len(sz_zbins)-1, ndims))
        for j in range(len(sz_zbins)-1):
            arg_bin = np.where((sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
            sz_dv[j] = np.percentile(sz_vel[arg_bin], 100-plim, axis=0) - np.percentile(sz_vel[arg_bin], plim, axis=0) if len(arg_bin) > 0 else np.nan
    elif(len(sz_ybins) < 3):
        sz_dv = np.zeros((len(sz_zbins)-1, ndims))
        for j in range(len(sz_zbins)-1):
            arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[0]) & (sz_loc[:, 1] <= sz_ybins[-1]) & 
                                (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
            sz_dv[j] = np.percentile(sz_vel[arg_bin], 100-plim, axis=0) - np.percentile(sz_vel[arg_bin], plim, axis=0) if len(arg_bin) > 0 else np.nan
    else:
        sz_dv = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1, ndims))
        sz_dv.fill(np.nan)
        for i in range(len(sz_ybins)-1):
            for j in range(len(sz_zbins)-1):
                arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                                    (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
                sz_dv[i, j] = np.percentile(sz_vel[arg_bin], 100-plim, axis=0) - np.percentile(sz_vel[arg_bin], plim, axis=0) if len(arg_bin) > 0 else np.nan

    return np.linalg.norm(sz_dv, axis=-1)




guesses = initial_guesses
i1, i2 = guesses
logger.debug(f'Iteration {0}: {guesses[0]}, {guesses[1]}...')

for i in guesses:
    if(Path(f'{output_dir_parent}/{param_s}/restart_i{i}/data/velocity_c0.h5').is_file()):
        continue
    else:
        subprocess.run(
            [
                "mpirun", "-n", "64",
                sys.executable, "param_impose_model_all.py",
                sys.argv[1], sys.argv[2], str(i)
            ],
            check=True
        )

output_dir = f'{output_dir_parent}/{param_s}/data/'
yres = 0
mesh_loc, arg_mesh = load_tmesh(output_dir, yres)
surf_x = extract_surf_loc(mesh_loc)

flag = 'l'
step = 19
mesh_vel = load_tvel(output_dir, step, yres, arg_mesh, flag)
surf_vx = extract_surf_vel(mesh_vel)
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

for it_cnt in range(5):
    load_t = np.empty(0)
    recur_t = np.empty(0)
    for i in guesses:
        output_dir = f'{output_dir_parent}/{param_s}/restart_i{i}/data/'
        sz_loc, sz_vel = load_sz_args(output_dir, 0, 'c')
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
        load_t = np.append(load_t, 25*(i+1)*2)
        recur_t = np.append(recur_t, np.nanmax(sz_dv)/np.nanmax(surf_vx))

    i1, i2 = guesses
    t1, t2 = recur_t
    r_ti = (t2 - t1) / (i2 - i1)
    i_next = (2*dt-t1+i1*r_ti)/(r_ti-2*dt)

    # fig, ax = plt.subplots(figsize=(6, 6))

    # ax.plot([200, 1000], [200, 1000], c='k', ls='--')
    # ax.plot(load_t, recur_t, c='C0')
    # ax.scatter(load_t, recur_t, c='C0')
    # ax.scatter(25*(i_next+1)*2, r_ti*(i_next-i1)+t1, c='C1')
    
    # ax.set_xlabel('Loading time * 2 (yrs)')
    # ax.set_ylabel('Recurrence interval (yrs)')
    # plt.tight_layout()
    # plt.savefig('figures/tmp.png')

    logger.debug(f'Get i = {i_next:.2f}...')
    if(np.abs(i_next - i1) <= 0.5):
        i_next = i1
        print(i_next)
        break
    elif(np.abs(i_next - i2) <= 0.5):
        i_next = i2
        print(i_next)
        break
    else:
        i_next = np.maximum(np.rint(i_next).astype(int), 0)
        if(i_next < i1):
            guesses = [i_next, i1]
        elif(i_next > i2):
            guesses = [i2, i_next]
        elif(i_next - i1 < i2 - i_next):
            guesses = [i1, i_next]
        else:
            guesses = [i_next, i2]

    logger.debug(f'Iteration {it_cnt+1}: {guesses[0]}, {guesses[1]}...')
    if(Path(f'{output_dir_parent}/{param_s}/restart_i{i_next}/data/velocity_c0.h5').is_file()):
        continue
    else:
        subprocess.run(
            [
                "mpirun", "-n", "64",
                sys.executable, "param_impose_model_all.py",
                sys.argv[1], sys.argv[2], str(i_next)
            ],
            check=True
        )

subprocess.run(
    [
        "ln", "-s",
        f"{output_dir_parent}/{param_s}/restart_i{i_next}",
        f"{output_dir_parent}/{param_s}/restart",
    ],
    check=True
)

logger.debug(f'Final: {i_next}')
