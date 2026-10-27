# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
from glob import glob

params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'RdYlBu',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 16, # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
}
mpl.rcParams.update(params)

# %%
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

def T_K(temperature):
    return reference_T0+temperature*(reference_T1-reference_T0)

def T_C(temperature):
    return reference_T0+temperature*(reference_T1-reference_T0)-273.15


def second_invariant(A, dim=2):
    if(dim == 2 or A.shape[1] == 3):
        res = np.sqrt(A[:, 0]**2/2. + A[:, 1]**2/2. + A[:, 2]**2)
    elif(dim == 3 or A.shape[1] == 6):
        res = np.sqrt(A[:, 0]**2/2. + A[:, 1]**2/2. + A[:, 2]**2/2. +
                      A[:, 3]**2 + A[:, 4]**2 + A[:, 5]**2)
    else:
        raise ValueError('dim must be 2 or 3')
    return res

# %%
def load_mesh(output_dir, yres):
    with h5py.File(f'{output_dir}/mesh_0.h5', 'r') as h5:
        mesh_loc = h5['vertices'][...]
        if(yres > 0):
            arg_mesh = np.lexsort((mesh_loc[:, 0], mesh_loc[:, 1], mesh_loc[:, 2]))
        else:
            arg_mesh = np.lexsort((mesh_loc[:, 0], mesh_loc[:, 1]))
        mesh_loc = mesh_loc[arg_mesh]

    ndims = 3 if yres > 0 else 2
    return l_km(mesh_loc).reshape((257, yres+1, 1025, ndims)), arg_mesh

def load_vel(output_dir, step, yres, arg_mesh, flag='l'):
    with h5py.File(f'{output_dir}/velocity_{flag}{step}.h5', 'r') as h5:
        mesh_vel = h5['data'][...][arg_mesh]

    ndims = 3 if yres > 0 else 2
    return v_mm_yr(mesh_vel).reshape((257, yres+1, 1025, ndims))

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

def load_ttemp(output_dir, yres, arg_mesh):
    with h5py.File(f'{output_dir}/tracer_temperature_0.h5', 'r') as h5:
        mesh_temp = h5['data'][...][arg_mesh]

    return T_C(mesh_temp).reshape((257, yres+1, 1025, 1))

def load_tvisc(output_dir, step, yres, arg_mesh, flag='l'):
    with h5py.File(f'{output_dir}/tracer_viscosity_{flag}{step}.h5', 'r') as h5:
        mesh_visc = h5['data'][...][arg_mesh]

    return logeta_Pa_s(mesh_visc).reshape((257, yres+1, 1025, 1))
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

    with h5py.File(f'{output_dir}/tracer_strainrate_{flag}{step}.h5', 'r') as h5:
        mesh_strainrate = h5['data'][...]

    with h5py.File(f'{output_dir}/tracer_stress_{flag}{step}.h5', 'r') as h5:
        mesh_stress = h5['data'][...]

    with h5py.File(f'{output_dir}/tracer_viscosity_{flag}{step}.h5', 'r') as h5:
        mesh_visc = h5['data'][...]

    return l_km(mesh_loc[arg_sz]), v_mm_yr(mesh_vel[arg_sz]), logeps_s(mesh_strainrate[arg_sz]), tau_MPa(mesh_stress[arg_sz]), logeta_Pa_s(mesh_visc[arg_sz])

def extract_surf_loc(mesh_loc):
    return mesh_loc[-1, 0, :, 0]

def extract_surf_vel(mesh_vel):
    return mesh_vel[-1, 0, :, 0]

def extract_sz_var_mean(sz_loc, sz_var, sz_ybins, sz_zbins):
    ndims = sz_loc.shape[-1]
    if(ndims == 2):
        sz_var_mean = np.zeros(len(sz_zbins)-1)
        sz_var_mean.fill(np.nan)
        for j in range(len(sz_zbins)-1):
            arg_bin = np.where((sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
            sz_var_mean[j] = np.mean(sz_var[arg_bin]) if len(arg_bin) > 0 else np.nan
    elif(len(sz_ybins) < 3):
        sz_var_mean = np.zeros(len(sz_zbins)-1)
        sz_var_mean.fill(np.nan)
        for j in range(len(sz_zbins)-1):
            arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[0]) & (sz_loc[:, 1] <= sz_ybins[-1]) & 
                                (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
            sz_var_mean[j] = np.mean(sz_var[arg_bin]) if len(arg_bin) > 0 else np.nan
    else:
        sz_var_mean = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))
        sz_var_mean.fill(np.nan)
        for i in range(len(sz_ybins)-1):
            for j in range(len(sz_zbins)-1):
                arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                                    (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
                sz_var_mean[i, j] = np.mean(sz_var[arg_bin]) if len(arg_bin) > 0 else np.nan

    return sz_var_mean

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

# %%
output_dir = '/home/x-jqfang/scratch/model_2610/fdynmid_3.00e-01/data/'
yres = 0
mesh_loc, arg_mesh = load_tmesh(output_dir, yres)

flag = 'l'
step = 19
mesh_vel = load_tvel(output_dir, step, yres, arg_mesh, flag)

sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, step, flag)
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]
sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
sz_meansr = extract_sz_var_mean(sz_loc, sz_sr, sz_ybins, sz_zbins)

# %%
mesh_temp = load_ttemp(output_dir, yres, arg_mesh)

core_left, core_right = -100, 400
core_bottom, core_top = -125, 0
core_buffer = 5
arg_x = np.where((mesh_loc[0, 0, :, 0] >= core_left-core_buffer) &
                 (mesh_loc[0, 0, :, 0] <= core_right+core_buffer))[0]
arg_z = np.where((mesh_loc[:, 0, 0, -1] >= core_bottom-core_buffer) &
                 (mesh_loc[:, 0, 0, -1] <= core_top+core_buffer))[0]

fig, ax = plt.subplots(figsize=(12, 3))

ax.set_aspect('equal')
mappable = ax.contourf(mesh_loc[0, 0, arg_x, 0], mesh_loc[arg_z, 0, 0, -1], mesh_temp[:, 0, arg_x, 0][arg_z],
                        levels=np.linspace(0, 1450, 101), cmap='coolwarm')

cs = ax.contour(mesh_loc[0, 0, arg_x, 0], mesh_loc[arg_z, 0, 0, -1], mesh_temp[:, 0, arg_x, 0][arg_z],
              colors='k', levels=np.arange(200, 1301, 200), zorder=10)
plt.clabel(cs, fontsize=10, fmt='%d', inline_spacing=5)

slab_top_coords = np.loadtxt('../geodata/slab_top_data.txt')
ax.plot(slab_top_coords[:, 0], slab_top_coords[:, 1], c='darkgray', ls='--', lw=2)
# arg_sz_bot = np.where(shear_zone_bottom_coords[:, 0] >= 0)[0]
# ax.plot(l_km(shear_zone_bottom_coords[arg_sz_bot, 0]), l_km(shear_zone_bottom_coords[arg_sz_bot, 1]), 
#         c='darkgray', ls='--', lw=2)

ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)

ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$z$ (km)')
ax.minorticks_on()

plt.tight_layout()
plt.savefig('fig_temp.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(6, 3))

ax.plot([0, -80], [0.7, 0.7], label=r'$\mu_{s}$', lw=3, c='b')
ax.plot([0, -5, -10, -60, -70, -80], [0.7, 0.7, 0.2, 0.2, 0.7, 0.7], label=r'$\mu_{d}$', ls='--', lw=3, c='r')
ax.set_xlim(0, -80)
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$\mu$')
ax.legend(loc='lower right', fontsize=12)
ax.minorticks_on()

plt.tight_layout()
plt.savefig('fig_friction.pdf', transparent=True)

# %%
output_dir = '/home/x-jqfang/scratch/model_2610/fdynmid_3.00e-01/data/'
output_dir_c = '/home/x-jqfang/scratch/model_2610/fdynmid_3.00e-01/restart/data/'
yres = 0

mesh_loc, arg_mesh = load_tmesh(output_dir, yres)
surf_x = extract_surf_loc(mesh_loc)

mesh_vel_l = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
surf_vx_l = extract_surf_vel(mesh_vel_l)
mesh_vel_i = load_tvel(output_dir, 9, yres, arg_mesh, 'i')
surf_vx_i = extract_surf_vel(mesh_vel_i)
mesh_vel_c = load_tvel(output_dir_c, 0, yres, arg_mesh, 'c')
surf_vx_c = extract_surf_vel(mesh_vel_c)

sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 19, 'l')
sz_dv_l = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 9, 'i')
sz_dv_i = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
sz_dv_c = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

ax = axes[0]

ax.plot(surf_x, surf_vx_i/10, 
        c='b', lw=3, label='Interseismic')
ax.plot([], [], c='r', lw=3, label='Coseismic')

ax.plot(surf_x, surf_vx_l/10,
        c='k', ls='--', lw=3, label='Long-term')

axt = ax.twinx()
axt.plot(surf_x, surf_vx_c/1e3, c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='b')
ax.legend(loc='lower left', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$u_x$ (m)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()
ax.set_title('(a)', loc='left')

ax = axes[1]

ax.plot(sz_zbins[1:], sz_dv_i/10, 
        c='b', lw=3, label='Interseismic')
ax.plot([], [], c='r', lw=3, label='Coseismic')

ax.plot(sz_zbins[1:], sz_dv_l/10,
        c='k', ls='--', lw=3, label='Long-term')

axt = ax.twinx()
axt.plot(sz_zbins[1:], sz_dv_c/1e3, c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$V$ (cm/yr)', c='b')
# ax.legend(loc='center left', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$U$ (m)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()
ax.set_title('(b)', loc='left')

fig.tight_layout()
plt.savefig('fig_surf_fault_vel.pdf', transparent=True)

# %%
output_dir = '/home/x-jqfang/scratch/model_2610/L_8.19e+07/fdynmid_3.00e-01/data/'
output_dir_c = '/home/x-jqfang/scratch/model_2610/L_5.12e+06/fdynmid_3.00e-01/restart/data/'
yres = 2

mesh_loc, arg_mesh = load_tmesh(output_dir, yres)
surf_x = extract_surf_loc(mesh_loc)

mesh_vel_l = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
surf_vx_l = extract_surf_vel(mesh_vel_l)
mesh_vel_i = load_tvel(output_dir, 10, yres, arg_mesh, 'i')
surf_vx_i = extract_surf_vel(mesh_vel_i)
mesh_vel_c = load_tvel(output_dir_c, 0, yres, arg_mesh, 'c')
surf_vx_c = extract_surf_vel(mesh_vel_c)

sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 19, 'l')
sz_dv_l = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 10, 'i')
sz_dv_i = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
sz_dv_c = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

ax = axes[0]

ax.plot(surf_x, surf_vx_i/10, 
        c='b', lw=3, label='Interseismic')
ax.plot([], [], c='r', lw=3, label='Coseismic')

ax.plot(surf_x, surf_vx_l/10,
        c='k', ls='--', lw=3, label='Long-term')

axt = ax.twinx()
axt.plot(surf_x, surf_vx_c/1e3, c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='b')
ax.legend(loc='lower left', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$u_x$ (m)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()
axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.set_title('(a)', loc='left')

ax = axes[1]

ax.plot(sz_zbins[1:], sz_dv_i/10, 
        c='b', lw=3, label='Interseismic')
ax.plot([], [], c='r', lw=3, label='Coseismic')

ax.plot(sz_zbins[1:], sz_dv_l/10,
        c='k', ls='--', lw=3, label='Long-term')

axt = ax.twinx()
axt.plot(sz_zbins[1:], sz_dv_c/1e3, c='r', lw=3)

ax.set_zorder(axt.get_zorder()+1)
ax.set_frame_on(False)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$V$ (cm/yr)', c='b')
# ax.legend(loc='center left', fontsize=12)
ax.tick_params(axis='y', labelcolor='b')
ax.minorticks_on()
axt.set_ylabel(r'$U$ (m)', c='r')
axt.tick_params(axis='y', labelcolor='r')
axt.minorticks_on()
axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.set_title('(b)', loc='left')

fig.tight_layout()
plt.savefig('fig_surf_fault_vel.pdf', transparent=True)

# %%


# %%
txt = np.loadtxt('../geodata/slab_top_data.txt')
xx = txt[::-1, 0]
zz = txt[::-1, 1]
# ss = np.cumsum(np.sqrt(1 + np.gradient(xx, zz)**2) * np.abs(np.gradient(zz)))
# ss -= ss.min()
# ds = np.diff(ss)
ds = np.sqrt(np.diff(xx)**2+np.diff(zz)**2)
ss = np.insert(np.cumsum(ds), 0, 0)
plt.plot(zz, ss)
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_sbins = np.interp(-sz_zbins, -zz, ss)
plt.plot(sz_zbins, sz_sbins)
plt.axvline(-5, c='k', ls='--')
plt.axvline(-10, c='k', ls='--')
plt.axvline(-20, c='k', ls='--')
plt.axvline(-30, c='k', ls='--')
plt.axvline(-40, c='k', ls='--')
plt.axvline(-50, c='k', ls='--')

# %%
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]
sz_dv = np.zeros((len(sz_zbins)-1, 3))

Z_list = ['6.00e+04', '5.00e+04', '4.00e+04', '3.00e+04', '2.00e+04', '1.00e+04']
Y_list = ['8.19e+07', '2.05e+07', '5.12e+06', '1.28e+06', '3.20e+05', '8.00e+04']
res_L = np.array([2*np.sqrt(float(Y_list[j])*5e3) for j in range(len(Y_list))])/1e3
res_dv = np.zeros((len(Z_list), len(Y_list), sz_dv.shape[0]))
res_W = np.zeros((len(Z_list), len(Y_list)))
res_M = np.zeros((len(Z_list), len(Y_list)))

output_dir = '/home/x-jqfang/scratch/model_2610/fdynmid_3.00e-01/restart/data/'
sz_dv_inf = np.zeros((len(sz_zbins)-1, 2))
sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
sz_dv_inf = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

for i in range(len(Z_list)):
    for j in range(len(Y_list)):
        output_dir = f'/home/x-jqfang/scratch/model_2610/L_{Y_list[j]}/Z_{Z_list[i]}/restart/data/'
        
        try:
            sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
        except:
            res_dv[i, j] = np.nan
            res_W[i, j] = np.nan
            res_M[i, j] = np.nan
            continue
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

        res_dv[i, j] = sz_dv
        dv_zz = np.interp(zz, sz_zbins[1:], sz_dv)
        arg_s = np.where((zz >= sz_zbins.min())
                         & (dv_zz > np.minimum(2e3, 0.1*dv_zz.max())) 
                         & (zz <= -5)
                         & (zz >= -np.minimum(float(Z_list[i])/1e3, 40)-10))[0]
        ds = np.sqrt(np.diff(xx)**2+np.diff(zz)**2)
        ss = np.insert(np.cumsum(ds), 0, 0)
        res_W[i, j] = np.ptp(ss[arg_s])
        res_M[i, j] = np.pi/4 * 30e9 * np.sum(ds[arg_s]*1e3 * (dv_zz[arg_s]+dv_zz[arg_s+1])/2/1e3) * res_L[j]*1e3

# %%
fig, ax = plt.subplots(figsize=(12, 3.75))

dt = 1
color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Y_list))[::-1])

ax.plot(sz_zbins[1:], sz_dv_inf*dt/1e3, lw=3,
            label=r'$\infty$', c='k', ls='--', zorder=10)

for i in range(len(Y_list)):
    ax.plot(sz_zbins[1:], res_dv[0, i]*dt/1e3, lw=3,
            label=f'{1280/2**i:.0f}', c=color_list[i])

ax.set_xlim(0, -75)
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)', c='k')
ax.legend(loc='upper right', title=r'$L$ (km)', ncol=2)
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('fig_Y2.pdf', transparent=True)

# %%
core_left, core_right = -100, 400
core_bottom, core_top = -125, 0
dt = 1

vlim = 20

fig, axes = plt.subplots(3, 2, figsize=(12, 4.5), sharex=True, sharey=True)

with h5py.File(f'/home/x-jqfang/scratch/model_2610/L_{Y_list[0]}/Z_6.00e+04/data/tracer_swarm_0.h5', 'r') as f:
    mesh_3d = f['data'][...]
    
for i in range(6):
    ax = axes.flatten()[i]
    ax.set_aspect('equal')
    with h5py.File(f'/home/x-jqfang/scratch/model_2610/L_{Y_list[i]}/Z_6.00e+04/restart/data/tracer_velocity_c0.h5', 'r') as f:
        vel_3d = f['data'][...]
        
    core_buffer = 5
    arg_frontcore = np.where((np.isclose(mesh_3d[:, 1], 0.)) &
                         (core_left-core_buffer <= l_km(mesh_3d[:, 0])) & 
                        (l_km(mesh_3d[:, 0]) <= core_right+core_buffer) &
                        (core_bottom-core_buffer <= l_km(mesh_3d[:, -1])) & 
                        (l_km(mesh_3d[:, -1]) <= core_top+core_buffer))[0]

    mappable = ax.tricontourf(l_km(mesh_3d[arg_frontcore, 0]), l_km(mesh_3d[arg_frontcore, -1]), 
                            np.linalg.norm(v_mm_yr(vel_3d[arg_frontcore]), axis=1)*dt/1e3, 
                            np.linspace(0, vlim, 101), cmap='RdYlBu_r', extend='max')

#     ax.plot(l_km(top_coords[:, 0]), l_km(top_coords[:, 1]), c='silver', ls='--', lw=2)
#     ax.plot(l_km(bottom_coords[:, 0]), l_km(bottom_coords[:, 1]), c='silver', ls='--', lw=2)
    
    ax.text(-90, -115, r'$L =$'+f'{1280/2**i:.0f} km',
            ha='left', va='bottom', c='k',
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, boxstyle='round'))

for ax in axes[:, 0]:
    ax.set_ylabel(r'$z$ (km)')
for ax in axes[-1, :]:
    ax.set_xlabel(r'$x$ (km)')
ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)
ax.minorticks_on()

cbax = fig.add_axes([0.92, 0.15, 0.01, 0.65])
cbar = fig.colorbar(mappable, cax=cbax, aspect=30)
cbar.ax.set_title(r'$u$ (m)', loc='left')
cbar.set_ticks(np.arange(0, vlim+0.01, 5))

plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.2, wspace=0.2)
plt.savefig('fig_Y1.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(12, 3.75))

dt = 1
color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Z_list))[::-1])

for i in range(len(Z_list)):
    ax.plot(sz_zbins[1:], res_dv[i, 2]*dt/1e3, lw=3,
            # label=f'{res_W[i].max():.0f}', c=color_list[i])
            label=f'{(70-i*10):.0f}', c=color_list[i],
            ls='-' if i > 0 else '--', zorder=20-i)

ax.set_xlim(0, -75)
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)', c='k')
# ax.legend(loc='upper right', title=r'$W$ (km)', ncol=2)
ax.legend(loc='upper right', title=r'$h$ (km)', ncol=2)
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

fig.tight_layout()
plt.savefig('fig_Z2.pdf', transparent=True)

# %%
core_left, core_right = -100, 400
core_bottom, core_top = -125, 0
dt = 1

vlim = 20

fig, axes = plt.subplots(3, 2, figsize=(12, 4.5), sharex=True, sharey=True)

with h5py.File(f'/home1/08524/jqfang/scratch/eq_2602/L_5.12e+06/Z_{Z_list[0]}/data/tracer_swarm_0.h5', 'r') as f:
    mesh_3d = f['data'][...]
    
for i in range(6):
    ax = axes.flatten()[i]
    ax.set_aspect('equal')
    with h5py.File(f'/home1/08524/jqfang/scratch/eq_2602/L_5.12e+06/Z_{Z_list[i]}/data/tracer_velocity_c0.h5', 'r') as f:
        vel_3d = f['data'][...]
        
    core_buffer = 5
    arg_frontcore = np.where((np.isclose(mesh_3d[:, 1], 0.)) &
                         (core_left-core_buffer <= l_km(mesh_3d[:, 0])) & 
                        (l_km(mesh_3d[:, 0]) <= core_right+core_buffer) &
                        (core_bottom-core_buffer <= l_km(mesh_3d[:, -1])) & 
                        (l_km(mesh_3d[:, -1]) <= core_top+core_buffer))[0]

    mappable = ax.tricontourf(l_km(mesh_3d[arg_frontcore, 0]), l_km(mesh_3d[arg_frontcore, -1]), 
                            np.linalg.norm(v_mm_yr(vel_3d[arg_frontcore]), axis=1)*dt/1e3, 
                            np.linspace(0, vlim, 101), cmap='RdYlBu_r', extend='max')

    ax.text(-90, -115, r'$h =$'+f'{70-i*10} km ('+r'$W \approx$'+f'{res_W[i].max():.0f} km)',
            ha='left', va='bottom', c='k', fontsize=14,
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, boxstyle='round'))

for ax in axes[:, 0]:
    ax.set_ylabel(r'$z$ (km)')
for ax in axes[-1, :]:
    ax.set_xlabel(r'$x$ (km)')
ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)
ax.minorticks_on()

cbax = fig.add_axes([0.92, 0.15, 0.01, 0.65])
cbar = fig.colorbar(mappable, cax=cbax, aspect=30)
cbar.ax.set_title(r'$u$ (m)', loc='left')
cbar.set_ticks(np.arange(0, vlim+0.01, 5))

plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.2, wspace=0.2)
plt.savefig('fig_Z1.pdf', transparent=True)

# %%


# %%
prefixes = ('wdisl', 'ymax', 'Gm', 'Gsz', 'fdynmid', 'fst')
label_list = (r'$\omega_{disl}$', r'$\tau_{y,max}$', r'$G_b$', r'$G_{sz}$', r'$\mu_{d,mid}$', r'$\mu_{s}$')

yres = 0
dir_names = sorted(
    d for d in glob('/home/x-jqfang/scratch/model_2610/*/data/')
    if Path(d).parents[0].name.startswith(prefixes)
)
dir_c_names = sorted(
    d for d in glob('/home/x-jqfang/scratch/model_2610/*/restart/data/')
    if Path(d).parents[1].name.startswith(prefixes)
)
mesh_loc, arg_mesh = load_tmesh(dir_names[0], yres)
surf_x = extract_surf_loc(mesh_loc)

sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

dt = 1

color_list = ['C0', 'C5', 'C3', 'C9', 'C4', 'C1']
ls_list = ['-', '-.', '-', '--', '-', ':']
marker_list = ['s', 'o', '^']

for k in range(len(prefixes)):
    dir_names_k = sorted(
        [d for d in dir_names
        if Path(d).parents[0].name.startswith(prefixes[k])],
        key=lambda d: float(Path(d).parents[0].name.split('_')[1])
    )
    dir_c_names_k = sorted(
        [d for d in dir_c_names
        if Path(d).parents[1].name.startswith(prefixes[k])],
        key=lambda d: float(Path(d).parents[1].name.split('_')[1])
    )
    vi = np.zeros(len(dir_names_k))
    Vc = np.zeros(len(dir_names_k))
    for i in range(len(dir_names_k)):
        output_dir = dir_names_k[i]
        output_dir_c = dir_c_names_k[i]
        mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
        surf_vel = extract_surf_vel(mesh_vel)
        sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
        vi[i] = surf_vel.max()
        Vc[i] = sz_dv.max()
        axes[0].scatter(vi[i]/10, Vc[i]/1e3, c=color_list[k], marker=marker_list[i], edgecolor='k', zorder=10)
        axes[1].scatter(vi[i]/10, Vc[i]/vi[i], c=color_list[k], marker=marker_list[i], edgecolor='k', zorder=10)
    axes[0].plot(vi/10, Vc/1e3, c=color_list[k], ls=ls_list[k], lw=3, label=label_list[k])
    axes[1].plot(vi/10, Vc/vi, c=color_list[k], ls=ls_list[k], lw=3, label=label_list[k])
    axes[0].scatter(vi[1]/10, Vc[1]/1e3, c='gray', marker=marker_list[1], edgecolor='k', zorder=10)
    axes[1].scatter(vi[1]/10, Vc[1]/vi[1], c='gray', marker=marker_list[1], edgecolor='k', zorder=10)
    # print(prefixes[k], vi, Vc/1e3)

axes[0].set_xlabel(r'$v_{pl}$ (cm/yr)')
axes[0].set_ylabel(r'$U_{max}$ (m)')
axes[0].minorticks_on()
axes[0].set_title('(a)', loc='left')

axes[1].set_xlabel(r'$v_{pl}$ (cm/yr)')
axes[1].set_ylabel(r'$T_r$ (year)')
axes[1].minorticks_on()
axes[1].legend()
axes[1].set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('fig_rheo_v_V.pdf', transparent=True)

# %%
k = 0
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

fig = plt.figure(figsize=(13, 6))

gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[4, 9], height_ratios=[1, 1])

ax = fig.add_subplot(gs[:, 0])

fmt = mticker.LogFormatterSciNotation(base=10)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_visc = load_tvisc(output_dir, 19, yres, arg_mesh, 'l')
    arg_x = np.where((surf_x > -500) & (surf_x < -100))[0]
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(np.mean(10**mesh_visc[:, 0, arg_x, 0], axis=-1), mesh_loc[:, 0, 0, -1], lw=3, 
            label=f'{case_value:.1f}', 
            c=c_list[i], ls=ls_list[i])

ax.set_xscale('log')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.set_ylabel(r'$z$ (km)')
ax.legend(loc='center right', fontsize=12, title=r'$\omega_{disl}$', title_fontsize=12)
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(a)', loc='left')

ax.set_ylim(mesh_loc[:, 0, 0, -1].min(), mesh_loc[:, 0, 0, -1].max())

ax = fig.add_subplot(gs[0, 1])

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{fmt(case_value):s}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

ax = fig.add_subplot(gs[1, 1])

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{fmt(case_value):s}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(c)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_wdisl.pdf', transparent=True)

# %%
fig = plt.figure(figsize=(18, 9))

k = 5
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

ax = fig.add_subplot(321)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'i')
    sz_meanst = extract_sz_var_mean(sz_loc, sz_st, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_meanst, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'Long-term $\tau_{II}$ (MPa)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$\mu_{s}$', title_fontsize=12)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_title('(a)', loc='left')
ax.set_title(r'Varying $\mu_s$')

ax = fig.add_subplot(323)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(c)', loc='left')

ax = fig.add_subplot(325)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(e)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

k = 4
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)[::-1]
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)[::-1]

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

ax = fig.add_subplot(322)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_meanst = extract_sz_var_mean(sz_loc, sz_st, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_meanst, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'Coseismic $\tau_{II}$ (MPa)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$\mu_{d,mid}$', title_fontsize=12)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_title('(b)', loc='left')
ax.set_title(r'Varying $\mu_{d,mid}$')

ax = fig.add_subplot(324)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(d)', loc='left')

ax = fig.add_subplot(326)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(f)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_f.pdf', transparent=True)

# %%
prefixes = ('wdisl', 'ymax', 'Gm', 'Gsz', 'fdynmid', 'fst')
label_list = (r'$\omega_{disl}$', r'$\tau_{y,max}$', r'$G_b$', r'$G_{sz}$', r'$\mu_{d,mid}$', r'$\mu_{s}$')

yres = 2
dir_names = sorted(
    d for d in glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/*/data/')
    if Path(d).parents[0].name.startswith(prefixes)
)
dir_c_names = sorted(
    d for d in glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/*/restart/data/')
    if Path(d).parents[1].name.startswith(prefixes)
)
mesh_loc, arg_mesh = load_tmesh(dir_names[0], yres)
surf_x = extract_surf_loc(mesh_loc)

sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

dt = 1

color_list = ['C0', 'C5', 'C3', 'C9', 'C4', 'C1']
ls_list = ['-', '-.', '-', '--', '-', ':']
marker_list = ['s', 'o', '^']

for k in range(len(prefixes)):
    dir_names_k = sorted(
        [d for d in dir_names
        if Path(d).parents[0].name.startswith(prefixes[k])],
        key=lambda d: float(Path(d).parents[0].name.split('_')[1])
    )
    dir_c_names_k = sorted(
        [d for d in dir_c_names
        if Path(d).parents[1].name.startswith(prefixes[k])],
        key=lambda d: float(Path(d).parents[1].name.split('_')[1])
    )
    vi = np.zeros(len(dir_names_k))
    Vc = np.zeros(len(dir_names_k))
    for i in range(len(dir_names_k)):
        output_dir = dir_names_k[i]
        output_dir_c = dir_c_names_k[i]
        mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
        surf_vel = extract_surf_vel(mesh_vel)
        sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
        vi[i] = surf_vel.max()
        Vc[i] = sz_dv.max()
        axes[0].scatter(vi[i]/10, Vc[i]/1e3, c=color_list[k], marker=marker_list[i], edgecolor='k', zorder=10)
        axes[1].scatter(vi[i]/10, Vc[i]/vi[i], c=color_list[k], marker=marker_list[i], edgecolor='k', zorder=10)
    axes[0].plot(vi/10, Vc/1e3, c=color_list[k], ls=ls_list[k], lw=3, label=label_list[k])
    axes[1].plot(vi/10, Vc/vi, c=color_list[k], ls=ls_list[k], lw=3, label=label_list[k])
    axes[0].scatter(vi[1]/10, Vc[1]/1e3, c='gray', marker=marker_list[1], edgecolor='k', zorder=10)
    axes[1].scatter(vi[1]/10, Vc[1]/vi[1], c='gray', marker=marker_list[1], edgecolor='k', zorder=10)
    # print(prefixes[k], vi, Vc/1e3)

axes[0].set_xlabel(r'$v_{pl}$ (cm/yr)')
axes[0].set_ylabel(r'$U_{max}$ (m)')
axes[0].minorticks_on()
axes[0].set_title('(a)', loc='left')

axes[1].set_xlabel(r'$v_{pl}$ (cm/yr)')
axes[1].set_ylabel(r'$T_r$ (year)')
axes[1].minorticks_on()
axes[1].legend()
axes[1].set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('fig_rheo_v_V.pdf', transparent=True)

# %%
k = 0
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

fig = plt.figure(figsize=(13, 6))

gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[4, 9], height_ratios=[1, 1])

ax = fig.add_subplot(gs[:, 0])

fmt = mticker.LogFormatterSciNotation(base=10)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_visc = load_tvisc(output_dir, 19, yres, arg_mesh, 'l')
    arg_x = np.where((surf_x > -500) & (surf_x < -100))[0]
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(np.mean(10**mesh_visc[:, 0, arg_x, 0], axis=-1), mesh_loc[:, 0, 0, -1], lw=3, 
#             label=f'{fmt(case_value):s}', 
            label=f'{case_value:.1f}', 
            c=c_list[i], ls=ls_list[i])

ax.set_xscale('log')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.set_ylabel(r'$z$ (km)')
ax.legend(loc='center right', fontsize=12, title=r'$\omega_{disl}$', title_fontsize=12)
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(a)', loc='left')

ax.set_ylim(mesh_loc[:, 0, 0, -1].min(), mesh_loc[:, 0, 0, -1].max())

ax = fig.add_subplot(gs[0, 1])

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{fmt(case_value):s}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

ax = fig.add_subplot(gs[1, 1])

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{fmt(case_value):s}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(c)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_wdisl.pdf', transparent=True)

# %%
k = 5
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

fig = plt.figure(figsize=(9, 9))

ax = fig.add_subplot(311)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'i')
    sz_meanst = extract_sz_var_mean(sz_loc, sz_st, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_meanst, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$\tau_{II}$ (MPa)')
# ax.legend(loc='upper right', fontsize=12, title=label_list[k])
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$\mu_{s}$', title_fontsize=12)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(312)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

ax = fig.add_subplot(313)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(c)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_fst.pdf', transparent=True)

# %%
k = 2
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(211)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value/1e9:.0f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')

ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(212)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value/1e9:.0f}', c=c_list[i], ls=ls_list[i])
    
ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$G_{b}$ (GPa)', title_fontsize=12)
ax.set_title('(b)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_Gm.pdf', transparent=True)

# %%
k = 3
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(211)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value/1e9:.0f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')

ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(212)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value/1e9:.0f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$G_{sz}$ (GPa)', title_fontsize=12)
ax.set_title('(b)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_Gsz.pdf', transparent=True)

# %%
k = 4
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)

c_list = ['C3', 'C1', 'C0']
ls_list = ['--', '-', ':']

fig = plt.figure(figsize=(9, 9))

ax = fig.add_subplot(311)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_meanst = extract_sz_var_mean(sz_loc, sz_st, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_meanst, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$\tau_{II}$ (MPa)')
# ax.legend(loc='upper right', fontsize=12, title=label_list[k])
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$\mu_{d,mid}$', title_fontsize=12)
ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(312)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(b)', loc='left')

ax = fig.add_subplot(313)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(c)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_fdynmid.pdf', transparent=True)

# %%
k = 1
dir_names_k = sorted(
    [d for d in dir_names
    if Path(d).parents[0].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[0].name.split('_')[1])
)
dir_c_names_k = sorted(
    [d for d in dir_c_names
    if Path(d).parents[1].name.startswith(prefixes[k])],
    key=lambda d: float(Path(d).parents[1].name.split('_')[1])
)


fig = plt.figure(figsize=(9, 6))

ax = fig.add_subplot(211)

for i in range(len(dir_names_k)):
    output_dir = dir_names_k[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(surf_x, surf_vx/10, lw=3, label=f'{case_value:.1f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (cm/yr)', c='k')

ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(212)

for i in range(len(dir_c_names_k)):
    output_dir_c = dir_c_names_k[i]
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    case_value = case_value = float(Path(dir_names_k[i]).parents[0].name.split('_')[1])
    ax.plot(sz_zbins[1:], sz_dv/1e3, lw=3, label=f'{case_value/1e6:.0f}', c=c_list[i], ls=ls_list[i])

ax.set_xlim(sz_zbins[:-1].max(), sz_zbins[:-1].min())
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)')
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
ax.legend(loc='upper right', fontsize=12, title=r'$\tau_{y,max}$ (MPa)', title_fontsize=12)
ax.set_title('(b)', loc='left')

ax.set_xlim(sz_zbins[1:].max(), sz_zbins[1:].min())

fig.tight_layout()
plt.savefig('fig_ymax.pdf', transparent=True)

# %%


# %%
txt = np.loadtxt('../geodata/slab_top_data.txt')
xx = txt[::-1, 0]
zz = txt[::-1, 1]
plt.plot(xx, zz)
arg_1 = np.argmin(np.abs(zz+5))
arg_2 = np.argmin(np.abs(zz+70))
plt.scatter([xx[arg_1], xx[arg_2]], [zz[arg_1], zz[arg_2]])
np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], xx[arg_2]-xx[arg_1]))

# %%
yres = 0
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
mesh_loc, arg_mesh = load_tmesh(dir_all[0], yres)
surf_x = extract_surf_loc(mesh_loc)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(b)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(a)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

plt.tight_layout()
plt.savefig('fig_theta_Ai.pdf', transparent=True)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(b)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[0, 0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(a)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[1, 1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/surf_vx_l_all, c='r', lw=3, marker='s')

ax.set_title('(d)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$T_r$ (yr)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[1, 0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
# p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
# ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/surf_vx_l_all, c='r', lw=3, marker='s')

ax.set_title('(c)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$T_r$ (yr)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

plt.tight_layout()
plt.savefig('fig_theta_Ai_Tr.pdf', transparent=True)

# %%
yres = 2
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
mesh_loc, arg_mesh = load_tmesh(dir_all[0], yres)
surf_x = extract_surf_loc(mesh_loc)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(b)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(a)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

plt.tight_layout()
plt.savefig('fig_theta_Ai.pdf', transparent=True)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(b)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[0, 0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/1e3, c='r', lw=3, marker='s')

ax.set_title('(a)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$U_{max}$ (m)', c='r')
ax.minorticks_on()
axt.minorticks_on()
axt.set_ylim(10, 35)

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[1, 1]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/gfactor_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/gfactor_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*dir_values))
ax.plot(theta, surf_vx_l_all/10, c='b', lw=3, marker='o')
axt = ax.twinx()
axt.plot(theta, sz_dv_c_all/surf_vx_l_all, c='r', lw=3, marker='s')

ax.set_title('(d)', loc='left')
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$T_r$ (yr)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

ax = axes[1, 0]
dir_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_8.19e+07/wpa_*/data/'),
              key=lambda d: float(Path(d).parents[0].name.split('_')[1])
          )
dir_values = np.zeros(len(dir_all))
surf_vx_l_all = np.zeros(len(dir_all))
sz_dv_c_all = np.zeros(len(dir_all))
dir_c_all = sorted(
              glob('/home/x-jqfang/scratch/model_2610/L_5.12e+06/wpa_*/restart/data/'),
              key=lambda d: float(Path(d).parents[1].name.split('_')[1])
          )

for i in range(len(dir_all)):
    output_dir = dir_all[i]
    output_dir_c = dir_c_all[i]
    mesh_vel = load_tvel(output_dir, 19, yres, arg_mesh, 'l')
    surf_vx = extract_surf_vel(mesh_vel)
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir_c, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)
    dir_values[i] = float(Path(output_dir).parents[0].name.split('_')[1])
    surf_vx_l_all[i] = surf_vx.max()
    sz_dv_c_all[i] = sz_dv.max()

ax.set_title('(c)', loc='left')
ax.plot(dir_values/1e6, surf_vx_l_all/10, c='b', lw=3, marker='o')
# p = np.polyfit(np.sqrt(dir_values[:3]/1e6), surf_vx_l_all[:3]/10, 1)
# ax.plot(dir_values/1e6, np.sqrt(dir_values/1e6)*p[0]+p[1], 'b--')
axt = ax.twinx()
axt.plot(dir_values/1e6, sz_dv_c_all/surf_vx_l_all, c='r', lw=3, marker='s')

ax.set_title('(a)', loc='left')
ax.set_xlabel(r'$A_i$ (Myr)')
ax.set_ylabel(r'$v_{pl}$ (cm/yr)', c='b')
axt.set_ylabel(r'$T_r$ (yr)', c='r')
ax.minorticks_on()
axt.minorticks_on()

axt.spines['left'].set_color('b')
axt.spines['right'].set_color('r')
ax.tick_params(axis='y', colors='b')
axt.tick_params(axis='y', colors='r')

plt.tight_layout()
plt.savefig('fig_theta_Ai_Tr.pdf', transparent=True)

# %%


