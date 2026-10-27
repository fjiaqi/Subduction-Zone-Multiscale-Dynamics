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
file_path = "./2017017_esupp_Table_S1.html"
tables = pd.read_html(file_path)
df = tables[0]
df = df[df['S'].eq('*')].copy()
df.loc[:, 'event_id'] = df['evTAG'].str.slice(0, -6)
grouped = df.groupby("event_id")

# Ensure numeric conversion
df["Mw"] = pd.to_numeric(df["Mw"], errors="coerce")
df["L (km)"] = pd.to_numeric(df["L (km)"], errors="coerce")
df["W (km)"] = pd.to_numeric(df["W (km)"], errors="coerce")
df["D (m)"] = pd.to_numeric(df["D (m)"], errors="coerce")

# Group by event_id and calculate median, min, max
stats = df.groupby("event_id").agg(
    Mw_med=("Mw", "median"),
    Mw_min=("Mw", "min"),
    Mw_max=("Mw", "max"),
    L_med=("L (km)", "median"),
    L_min=("L (km)", "min"),
    L_max=("L (km)", "max"),
    W_med=("W (km)", "median"),
    W_min=("W (km)", "min"),
    W_max=("W (km)", "max"),
    D_med=("D (m)", "median"),
    D_min=("D (m)", "min"),
    D_max=("D (m)", "max")
).reset_index()

# Calculate error ranges
Mw_err = np.vstack([stats["Mw_med"] - stats["Mw_min"], stats["Mw_max"] - stats["Mw_med"]])
L_err = np.vstack([stats["L_med"] - stats["L_min"], stats["L_max"] - stats["L_med"]])
W_err = np.vstack([stats["W_med"] - stats["W_min"], stats["W_max"] - stats["W_med"]])
D_err = np.vstack([stats["D_med"] - stats["D_min"], stats["D_max"] - stats["D_med"]])

df

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

Z_list = ['6.00e+04', '3.00e+04', '2.50e+04', '2.00e+04', '1.50e+04', '1.00e+04']
Y_list = ['8.19e+07', '2.05e+07', '5.12e+06', '1.28e+06', '3.20e+05', '8.00e+04']
res_L = np.array([2*np.sqrt(float(Y_list[j])*5e3) for j in range(len(Y_list))])/1e3
res_dv = np.zeros((len(Z_list), len(Y_list), sz_dv.shape[0]))
res_W = np.zeros((len(Z_list), len(Y_list)))
res_D = np.zeros((len(Z_list), len(Y_list)))
res_M = np.zeros((len(Z_list), len(Y_list)))

for i in range(len(Z_list)):
    for j in range(len(Y_list)):
        output_dir = f'/home/x-jqfang/scratch/model_2610/L_{Y_list[j]}/Z_{Z_list[i]}/restart/data/'
        
        try:
            sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
        except:
            res_dv[i, j] = np.nan
            res_W[i, j] = np.nan
            res_D[i, j] = np.nan
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
        res_D[i, j] = 1/2 * np.sum(ds[arg_s] * (dv_zz[arg_s]+dv_zz[arg_s+1])/2) / np.sum(ds[arg_s])
        res_M[i, j] = np.pi/4 * 30e9 * np.sum(ds[arg_s]*1e3 * (dv_zz[arg_s]+dv_zz[arg_s+1])/2/1e3) * res_L[j]*1e3

# %%
fig, ax = plt.subplots(figsize=(12, 4))

dt = 1
color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Z_list))[::-1])

for i in range(len(Z_list)):
    ax.plot(sz_zbins[1:], res_dv[i, 0]*dt/1e3, lw=3,
            label=f'{res_W[i, 2]:.0f}', c=color_list[i])

ax.set_xlim(0, -75)
ax.set_xlabel(r'$z$ (km)')
ax.set_ylabel(r'$U$ (m)', c='k')
ax.legend(loc='upper right', title=r'$W$ (km)', ncol=2)
ax.tick_params(axis='y', labelcolor='k')
ax.minorticks_on()
# ax.set_title('(a)', loc='left')

fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Z_list))[::-1])

for i in range(len(Z_list)):
    ax.plot(
        res_L, np.max(res_dv[i], axis=-1)/1e3,
        # res_L, res_D[i]/1e3,
        c=color_list[i], marker='o', markersize=5, 
        label=f'{res_W[i].max():.0f}', lw=3, zorder=10
    )

ax.minorticks_on()
ax.set_xlabel(r'$L$ (km)')
ax.set_ylabel(r'$U_{max}$ (m)')
ax.set_xscale('log')
ax.legend(title=r'$W$ (km)', loc='best')
ax.set_xlim(10, 3000)

plt.tight_layout()
plt.savefig('fig_U_L_W.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Z_list))[::-1])

for i in range(len(Z_list)):
    if(i < 3):
        ax.plot(
            res_L[:-2], 2/3 * (np.log10(res_M[i, :-2]) - 9.1),
            c=color_list[i], marker='o', markersize=5, 
            label=f'{res_W[i].max():.0f}', lw=3, zorder=10
        )
        ax.plot(
            res_L[-3:], 2/3 * (np.log10(res_M[i, -3:]) - 9.1),
            c=color_list[i], marker='o', markersize=3,
            ls='--', lw=1, zorder=10
        )
    elif(i < 4):
        ax.plot(
            res_L[:-1], 2/3 * (np.log10(res_M[i, :-1]) - 9.1),
            c=color_list[i], marker='o', markersize=5, 
            label=f'{res_W[i].max():.0f}', lw=3, zorder=10
        )
        ax.plot(
            res_L[-2:], 2/3 * (np.log10(res_M[i, -2:]) - 9.1),
            c=color_list[i], marker='o', markersize=3,
            ls='--', lw=1, zorder=10
        )
    else:
        ax.plot(
            res_L, 2/3 * (np.log10(res_M[i]) - 9.1),
            c=color_list[i], marker='o', markersize=5, 
            label=f'{res_W[i].max():.0f}', lw=3, zorder=10
        )

ax.errorbar(
    stats["L_med"], stats["Mw_med"],
    xerr=L_err, yerr=Mw_err, color='gray',
    fmt='o', ecolor='gray', alpha=0.7, capsize=3
)

ax.minorticks_on()
ax.set_xlabel(r'${L}$ (km)')
ax.set_ylabel('$M_w$')
ax.set_xscale('log')
ax.legend(title=r'$W$ (km)', loc='best')
ax.set_xlim(10, 3000)

plt.tight_layout()
plt.savefig('fig_Mw_L_W.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Y_list))[::-1])

for j in range(len(Y_list)):
    ax.plot(
        res_W[:, j], 2/3 * (np.log10(res_M[:, j]) - 9.1),
        c=color_list[j], marker='o', markersize=5, 
        label=f'{res_L[j]:.0f}', lw=3, zorder=10
    )

ax.errorbar(
    stats["W_med"], stats["Mw_med"],
    xerr=W_err, yerr=Mw_err, color='gray',
    fmt='o', ecolor='gray', alpha=0.7, capsize=3
)


ax.minorticks_on()
ax.set_xlabel(r'${W}$ (km)')
ax.set_ylabel('$M_w$')
ax.set_xscale('log')
ax.legend(title=r'$L$ (km)', loc='best')
# ax.set_xlim(10, 3000)

plt.tight_layout()
plt.savefig('fig_Mw_W_L.pdf', transparent=True)

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Z_list))[::-1])

for i in range(len(Z_list)):
    axes[0].plot(
        res_L, np.max(res_dv[i], axis=-1)/1e3,
        c=color_list[i], marker='o', markersize=5, 
        label=f'{res_W[i].max():.0f}', lw=3, zorder=10
    )
    axes[1].plot(
        res_L, 2/3 * (np.log10(res_M[i]) - 9.1),
        c=color_list[i], marker='o', markersize=5, 
        label=f'{res_W[i].max():.0f}', lw=3, zorder=10
    )

axes[0].minorticks_on()
# axes[0].set_xlabel(r'$L$ (km)')
axes[0].set_ylabel(r'$U_{max}$ (m)')
axes[0].set_xscale('log')
axes[0].legend(title=r'$W$ (km)', loc='best')
axes[0].set_xlim(10, 3000)

axes[1].errorbar(
    stats["L_med"], stats["Mw_med"],
    xerr=L_err, yerr=Mw_err, color='gray',
    fmt='o', ecolor='gray', alpha=0.7, capsize=3
)


axes[1].minorticks_on()
axes[1].set_xlabel(r'${L}$ (km)')
axes[1].set_ylabel('$M_w$')

plt.tight_layout()
plt.savefig('fig_U_Mw_L_W.pdf', transparent=True)

# %%


# %%
df_sub = pd.read_csv("all_transects.csv")

cnt = 0
seg_id = np.zeros(len(df_sub))
for i in range(len(df_sub)-1):
    if(np.isclose(df_sub['theta1'].iloc[i], df_sub['theta1'].iloc[i+1])):
        seg_id[i] = cnt
        seg_id[i+1] = cnt
    else:
        cnt += 1
        seg_id[i+1] = cnt

seg_theta_min = np.zeros(len(df_sub))
seg_theta_max = np.zeros(len(df_sub))
for i in range(len(df_sub)):
    seg_theta_min[i] = df_sub['theta'][(seg_id == seg_id[i]) & (df_sub['theta'] > 0)].min()
    seg_theta_max[i] = df_sub['theta'][(seg_id == seg_id[i]) & (df_sub['theta'] > 0)].max()

cnt = 0
seg_id = np.zeros(len(df_sub))
for i in range(len(df_sub)-1):
    if(np.isclose(df_sub['W1'].iloc[i], df_sub['W1'].iloc[i+1])):
        seg_id[i] = cnt
        seg_id[i+1] = cnt
    else:
        cnt += 1
        seg_id[i+1] = cnt

seg_W_min = np.zeros(len(df_sub))
seg_W_max = np.zeros(len(df_sub))
for i in range(len(df_sub)):
    seg_W_min[i] = df_sub['W'][(seg_id == seg_id[i]) & (df_sub['W'] > 0)].min()
    seg_W_max[i] = df_sub['W'][(seg_id == seg_id[i]) & (df_sub['W'] > 0)].max()

# %%
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]
sz_dv = np.zeros((len(sz_zbins)-1, 3))

param_name = 'gfactor'
param_list = ['1.50e+00', '1.25e+00', '1.00e+00', '7.50e-01', '5.00e-01', '4.00e-01', '3.00e-01', '2.50e-01']
Y_list = ['8.19e+07', '2.05e+07', '5.12e+06', '1.28e+06', '3.20e+05', '8.00e+04']
arg_1 = np.argmin(np.abs(zz+5))
arg_2 = np.argmin(np.abs(zz+70))
res_theta = np.rad2deg(np.arctan2(zz[arg_1]-zz[arg_2], (xx[arg_2]-xx[arg_1])*np.array(param_list).astype(float)))
res_L = np.array([2*np.sqrt(float(Y_list[j])*5e3) for j in range(len(Y_list))])/1e3
res_dv = np.zeros((len(param_list), len(Y_list), sz_dv.shape[0]))
res_W = np.zeros((len(param_list), len(Y_list)))
res_M = np.zeros((len(param_list), len(Y_list)))

for i in range(len(param_list)):
    for j in range(len(Y_list)):
        output_dir = f'/home/x-jqfang/scratch/model_2610/L_{Y_list[j]}/{param_name}_{param_list[i]}/restart/data/'
        
        sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

        res_dv[i, j] = sz_dv
        dv_zz = np.interp(zz, sz_zbins[1:], sz_dv)
        arg_s = np.where((zz >= sz_zbins.min())
                         & (dv_zz > np.minimum(2e3, 0.1*dv_zz.max())) 
                         & (zz <= -5)
                         & (zz >= -70))[0]
        ds = np.sqrt(np.diff(xx*float(param_list[i]))**2+np.diff(zz)**2)
        ss = np.insert(np.cumsum(ds), 0, 0)
        res_W[i, j] = np.ptp(ss[arg_s])
        res_M[i, j] = np.pi/4 * 30e9 * np.sum(ds[arg_s]*1e3 * (dv_zz[arg_s]+dv_zz[arg_s+1])/2/1e3) * res_L[j]*1e3

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(param_list))[::-1])

for i in range(len(param_list)):
    ax.plot(
        res_L, np.max(res_dv[i], axis=-1)/1e3,
        c=color_list[i], marker='o', markersize=5, 
        label=f'{res_theta[i]:.0f}', lw=3, zorder=10
    )

ax.minorticks_on()
ax.set_xlabel(r'$L$ (km)')
ax.set_ylabel(r'$U_{max}$ (m)')
ax.set_xscale('log')
ax.legend(title=r'$\Theta$ ($^\circ$)', loc='best')
ax.set_xlim(10, 3000)

plt.tight_layout()
plt.savefig('fig_U_L_theta.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(param_list))[::-1])

for i in range(len(param_list)):
    ax.plot(
        res_L, np.max(res_dv[i], axis=-1)/1e3,
        c=color_list[i], marker='o', markersize=5, 
        label=f'{res_W[i].max():.0f}', lw=3, zorder=10
    )

ax.minorticks_on()
ax.set_xlabel(r'$L$ (km)')
ax.set_ylabel(r'$U_{max}$ (m)')
ax.set_xscale('log')
ax.legend(title=r'$W_{max}$ (km)', loc='best')
ax.set_xlim(10, 3000)

plt.tight_layout()
plt.savefig('fig_U_L_Wmax.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Y_list))[::-1])

for j in range(len(Y_list)):
    ax.plot(
        res_theta, 2/3 * (np.log10(res_M[:, j]) - 9.1),
        c=color_list[j], marker='o', markersize=5, 
        label=f'{res_L[j]:.0f}', lw=3, zorder=10
    )

ax.errorbar(df_sub['theta1'], df_sub['Mmax1'], 
             xerr=[np.maximum(df_sub['theta1']-seg_theta_min, 0), 
                   np.maximum(seg_theta_max-df_sub['theta1'], 0)], 
             fmt='o', capsize=3, c='gray')
ax.set_xlim(0, 90)
ax.set_ylim(5, 10)

ax.minorticks_on()
ax.set_xlabel(r'$\Theta$ ($^\circ$)')
ax.set_ylabel(r'$M_{w, max}$')
ax.legend(title=r'$L$ (km)', loc='best')

plt.tight_layout()
plt.savefig('fig_Mw_theta_L.pdf', transparent=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Y_list))[::-1])

for j in range(len(Y_list)):
    ax.plot(
        res_W[:, j], 2/3 * (np.log10(res_M[:, j]) - 9.1),
        c=color_list[j], marker='o', markersize=5, 
        label=f'{res_L[j]:.0f}', lw=3, zorder=10
    )

ax.errorbar(df_sub['W1'], df_sub['Mmax1'], 
             xerr=[np.maximum(df_sub['W1']-seg_W_min, 0), 
                   np.maximum(seg_W_max-df_sub['W1'], 0)], 
             fmt='o', capsize=3, c='gray')
ax.set_xlim(0, 300)
ax.set_ylim(5, 10)

ax.minorticks_on()
ax.set_xlabel(r'$W_{max}$ (km)')
ax.set_ylabel(r'$M_{w, max}$')
ax.legend(title=r'$L$ (km)', loc='best')

plt.tight_layout()
plt.savefig('fig_U_Wmax_L.pdf', transparent=True)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

color_list = plt.cm.coolwarm(np.linspace(0, 1, len(Y_list))[::-1])

for j in range(len(Y_list)):
    axes[0].plot(
        res_theta, 2/3 * (np.log10(res_M[:, j]) - 9.1),
        c=color_list[j], marker='o', markersize=5, 
        label=f'{res_L[j]:.0f}', lw=3, zorder=10
    )
    axes[1].plot(
        res_W[:, j], 2/3 * (np.log10(res_M[:, j]) - 9.1),
        c=color_list[j], marker='o', markersize=5, 
        label=f'{res_L[j]:.0f}', lw=3, zorder=10
    )


axes[0].errorbar(df_sub['theta1'], df_sub['Mmax1'], 
             xerr=[np.maximum(df_sub['theta1']-seg_theta_min, 0), 
                   np.maximum(seg_theta_max-df_sub['theta1'], 0)], 
             fmt='o', capsize=3, c='gray')
axes[0].set_xlim(0, 90)
axes[0].set_ylim(5, 10)

axes[0].minorticks_on()
axes[0].set_xlabel(r'$\Theta$ ($^\circ$)')
axes[0].set_ylabel(r'$M_{w, max}$')
axes[0].legend(title=r'$L$ (km)', loc='best')
axes[0].set_title('(a)', loc='left')
    
axes[1].errorbar(df_sub['W1'], df_sub['Mmax1'], 
             xerr=[np.maximum(df_sub['W1']-seg_W_min, 0), 
                   np.maximum(seg_W_max-df_sub['W1'], 0)], 
             fmt='o', capsize=3, c='gray')
axes[1].set_xlim(0, 300)
axes[1].set_ylim(5, 10)

axes[1].minorticks_on()
axes[1].set_xlabel(r'$W_{max}$ (km)')
axes[1].set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('fig_Mw_theta_Wmax_L.pdf', transparent=True)

# %%

# %%
sz_zbins = -np.arange(0, 81, 5)[::-1]
sz_ybins = [-0.01, 0.01]

output_dir = '/home/x-jqfang/scratch/model_2610/fdynmid_3.00e-01/restart/data/'

sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
res_dv_inf = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

Y_list = ['8.19e+07', '2.05e+07', '5.12e+06', '1.28e+06']
res_L = np.array([2*np.sqrt(float(Y_list[j])*5e3) for j in range(len(Y_list))])/1e3
res_dv = np.zeros((len(Y_list), len(sz_zbins)-1))

for j in range(len(Y_list)):
    output_dir = f'/home/x-jqfang/scratch/model_2610/L_{Y_list[j]}/Z_6.00e+04/restart/data/'
    
    sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c')
    sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

    res_dv[j] = sz_dv

L1_list = [['0.125', '0.25', '0.5', '1.0'], 
           ['0.0625', '0.125', '0.25', '0.5'],
           ['0.0', '0.0', '0.0', '0.0']]
L2_list = ['0.125', '0.25', '0.5', '1.0']
res_L_3d = np.array([1280*float(L2_list[j]) for j in range(len(L2_list))])
res_dv_3d = np.zeros((len(L1_list), len(L2_list), len(sz_zbins)-1))

for i in tqdm(range(len(L1_list))):
    for j in range(len(L2_list)):
        output_dir = glob(f'/home/x-jqfang/scratch/model_2610/3d/3d_{L1_list[i][j]}_{L2_list[j]}/restart_*/data/')[0]
        basic_dir = f'/home/x-jqfang/scratch/model_2610/3d/3d_1.1_1.1/data/'
        
        sz_loc, sz_vel, sz_sr, sz_st, sz_visc = load_sz_args(output_dir, 0, 'c', basic_dir=basic_dir)
        sz_dv = extract_sz_dv(sz_loc, sz_vel, sz_ybins, sz_zbins)

        res_dv_3d[i, j] = sz_dv

# %%
fig = plt.figure(figsize=(10, 4))

gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[3, 2])

ax = fig.add_subplot(gs[0])

ax.axhline(res_dv_inf.max()/1e3, ls='--', color='k', lw=2)

ax.plot(
    res_L, np.max(res_dv, axis=-1)/1e3,
    marker='o', markersize=5, 
    ls='-.', lw=2, zorder=10,
    c='C7', label='2.5D'
)

for i in range(len(L1_list)):
    ax.plot(
        res_L_3d, np.max(res_dv_3d[i], axis=-1)/1e3,
        marker='x', markersize=5, 
        ls='-', lw=2, zorder=5,
        c=['r', 'm', 'b'][i],
        label=['3D rectangle', '3D trapezoid', '3D triangle'][i]
    )

ax.minorticks_on()
ax.set_xlabel(r'$L$ (km)')
ax.set_ylabel(r'$U_{max}$ (m)')
ax.set_xscale('log')
ax.legend()
ax.set_title('(a)', loc='left')
ax.minorticks_on()
ax.set_xlim(100, 3000)
ax.set_ylim(0, 30)

ax = fig.add_subplot(gs[1])

ax.plot([0, 2, 2, 4], [1, 1, 0, 0], lw=2, c='r', label='Rectangle')
ax.plot([0, 1, 2, 4], [1, 1, 0, 0], lw=2, c='m', label='Trapezoid')
ax.plot([0, 2, 4], [1, 0, 0], lw=2, c='b', label='Triangle')
ax.axhline(0, lw=1, color='k', ls='--')
# ax.legend()
ax.set_xlim(0, 3)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks([0, 2], ['0', r'${L}/2$'])
ax.set_yticks([0, 1], ['0', r'$\Delta \tau$'])
ax.set_xlabel(r'$y$')
ax.set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('fig_D_cmp.pdf', transparent=True)

# %%


