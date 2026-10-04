import logging
import underworld as uw
import underworld.function as fn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d
import shapely.geometry as shg
from shapely import ops
from mpi4py import MPI
from pathlib import Path
import csv
import sys

plt.rcParams['font.size'] = 16

###################################################################################################################################################

param_fdynmid = 0.3
param_fst = 0.5
param_fdynup = 0.7
param_fstic = param_fst*2-param_fdynmid
param_etamax = 1e24
param_ymax = 200e6
param_wdisl = 1.0
param_Gm = 60e9
param_Gsz = 30e9
param_Cm = 100e6
param_Csz = 10e6
param_wpa = 50e6
param_epa = 80e6
param_gfactor = 1.0
param_Y = 81920e3
param_Z = 60e3

param_idx = int(sys.argv[1])
param_value = float(sys.argv[2])

param_list = ['fdynmid', 'fdynup', 'fst', 'etamax', 'ymax', 
              'wdisl', 'Gm', 'Gsz', 'Cm', 'Csz',
              'wpa', 'epa', 'gfactor', 'Y', 'Z']
param_name = param_list[param_idx]
locals()[f'param_{param_name}'] = param_value

output_dir_parent = '/home1/08524/jqfang/scratch/newnew_2607'

##################################################################################################################################################

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

##################################################################################################################################################

age_west_plate = param_wpa * s_per_year / reference_time
age_east_plate = param_epa * s_per_year / reference_time
age_max_cut = 140e6 * s_per_year / reference_time
vel_west_plate = 5e-2 / s_per_year / reference_velocity

shear_zone_width = 6e3 / reference_length
ridge_width = 40e3 / reference_length
ridge_loc = -age_west_plate*vel_west_plate
slab_top_max_depth = 400e3 / reference_length
west_crust_depth = 6e3 / reference_length
east_crust_depth = 20e3 / reference_length

domain_width_west = -ridge_loc + ridge_width
domain_width_east = 1500e3 / reference_length
domain_width = domain_width_east + domain_width_west
domain_depth = 660e3 / reference_length
lmantle_top_depth = 660e3 / reference_length
air_depth = 0. / reference_length
domain_length = 5e3 / reference_length

###################################################################################################################################################

z_arr = np.linspace(-domain_depth*1.2, domain_depth*1e-20, 501)
t_arr = np.linspace(0, np.maximum(age_west_plate, age_east_plate)*1.2, 101)
T_arr = np.zeros((len(z_arr), len(t_arr)))
for i in range(len(t_arr)):
    if(t_arr[i] > 0):
        # T_arr[:, i] = erf(-z_arr*reference_length/2./np.sqrt(reference_diffusivity*tz * reference_time))
        T_arr[:, i] = erf(-z_arr/2./np.sqrt(t_arr[i]))
    else:
        T_arr[:, i] = 1.
        T_arr[-1, i] = 0.

zz_plate, tt_plate = np.meshgrid(z_arr, t_arr)
plate_temp_func = LinearNDInterpolator((zz_plate.ravel(), tt_plate.ravel()), T_arr.T.ravel(), fill_value=1)


surface_coords = np.array([[-domain_width_west, 0.], [domain_width_east, 0.]])
surface_line = shg.LineString(surface_coords)
top_coords = np.array([[-domain_width_west, air_depth], [domain_width_east, air_depth]])
top_line = shg.LineString(top_coords)
bottom_coords = np.array([[-domain_width_west, -domain_depth], [domain_width_east, -domain_depth]])
bottom_line = shg.LineString(bottom_coords)
west_wall_coords = np.array([[-domain_width_west, -domain_depth], [-domain_width_west, air_depth]])
west_wall_line = shg.LineString(west_wall_coords)
east_wall_coords = np.array([[domain_width_east, -domain_depth], [domain_width_east, air_depth]])
east_wall_line = shg.LineString(east_wall_coords)

n_interp = 101
west_lithos_bottom_coords = np.zeros((n_interp, 2))
west_lithos_bottom_coords[:, 0] = np.linspace(ridge_loc, 0, n_interp)
lithos_threshold = 0.9
z_arr_new = np.linspace(z_arr[np.where(T_arr[:, -1] <= lithos_threshold)[0][0]], 0, 1001)
for i in range(1, n_interp):
    xp = west_lithos_bottom_coords[i, 0]
    tz = plate_temp_func((z_arr_new, np.repeat((xp-ridge_loc)/vel_west_plate, len(z_arr_new))))
    west_lithos_bottom_coords[i, 1] = z_arr_new[np.where(tz <= lithos_threshold)[0][0]]
west_lithos_bottom_xzfunc_tmp = interp1d(west_lithos_bottom_coords[:, 0], west_lithos_bottom_coords[:, 1], 
                                       fill_value='extrapolate')

slab_top_coords = np.genfromtxt('./geodata/slab_top_data.txt')[::-1, :]*1e3/reference_length
slab_top_coords[:, 0] *= param_gfactor
slab_top_coords[:, 1] = slab_top_coords[:, 1] - slab_top_coords[:, 1].max()
slab_top_down_slope = np.polyfit(slab_top_coords[-5:, 0], slab_top_coords[-5:, 1], 1)[0]
if(-slab_top_max_depth < np.min(slab_top_coords[:, 1])):
    slab_top_coords = np.vstack((slab_top_coords, 
                                 np.array([slab_top_coords[-1, 0]-(slab_top_coords[-1, 1]+slab_top_max_depth)/slab_top_down_slope, 
                                           -slab_top_max_depth])))
slab_top_line = shg.LineString(slab_top_coords)
slab_top_xzfunc = interp1d(slab_top_coords[:, 0], slab_top_coords[:, 1], fill_value='extrapolate')
slab_top_zxfunc = interp1d(slab_top_coords[:, 1], slab_top_coords[:, 0], fill_value='extrapolate')

shear_zone_bottom_line = slab_top_line.offset_curve(-shear_zone_width)
shear_zone_bottom_coords = np.array(shear_zone_bottom_line.coords)
shear_zone_bottom_up_slope = np.polyfit(shear_zone_bottom_coords[:2, 0], shear_zone_bottom_coords[:2, 1], 1)[0]
shear_zone_bottom_coords = np.vstack((np.array([shear_zone_bottom_coords[0, 0]
                                                -shear_zone_bottom_coords[0, 1]/shear_zone_bottom_up_slope, 0]), 
                                      shear_zone_bottom_coords))
shear_zone_bottom_line = shg.LineString(shear_zone_bottom_coords)
shear_zone_bottom_xzfunc = interp1d(shear_zone_bottom_coords[:, 0], shear_zone_bottom_coords[:, 1], fill_value='extrapolate')
shear_zone_bottom_zxfunc = interp1d(shear_zone_bottom_coords[:, 1], shear_zone_bottom_coords[:, 0], fill_value='extrapolate')

slab_bottom_line = slab_top_line.offset_curve(west_lithos_bottom_coords[-1, 1])
slab_bottom_coords = np.array(slab_bottom_line.coords)
slab_bottom_xzfunc_tmp = interp1d(slab_bottom_coords[:, 0], slab_bottom_coords[:, 1], fill_value='extrapolate')

n_interp = 9
joint_coords = np.zeros((n_interp, 2))
joint_coords[:, 0] = np.linspace(slab_bottom_coords[:, 0].min(), 0, n_interp+2)[1:-1]
for i in range(n_interp):
    xp = joint_coords[i, 0]
    joint_coords[i, 1] = (slab_bottom_xzfunc_tmp(joint_coords[i, 0]) * (xp-slab_bottom_coords[:, 0].min())
                          + west_lithos_bottom_xzfunc_tmp(joint_coords[i, 0]) * (0-xp)) / (0-slab_bottom_coords[:, 0].min())

west_lithos_bottom_coords = west_lithos_bottom_coords[west_lithos_bottom_coords[:, 0] <= slab_bottom_coords[:, 0].min()]
slab_bottom_coords = slab_bottom_coords[slab_bottom_coords[:, 0] >= 0]
west_lithos_bottom_no_slab_coords = np.vstack((west_lithos_bottom_coords, joint_coords))
west_lithos_bottom_coords = np.vstack((west_lithos_bottom_coords, joint_coords, slab_bottom_coords))
west_lithos_bottom_line = shg.LineString(west_lithos_bottom_coords)
west_lithos_bottom_xzfunc = interp1d(west_lithos_bottom_coords[:, 0], west_lithos_bottom_coords[:, 1], fill_value='extrapolate')

slab_front_coords = np.array([[west_lithos_bottom_coords[-1, 0], west_lithos_bottom_coords[-1, 1]],
                              [slab_top_coords[-1, 0], slab_top_coords[-1, 1]]])
slab_front_slope_rad = np.arctan((slab_front_coords[1, 1] - slab_front_coords[0, 1])
                                 / (slab_front_coords[1, 0] - slab_front_coords[0, 0]))
slab_front_line = shg.LineString(slab_front_coords)
slab_front_xzfunc = interp1d(slab_front_coords[:, 0], slab_front_coords[:, 1], fill_value='extrapolate')

east_lithos_bottom_coords = np.zeros((2, 2))
tz = plate_temp_func((z_arr_new, np.repeat(age_east_plate, len(z_arr_new))))
east_lithos_bottom_coords[:, 0] = np.array([0, domain_width_east])
east_lithos_bottom_coords[:, 1] = z_arr_new[np.where(tz <= lithos_threshold)[0][0]]
east_lithos_bottom_line = shg.LineString(east_lithos_bottom_coords)
pt_tmp = east_lithos_bottom_line.intersection(slab_top_line)
if(pt_tmp.intersects(east_lithos_bottom_line)):
    east_lithos_bottom_line = ops.split(east_lithos_bottom_line, pt_tmp).geoms[-1]
else:
    east_lithos_bottom_line = ops.split(east_lithos_bottom_line, 
                                        pt_tmp.buffer(pt_tmp.distance(east_lithos_bottom_line))).geoms[-1]
east_lithos_bottom_coords = np.array(east_lithos_bottom_line.coords)
east_lithos_bottom_xzfunc = interp1d(east_lithos_bottom_coords[:, 0], east_lithos_bottom_coords[:, 1], 
                                   fill_value='extrapolate')


x_arr_min = 3.*z_arr[np.where(T_arr[:, -1] > 0.95)[0][-1]]
x_arr_max = -2.*z_arr[np.where(T_arr[:, -1] > 0.95)[0][-1]]
x_arr = np.linspace(x_arr_min, x_arr_max, 501)
u0 = -1. + plate_temp_func((x_arr, np.repeat(age_west_plate, len(x_arr))))
u0[x_arr > 0] = -1.

t_arr = np.linspace(0, slab_top_line.length/vel_west_plate*1.2, 101)
u_arr = np.zeros((len(x_arr), len(t_arr)))

kappa = 1.

u_arr[:, 0] = 1+u0
for i in range(1, len(t_arr)):
    pt_tmp = slab_top_line.interpolate(t_arr[i]*vel_west_plate)
    u_east = -1.+plate_temp_func([pt_tmp.xy[1][0], age_east_plate])
    u0[x_arr > 0] = u_east
    ug = gaussian_filter1d(u0, np.sqrt(2.*kappa*t_arr[i])/(x_arr[1]-x_arr[0]), axis=0)
    u_arr[:, i] = 1+ug

t_arr[0] = -(t_arr[1]-t_arr[0])*1e-3 # Avoid interpolation error for small time

xx_slab, tt_slab = np.meshgrid(x_arr, t_arr)
slab_temp_func = LinearNDInterpolator((xx_slab.ravel(), tt_slab.ravel()), u_arr.T.ravel(), fill_value=1)

####################################################################################################################################################

def deform_base_func(x, left, right, buffer):
    x = np.array(x, ndmin=1)
    f_x = np.zeros(len(x))
    half_buffer = buffer/2. 
    for i in range(len(x)):
        if(x[i] < left-half_buffer):
            f_x[i] = x[i]
        elif(x[i] < left+half_buffer):
            f_x[i] = buffer/np.pi * np.sin(np.pi/buffer*(x[i]-(left-half_buffer))) + left-half_buffer
        elif(x[i] < right-half_buffer):
            f_x[i] = 2.*left - x[i]
        elif(x[i] < right+half_buffer):
            f_x[i] = -buffer/np.pi * np.sin(np.pi/buffer*(x[i]-(right-half_buffer))) + 2.*left-(right-half_buffer)
        else:
            f_x[i] = x[i] + 2.*(left-right)
    
    return f_x

def left_right_forward_map(left, right, coef):
    f_left = (left*coef+left)/(coef+2.*(left-right)*coef+1.)
    f_right = ((2.*left-right)*coef+right)/(coef+2.*(left-right)*coef+1.)
    return f_left, f_right

def left_right_backward_map(left, right, coef):
    A = np.array([[coef+1.-2.*left*coef, 2.*left*coef],
                  [2.*coef-2.*right*coef, 2.*right*coef-coef+1]])
    b = np.array([[left+left*coef], [right+right*coef]])
    sol = solve(A, b)
    return sol[0, 0], sol[1, 0]

def deform_map(x, xmin, xmax, left, right, buffer, ratio):
    xx = (x - xmin) / (xmax - xmin)
    xleft = (left - xmin) / (xmax - xmin)
    xright = (right - xmin) / (xmax - xmin)
    xbuffer = buffer / (xmax - xmin)
    xcoef = (ratio - 1.) / (ratio + 1.)
    xleft, xright = left_right_backward_map(xleft, xright, xcoef)
    x_deform = xx + deform_base_func(xx, xleft, xright, xbuffer) * xcoef
    xmin_deform = 0. + deform_base_func(0., xleft, xright, xbuffer)[0] * xcoef
    xmax_deform = 1. + deform_base_func(1., xleft, xright, xbuffer)[0] * xcoef
    x_deform = (x_deform - xmin_deform) / xmax_deform
    x_deform = x_deform * (xmax - xmin) + xmin

    return x_deform

#####################################################################################################################################################

def getGlobalSwarmVar(swarmdata):
    if(uw.mpi.size == 1):
        return swarmdata
    datalenmax = np.array(0, 'i')
    localsize = np.array(swarmdata.shape[0], 'i')
    MPI.COMM_WORLD.Allreduce(localsize, datalenmax, op=MPI.MAX)
    datadimmax = np.array(0, 'i')
    localdim = np.array(swarmdata.shape[1], 'i')
    MPI.COMM_WORLD.Allreduce(localdim, datadimmax, op=MPI.MAX)
    datalength = int(datalenmax * uw.mpi.size)

    globalswarm = np.zeros([datalength, datadimmax])
    dataout = np.empty(datalength)

    for i in range(datadimmax):
        datain = np.full([1, datalenmax], np.nan)[0]
        if(i < localdim):
            datain[:localsize] = swarmdata[:localsize, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalswarm[:, i] = dataout

    uw.mpi.barrier()
    if(uw.mpi.rank == 0):
        globalswarm = globalswarm[~np.isnan(globalswarm)]
        globalswarm = globalswarm.reshape(int(globalswarm.size/datadimmax), datadimmax)

    return globalswarm

def getGlobalMeshVar(mesh, fielddata):
    if(uw.mpi.size == 1):
        return mesh.data, fielddata
    datalength = np.array(0, 'd')
    datalenmax = np.array(0, 'd')
    localsize = np.array(len(mesh.data[:,0]), 'd')
    MPI.COMM_WORLD.Allreduce(localsize, datalenmax, op=MPI.MAX)
    datalenmax = int(datalenmax)
    datalength = int(datalenmax * uw.mpi.size)
    globalmesh = np.zeros([datalength, mesh.dim])
    globalfield = np.zeros([datalength, fielddata.shape[1]])
    dataout= np.empty(datalength)

    localid = range(mesh.nodesLocal)
    for i in range(mesh.dim):
        datain = np.full([1, datalenmax], np.nan)[0]
        datain[localid] = mesh.data[localid, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalmesh[:, i] = dataout
    for i in range(fielddata.shape[1]):
        datain = np.full([1, datalenmax], np.nan)[0]
        datain[localid] = fielddata[localid, i]
        MPI.COMM_WORLD.Gather(datain, dataout, root=0)
        globalfield[:, i] = dataout

    uw.mpi.barrier()
    if(uw.mpi.rank == 0):
        globalmesh = globalmesh[~np.isnan(globalmesh)]
        globalfield = globalfield[~np.isnan(globalfield)]
        globalmesh=globalmesh.reshape(int(globalmesh.size/mesh.dim), mesh.dim)
        globalfield = globalfield.reshape(int(globalfield.size/fielddata.shape[1]), fielddata.shape[1])
        if(mesh.dim == 2):
            globalfield = globalfield[np.lexsort([globalmesh[:, 0], globalmesh[:, 1]]), :]
            globalmesh = globalmesh[np.lexsort([globalmesh[:, 0], globalmesh[:, 1]]), :]    
        else:
            globalfield = globalfield[np.lexsort([globalmesh[:, 0], globalmesh[:, 1], globalmesh[:, 2]]), :]
            globalmesh = globalmesh[np.lexsort([globalmesh[:, 0], globalmesh[:, 1], globalmesh[:, 2]]), :]

    return globalmesh, globalfield

def plate_quantity(var, x, xmin, xmax, **kwargs):
    arg_p = np.where((x>xmin) & (x<xmax))[0]
    res = np.array([np.average(var[arg_p])])
    if(kwargs.get('return_std', False)):
        res = np.append(res, np.std(var[arg_p]))
    if(kwargs.get('return_min', False)):
        res = np.append(res, np.min(var[arg_p]))
    if(kwargs.get('return_max', False)):
        res = np.append(res, np.max(var[arg_p]))
    if(kwargs.get('return_rms', False)):
        res = np.append(res, np.sqrt(np.average(var[arg_p]**2)))
    return res

def shear_zone_quantity(var, material, material_idx, **kwargs):
    arg_sz = np.where(material == material_idx)[0]
    res = np.array([np.average(var[arg_sz])])
    if(kwargs.get('return_std', False)):
        res = np.append(res, np.std(var[arg_sz]))
    if(kwargs.get('return_min', False)):
        res = np.append(res, np.min(var[arg_sz]))
    if(kwargs.get('return_max', False)):
        res = np.append(res, np.max(var[arg_sz]))
    if(kwargs.get('return_rms', False)):
        res = np.append(res, np.sqrt(np.average(var[arg_sz]**2)))
    return res

######################################################################################################################################################

xres = 1024
zres = 256
yres = 2
inner_method = 'mumps'
restart_viscous = False
restart_step = -100
basicinput_dir = Path('../input/')
input_dir = Path('../input/')
restart_flag = 'i'
nsteps_long = 20
nsteps_short = 0
nsteps_ic = 30
impose_event = False

dt_short = 25. * s_per_year / reference_time
dt_event = 1. * s_per_year / reference_time

sz_dependence = 'depth'
friction_trans_depth = np.array([5e3, 10e3, param_Z, param_Z+10e3]) / reference_length
friction_trans_true_temp = np.array([373., 423., 623., 723.])

if(len(sys.argv) == 3):
    output_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/')
    coutput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/')
    cinput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
elif(len(sys.argv) == 4):
    param_rstep = int(sys.argv[3])
    output_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/restart_i{param_rstep}/')
    coutput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/restart_i{param_rstep}/')
    restart_step = param_rstep
    basicinput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
    input_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
    cinput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
    restart_flag = 'i'
    nsteps_long = 0
    nsteps_short = 0
    nsteps_ic = 0
    impose_event = True
    nsteps_post = 0
elif(len(sys.argv) == 5):
    param_rstep = int(sys.argv[3])
    param_Y = float(sys.argv[4])
    output_dir = Path(f'{output_dir_parent}/L_{param_Y:.2e}/{param_name}_{param_value:.2e}/restart_i{param_rstep}/')
    coutput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/restart_i{param_rstep}/')
    restart_step = param_rstep
    basicinput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
    input_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/data/')
    cinput_dir = Path(f'{output_dir_parent}/L_8.19e+07/{param_name}_{param_value:.2e}/restart_i{param_rstep}/data/')
    restart_flag = 'i'
    nsteps_long = 0
    nsteps_short = 0
    nsteps_ic = 0
    impose_event = True
    nsteps_post = 0
else:
    raise ValueError(f"Invalid number of arguments: {len(sys.argv)}. Must be 3, 4 or 5.")
data_dir = output_dir / 'data'
cdata_dir = coutput_dir / 'data'
figure_dir = output_dir / 'figures'
var_dir = output_dir / 'vars'
if(uw.mpi.rank == 0):
    output_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    cdata_dir.mkdir(exist_ok=True, parents=True)
    figure_dir.mkdir(exist_ok=True, parents=True)
    var_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("progress_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(output_dir / f"progress.log", mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(f'Started using {uw.mpi.size} threads')

domain_length = 5e3 / reference_length
resist_length = param_Y / reference_length
phi_tr = np.minimum(0.5, param_Y/1e7)
nits = 1000
residual_tol = 1e-2
nits_min = 1

class Material:
    def __init__(self, index, label, 
                 density, diffusivity, alpha,
                 max_viscosity, min_viscosity, 
                 diff_E, diff_V, diff_A,
                 disl_n, disl_E, disl_V, disl_A,
                 shear_modulus_high, shear_modulus_low,
                 cohesion, max_yield_stress):
        self.index = index
        self.label = label
        self.density = density / reference_density
        self.diffusivity = diffusivity / reference_diffusivity
        self.alpha = alpha
        self.max_viscosity = max_viscosity / reference_viscosity
        self.min_viscosity = min_viscosity / reference_viscosity
        self.diff_E = diff_E
        self.diff_V = diff_V
        self.diff_A = diff_A / reference_viscosity
        self.disl_n = disl_n
        self.disl_E = disl_E
        self.disl_V = disl_V
        self.disl_A = disl_A / (reference_viscosity/reference_stress**(1.-disl_n))
        self.shear_modulus_high = shear_modulus_high / reference_stress
        self.shear_modulus_low = shear_modulus_low / reference_stress
        self.cohesion = cohesion / reference_stress
        self.max_yield_stress = max_yield_stress / reference_stress


air_dict        = { 'density': 1000., 'diffusivity': 1e-5, 'alpha': 0.,
                    'max_viscosity': 1e17, 'min_viscosity': 1e17,
                    'diff_E': 0., 'diff_V': 0., 'diff_A': 2e17,
                    'disl_n': 1., 'disl_E': 0., 'disl_V': 0., 'disl_A': 2e17,
                    'shear_modulus_high': 750e9, 'shear_modulus_low': 750e9,
                    'cohesion': 500e6, 'max_yield_stress': 500e6 }

mantle_dict     = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': param_etamax, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 532e3, 'disl_V': 10.56e-6, 'disl_A': 1.9998e16*param_wdisl,
                    'shear_modulus_high': param_Gm, 'shear_modulus_low': param_Gm,
                    'cohesion': param_Cm, 'max_yield_stress': param_ymax }

shear_zone_dict = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': param_etamax, 'min_viscosity': 1e10,
                    'diff_E': 0e3, 'diff_V': 0e-6, 'diff_A': param_etamax,
                    'disl_n': 2.3, 'disl_E': 154e3, 'disl_V': 0e-6, 'disl_A': 9.8594e16,
                    'shear_modulus_high': param_Gsz, 'shear_modulus_low': param_Gsz,
                    'cohesion': param_Csz, 'max_yield_stress': param_ymax }

crust_dict      = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': param_etamax, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 532e3, 'disl_V': 10.56e-6, 'disl_A': 1.9998e16*param_wdisl,
                    'shear_modulus_high': param_Gsz, 'shear_modulus_low': param_Gsz,
                    'cohesion': param_Cm, 'max_yield_stress': param_ymax }

lmantle_dict    = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': param_etamax, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 2.5e-6, 'diff_A': 1.7241e12,
                    'disl_n': 1.0, 'disl_E': 0e3, 'disl_V': 0e-6, 'disl_A': param_etamax,
                    'shear_modulus_high': 60e9, 'shear_modulus_low': 60e9,
                    'cohesion': param_Cm, 'max_yield_stress': param_ymax }

air        = Material(index=0, label='air', **air_dict)
west_plate = Material(index=1, label='west_plate', **mantle_dict)
east_plate = Material(index=2, label='east_plate', **mantle_dict)
shear_zone = Material(index=3, label='shear_zone', **shear_zone_dict)
mantle     = Material(index=4, label='mantle', **mantle_dict)
west_crust = Material(index=5, label='west_crust', **crust_dict)
east_crust = Material(index=6, label='east_crust', **crust_dict)
lmantle    = Material(index=7, label='lmantle', **lmantle_dict)


if(restart_step < -1):
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, yres, zres), 
                                        minCoord    = (-domain_width_west, 0, -domain_depth), 
                                        maxCoord    = ( domain_width_east, domain_length,  air_depth) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, zres), 
                                        minCoord    = (-domain_width_west, -domain_depth), 
                                        maxCoord    = ( domain_width_east,  air_depth) )

    refine_left_x = -50e3 / reference_length
    refine_right_x = 450e3 / reference_length
    refine_buffer_x = 0e3 / reference_length
    refine_ratio_x = 4.

    refine_left_z = np.maximum(west_lithos_bottom_coords.min(), east_lithos_bottom_coords.min())
    refine_right_z = air_depth
    refine_buffer_z = 0e3 / reference_length
    refine_ratio_z = 3.

    mesh_deform_data = mesh.data.copy()
    mesh_deform_data[:, 0] = deform_map(mesh.data[:, 0], -domain_width_west, domain_width_east, 
                                        refine_left_x, refine_right_x, refine_buffer_x, refine_ratio_x)
    mesh_deform_data[:, -1] = deform_map(mesh.data[:, -1], -domain_depth, air_depth, 
                                        refine_left_z, refine_right_z, refine_buffer_z, refine_ratio_z)
    with mesh.deform_mesh(isRegular=False):
        mesh.data[:] = mesh_deform_data
    if(uw.mpi.rank == 0):
        logger.debug(f'Mesh initialized')
else:
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, yres, zres),
                                         minCoord    = (0., 0., 0.),
                                         maxCoord    = (1., 1., 1.) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, zres) )
    mesh.load(str(basicinput_dir / 'mesh_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Mesh loaded')

velocityField    = mesh.add_variable(         nodeDofCount=mesh.dim )
pressureField    = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField = mesh.add_variable(         nodeDofCount=1 )

swarm = uw.swarm.Swarm( mesh=mesh )
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
swarm.allow_parallel_nn = True

if(restart_step < -1):
    velocityField.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Velocity initialized')
    pressureField.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Pressure initialized')

    bend_buffer_width = 80e3 / reference_length
    slab_front_buffer_width = 80e3 / reference_length

    slab_top_x = slab_top_zxfunc(mesh.data[:, -1])
    slab_top_z = slab_top_xzfunc(mesh.data[:, 0])
    slab_front_z = slab_front_xzfunc(mesh.data[:, 0])
    for idx in range(mesh.data.shape[0]):
        xm = mesh.data[idx, 0]
        zm = mesh.data[idx, -1]
        if(zm > 0.):
            temperatureField.data[idx] = 0.
        elif(xm < ridge_loc):
            temperatureField.data[idx] = 1.
        elif(xm >= ridge_loc and xm < 0.):
            tx = np.minimum((xm-ridge_loc)/vel_west_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm >= 0
        and xm <= slab_front_coords[:, 0].max()*1.5
        and zm <= 0):
            xs = slab_top_x[idx]
            zs = slab_top_z[idx]
            zf = slab_front_z[idx]
            if(xm <= xs
            and zm <= zs
            and zm >= zf):
                pt_m = shg.Point(xm, zm)
                xd = -slab_top_line.distance(pt_m)
                tp = slab_top_line.project(pt_m)/vel_west_plate
                if(xm < bend_buffer_width):
                    buffer_coef = xm/bend_buffer_width
                    xd = (zm-zs)*(1.-buffer_coef) + xd*buffer_coef
                    pt_tmp = shg.Point(xm, zs)
                    tp = slab_top_line.project(pt_tmp)/vel_west_plate*(1-buffer_coef) + tp*buffer_coef
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm >= slab_top_coords[:, 1].min()):
                xd = xm-xs
                pt_tmp = shg.Point(xs, zm)
                tp = slab_top_line.project(pt_tmp)/vel_west_plate
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm <= xs
                and zm <= zs
                and zm < zf):
                xd = (xm-xs)*np.cos(slab_front_slope_rad)
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((zf-zm)*np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm < slab_top_coords[:, 1].min()):
                xd = xm-xs
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((slab_top_coords[:, 1].min()-zm)/np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            else:
                tx = np.minimum(age_east_plate, age_max_cut)
                temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm > slab_front_coords[:, 0].max()*1.2):
            tx = np.minimum(age_east_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        else:
            temperatureField.data[idx] = 1.

    temperatureField.data[temperatureField.data > 0.99] = 1.
    temperatureField.data[:] = np.clip(temperatureField.data, 0., 1.)
    if(uw.mpi.rank == 0):
        logger.debug(f'Temperature initialized')

    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=16 )
    swarm.populate_using_layout( layout=swarmLayout )
    if(uw.mpi.rank == 0):
        logger.debug(f'Swarm initialized')

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust, lmantle]

    west_lithos_bottom_z = west_lithos_bottom_xzfunc(swarm.data[:, 0])
    shear_zone_bottom_z = shear_zone_bottom_xzfunc(swarm.data[:, 0])
    slab_top_x = slab_top_zxfunc(swarm.data[:, -1])
    slab_top_z = slab_top_xzfunc(swarm.data[:, 0])
    slab_front_z = slab_front_xzfunc(swarm.data[:, 0])
    east_lithos_bottom_z = east_lithos_bottom_xzfunc(swarm.data[:, 0])
    for idx in range(swarm.data.shape[0]):
        xp = swarm.data[idx, 0]
        zp = swarm.data[idx, -1]
        if(zp > 0.):
            materialIndex.data[idx] = air.index
        elif(zp >= shear_zone_bottom_z[idx]
            and zp <= slab_top_z[idx]
            and zp >= slab_front_z[idx]
            # and zp >= east_lithos_bottom_coords.min()):
            and zp >= east_lithos_bottom_coords.min()
            and xp >= 0):
            materialIndex.data[idx] = shear_zone.index
        elif(xp >= ridge_loc
            and zp > west_lithos_bottom_z[idx]*0.5
            and zp > -west_crust_depth
            # and zp < shear_zone_bottom_z[idx]):
            and xp < 0):
            materialIndex.data[idx] = west_crust.index
        elif(xp >= ridge_loc
            and zp >= west_lithos_bottom_z[idx]
            and zp >= slab_front_z[idx]
            and zp < shear_zone_bottom_z[idx]):
            materialIndex.data[idx] = west_plate.index
        elif(xp >= slab_top_x[idx]
            and zp > -east_crust_depth):
            materialIndex.data[idx] = east_crust.index
        elif(xp >= slab_top_x[idx] 
            and zp >= east_lithos_bottom_z[idx]):
            materialIndex.data[idx] = east_plate.index
        elif(zp < -lmantle_top_depth):
            materialIndex.data[idx] = lmantle.index
        else:
            materialIndex.data[idx] = mantle.index
    if(uw.mpi.rank == 0):
        logger.debug(f'Material index initialized')

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Previous stress initialized')
else:
    velocityField.load(str(input_dir / f'velocity_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Velocity loaded')
    pressureField.load(str(input_dir / f'pressure_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Pressure loaded')
    temperatureField.load(str(basicinput_dir / f'temperature_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Temperature loaded')

    swarm.load(str(basicinput_dir / f'swarm_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Swarm loaded')

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust, lmantle]

    materialIndex.load(str(basicinput_dir / f'material_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Material index loaded')

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.load(str(input_dir / f'prestress_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Previous stress loaded')


dx_local = np.diff(np.unique(mesh.data[:, 0]))
dz_local = np.diff(np.unique(mesh.data[:, -1]))
if(yres > 0):
    dy_local = np.diff(np.unique(mesh.data[:, 1]))
else:
    dy_local = np.array([0.])
dxdydz_local_min = np.array([dx_local.min(), dy_local.min(), dz_local.min()])
dxdydz_local_max = np.array([dx_local.max(), dy_local.max(), dz_local.max()])
dxdydz_min = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_min, dxdydz_min, op=MPI.MIN)
dxdydz_max = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_max, dxdydz_max, op=MPI.MAX)

if(uw.mpi.rank == 0):
    msg = f'xres: {xres}, dx: {l_km(dxdydz_min[0]):.3e} - {l_km(dxdydz_max[0]):.3e} km\n'
    msg += f'yres: {yres}, dy: {l_km(dxdydz_min[1]):.3e} - {l_km(dxdydz_max[1]):.3e} km\n'
    msg += f'zres: {zres}, dz: {l_km(dxdydz_min[2]):.3e} - {l_km(dxdydz_max[2]):.3e} km'
    logger.debug(msg)

step = 0
time = 0.

temporaryVelocity = velocityField.copy()
volume_integral = uw.utils.Integral(1., mesh)
volume = volume_integral.evaluate()[0]
temporaryPressure = pressureField.copy()
temporaryStress = previousStress.copy()
temporaryStress.data[:] = previousStress.data[:]

appliedTraction = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=mesh.dim )
appliedTraction.data[:] = 0.

if(yres > 0):
    topWall = mesh.specialSets["MaxK_VertexSet"]
    bottomWall = mesh.specialSets["MinK_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    frontWall = mesh.specialSets["MinJ_VertexSet"]
    backWall = mesh.specialSets["MaxJ_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = frontWall + backWall
    kWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls, kWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls+kWalls, iWalls+kWalls, iWalls+jWalls) )
else:
    topWall = mesh.specialSets["MaxJ_VertexSet"]
    bottomWall = mesh.specialSets["MinJ_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls, iWalls) )


surfaceSwarm = uw.swarm.Swarm(mesh)
advector_surface = uw.systems.SwarmAdvector( swarm=surfaceSwarm, velocityField=velocityField, order=2 )
xres_surface = 501
if(yres > 0):
    yres_surface = yres+1
    surfacePoints = np.zeros((xres_surface*yres_surface, mesh.dim))
    x_surface = np.linspace(-domain_width_west, domain_width_east, xres_surface)
    y_surface = np.linspace(0, domain_length, yres_surface)
    xx_surface, yy_surface = np.meshgrid(x_surface, y_surface)
    surfacePoints[:, 0] = xx_surface.ravel()
    surfacePoints[:, 1] = yy_surface.ravel()
else:
    surfacePoints = np.zeros((xres_surface, mesh.dim))
    surfacePoints[:, 0] = np.linspace(-domain_width_west, domain_width_east, xres_surface)
if(air_depth > 0.):
    surfacePoints[:, -1] = -1e3 / reference_length
else:
    surfacePoints[:, -1] = 0.
surface_particles = surfaceSwarm.add_particles_with_coordinates( surfacePoints )

tracerSwarm = uw.swarm.Swarm(mesh)
tracer_particles = tracerSwarm.add_particles_with_coordinates( mesh.data[:] )
tracerMaterial = tracerSwarm.add_variable( dataType="int", count=1 )
tracerModulus = tracerSwarm.add_variable( dataType="double", count=1 )
tracerTemperature = tracerSwarm.add_variable( dataType="double", count=1 )
tracerVelocity = tracerSwarm.add_variable( dataType="double", count=mesh.dim )
tracerViscosity = tracerSwarm.add_variable( dataType="double", count=1 )
tracerStrainRate = tracerSwarm.add_variable( dataType="double", count=1 )
tracerStress = tracerSwarm.add_variable( dataType="double", count=1 )

boussinesq = 1.

densityMap = { material.index : material.density - boussinesq for material in material_list }
densityFn  = fn.branching.map( fn_key=materialIndex, mapping=densityMap )

diffusivityMap = { material.index : material.diffusivity for material in material_list }
diffusivityFn = fn.branching.map( fn_key=materialIndex, mapping=diffusivityMap )

alphaMap = { material.index : material.alpha for material in material_list }
alphaFn = fn.branching.map( fn_key=materialIndex, mapping=alphaMap )

unit_z = np.zeros(mesh.dim)
unit_z[-1] = -1.
buoyancyFn = (Rb * densityFn - Ra * temperatureField * alphaFn) * unit_z


maxViscosityMap = { material.index : material.max_viscosity for material in material_list }
maxViscosity = fn.branching.map( fn_key=materialIndex, mapping=maxViscosityMap )

minViscosityMap = { material.index : material.min_viscosity for material in material_list }
minViscosity = fn.branching.map( fn_key=materialIndex, mapping=minViscosityMap )

diffEMap = { material.index : material.diff_E for material in material_list }
diffE = fn.branching.map( fn_key=materialIndex, mapping=diffEMap )

diffVMap = { material.index : material.diff_V for material in material_list }
diffV = fn.branching.map( fn_key=materialIndex, mapping=diffVMap )

diffAMap = { material.index : material.diff_A for material in material_list }
diffA = fn.branching.map( fn_key=materialIndex, mapping=diffAMap )

dislNMap = { material.index : material.disl_n for material in material_list }
dislN = fn.branching.map( fn_key=materialIndex, mapping=dislNMap )

dislEMap = { material.index : material.disl_E for material in material_list }
dislE = fn.branching.map( fn_key=materialIndex, mapping=dislEMap )

dislVMap = { material.index : material.disl_V for material in material_list }
dislV = fn.branching.map( fn_key=materialIndex, mapping=dislVMap )

dislAMap = { material.index : material.disl_A for material in material_list }
dislA = fn.branching.map( fn_key=materialIndex, mapping=dislAMap )

shearModulusHighMap = { material.index : material.shear_modulus_high for material in material_list }
shearModulusHigh = fn.branching.map( fn_key=materialIndex, mapping=shearModulusHighMap )

shearModulusLowMap = { material.index : material.shear_modulus_low for material in material_list }
shearModulusLow = fn.branching.map( fn_key=materialIndex, mapping=shearModulusLowMap )

cohesionMap = { material.index : material.cohesion for material in material_list }
cohesionFn = fn.branching.map( fn_key=materialIndex, mapping=cohesionMap )

maxYieldStressMap = { material.index : material.max_yield_stress for material in material_list }
maxYieldStress = fn.branching.map( fn_key=materialIndex, mapping=maxYieldStressMap )

strainRate = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2nd_Invariant = fn.tensor.second_invariant(strainRate)
tiny_strain_rate = 1e-25 * reference_time
slipVelocity = 2. * strainRate_2nd_Invariant * shear_zone_width
# slipVelocityLocal = 2. * strainRate_2nd_Invariant * np.average([dxdydz_min[0], dxdydz_min[-1]])
constSym = 0.*strainRate
temporaryStress_2nd_Invariant = fn.tensor.second_invariant(constSym+temporaryStress)
tiny_stress = 1e-20 / reference_stress

depth = fn.misc.max(-fn.input()[mesh.dim-1], 0.)
lithostaticPressure = fn.misc.max(Rb*depth, 0.)
trueLithostaticPressure = reference_density * gravity * depth*reference_length
trueTemperature = reference_T0 + temperatureField * (reference_T1-reference_T0)
# adiabatic = 0.3e-3 # K/m
# trueTemperature = reference_T0 + temperatureField * (reference_T1-reference_T0) + adiabatic * depth*reference_length


factor_lithos = 0.
factor_shear_zone = 0.95
factor_trans_depth = np.array([60e3, 80e3]) / reference_length
factorTransition = ((depth-factor_trans_depth[0])
                     / (factor_trans_depth[1]-factor_trans_depth[0])
                     * (factor_lithos-factor_shear_zone)) + factor_shear_zone
porePressureFactorCenter = fn.branching.conditional( [ (materialIndex < shear_zone.index, factor_lithos),
                                                       (materialIndex > shear_zone.index, factor_lithos),
                                                       (depth <= factor_trans_depth[0], factor_shear_zone),
                                                       (depth <= factor_trans_depth[1], factorTransition),
                                                       (True, factor_lithos) ] )
if(yres > 0):
    factor_trans_y = np.array([-0.1, -0.1, 1.1, 1.1]) * domain_length
    yFactorFactor = fn.branching.conditional( [ (fn.input()[1] < factor_trans_y[0], 0.),
                                                (fn.input()[1] < factor_trans_y[1], 
                                                    (fn.input()[1]-factor_trans_y[0])/(factor_trans_y[1]-factor_trans_y[0])),
                                                (fn.input()[1] > factor_trans_y[3], 0.),
                                                (fn.input()[1] > factor_trans_y[2],
                                                    (factor_trans_y[3]-fn.input()[1])/(factor_trans_y[3]-factor_trans_y[2])),
                                                (True, 1.) ] )
    factor_shear_zone_edge = 0.95
    factor_trans_depth_edge = np.array([60e3, 80e3]) / reference_length
    factorTransitionEdge = ((depth-factor_trans_depth_edge[0])
                        / (factor_trans_depth_edge[1]-factor_trans_depth_edge[0])
                        * (factor_lithos-factor_shear_zone_edge)) + factor_shear_zone_edge
    porePressureFactorEdge = fn.branching.conditional( [ (materialIndex < shear_zone.index, factor_lithos),
                                                        (materialIndex > shear_zone.index, factor_lithos),
                                                        (depth <= factor_trans_depth_edge[0], factor_shear_zone_edge),
                                                        (depth <= factor_trans_depth_edge[1], factorTransitionEdge),
                                                        (True, factor_lithos) ] )
    porePressureFactor = porePressureFactorCenter * yFactorFactor + porePressureFactorEdge * (1.-yFactorFactor)
else:
    porePressureFactor = porePressureFactorCenter

normalStressLithos = (1. - porePressureFactor) * lithostaticPressure
normal_pressure_shear_zone = 100e6 / reference_stress
normalStress = fn.branching.conditional( [ (materialIndex < shear_zone.index, normalStressLithos),
                                            (materialIndex > shear_zone.index, normalStressLithos),
                                            (True, fn.misc.min(normal_pressure_shear_zone, normalStressLithos)) ] )

# truePressure = (1. - porePressureFactor) * trueLithostaticPressure
truePressure = normalStress * reference_stress
diffViscosity = diffA * fn.math.exp((diffE+truePressure*diffV) / (R_const*trueTemperature))
dislViscosity = dislA * (fn.math.exp((dislE+truePressure*dislV) / (R_const*trueTemperature)) 
                         * fn.math.pow(temporaryStress_2nd_Invariant+tiny_stress, 1.-dislN))
# pureViscosity = dislViscosity
pureViscosity = 1./(1./diffViscosity + 1./dislViscosity)
# pureViscosity = fn.misc.min(diffViscosity, dislViscosity)
viscosityFn = fn.misc.max(fn.misc.min(pureViscosity, maxViscosity), minViscosity)

shearModulusFn = shearModulusHigh + (shearModulusLow - shearModulusHigh) * fn.math.pow(temperatureField, 2.)
maxwellTime = viscosityFn / shearModulusFn


def diagnose(**kwargs):
    surf_vars = getGlobalSwarmVar(np.hstack((surfaceSwarm.data,
                                             velocityField.evaluate(surfaceSwarm.data))))
    surf_loc = surf_vars[:, 0:mesh.dim]
    surf_vel = surf_vars[:, mesh.dim:2*mesh.dim]

    pt_loc, pt_vars = getGlobalMeshVar(mesh, np.hstack((materialIndex.evaluate(mesh),
                                                        slipVelocity.evaluate(mesh))))
    # pt_vars = getGlobalSwarmVar(np.hstack((materialIndex.data,
    #                                        slipVelocity.evaluate(swarm))))
    
    pt_material = pt_vars[:, 0]
    pt_slip_vel = pt_vars[:, 1]

    if(uw.mpi.rank == 0):
        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]
        plate_vel_quantity = plate_quantity(surf_vel[:, 0], surf_loc[:, 0],
                                            ridge_loc, 0.,
                                            return_std=True,
                                            return_max=True,
                                            return_rms=True)

        slip_vel_quantity = shear_zone_quantity(pt_slip_vel, pt_material,
                                                shear_zone.index,
                                                return_std=True,
                                                return_max=True,
                                                return_rms=True)

        msg = f'Step, time (yr)\n'
        msg += f'{step:d} {t_yr(time):.3f}\n'
        msg += f'Plate velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(plate_vel_quantity)):
            msg += f'{v_mm_yr(plate_vel_quantity[i]):.3e} '
        msg += '\n'
        msg += f'Slip velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(slip_vel_quantity)):
            msg += f'{v_mm_yr(slip_vel_quantity[i]):.3e} '
        logger.debug(msg)

        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]

        if(kwargs.get('plot') is not None):
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]))
            ax.scatter(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]), s=1)
            ax.set_title(f'Step = {step}')
            ax.set_xlim(-l_km(domain_width_west), l_km(domain_width_east))
            ax.set_xlabel('x (km)')
            ax.set_ylabel('vx (mm/yr)')
            ax.grid()
            ax.minorticks_on()
            plt.tight_layout()
            plt.savefig(figure_dir / kwargs.get('plot'))
            plt.close()

        diagnostics = np.r_[plate_vel_quantity, slip_vel_quantity]
    else:
        diagnostics = np.zeros(8)

    MPI.COMM_WORLD.Bcast(diagnostics, root=0)
    return diagnostics

def checkpoint(step, flag='', **kwargs):
    if(kwargs.get('mesh', False)):
        mesh.save(str(data_dir / f'mesh_{flag}{step:d}.h5'))
    if(kwargs.get('velocity', False)):
        velocityField.save(str(data_dir / f'velocity_{flag}{step:d}.h5'))
    if(kwargs.get('pressure', False)):
        pressureField.save(str(data_dir / f'pressure_{flag}{step:d}.h5'))
    if(kwargs.get('temperature', False)):
        temperatureField.save(str(data_dir / f'temperature_{flag}{step:d}.h5'))
    if(kwargs.get('swarm', False)):
        swarm.save(str(data_dir / f'swarm_{flag}{step:d}.h5'))
    if(kwargs.get('material', False)):
        materialIndex.save(str(data_dir / f'material_{flag}{step:d}.h5'))
    if(kwargs.get('prestress', False)):
        previousStress.save(str(data_dir / f'prestress_{flag}{step:d}.h5'))

def trace(step, flag='', **kwargs):
    if(kwargs.get('swarm', False)):
        tracerSwarm.save(str(data_dir / f'tracer_swarm_{flag}{step:d}.h5'))
    if(kwargs.get('material', False)):
        tracerMaterial.data[:] = materialIndex.evaluate(tracerSwarm.data)
        tracerMaterial.save(str(data_dir / f'tracer_material_{flag}{step:d}.h5'))
    if(kwargs.get('modulus', False)):
        tracerModulus.data[:] = shearModulusFn.evaluate(tracerSwarm.data)
        tracerModulus.save(str(data_dir / f'tracer_modulus_{flag}{step:d}.h5'))
    if(kwargs.get('temperature', False)):
        tracerTemperature.data[:] = temperatureField.evaluate(tracerSwarm.data)
        tracerTemperature.save(str(data_dir / f'tracer_temperature_{flag}{step:d}.h5'))
    if(kwargs.get('velocity', False)):
        tracerVelocity.data[:] = velocityField.evaluate(tracerSwarm.data)
        tracerVelocity.save(str(data_dir / f'tracer_velocity_{flag}{step:d}.h5'))
    if(kwargs.get('viscosity', False)):
        tracerViscosity.data[:] = viscosityFn.evaluate(tracerSwarm.data)
        tracerViscosity.save(str(data_dir / f'tracer_viscosity_{flag}{step:d}.h5'))
    if(kwargs.get('strainrate', False)):
        tracerStrainRate.data[:] = strainRate_2nd_Invariant.evaluate(tracerSwarm.data)
        tracerStrainRate.save(str(data_dir / f'tracer_strainrate_{flag}{step:d}.h5'))
    if(kwargs.get('stress', False)):
        stress_2nd_Invariant = fn.tensor.second_invariant(stressFn)
        tracerStress.data[:] = stress_2nd_Invariant.evaluate(tracerSwarm.data)
        tracerStress.save(str(data_dir / f'tracer_stress_{flag}{step:d}.h5'))

def save_domain_vars(step, flag='', save_on='mesh'):
    if(save_on == 'mesh'):
        pt_loc, pt_vars = getGlobalMeshVar(mesh, np.hstack((materialIndex.evaluate(mesh),
                                                            slipVelocity.evaluate(mesh),
                                                            diffViscosity.evaluate(mesh),
                                                            dislViscosity.evaluate(mesh),
                                                            viscosityFn.evaluate(mesh),
                                                            elastViscosity.evaluate(mesh),
                                                            viscosityEff.evaluate(mesh),
                                                            viscosityType.evaluate(mesh),
                                                            frictionFn.evaluate(mesh),
                                                            yieldStress.evaluate(mesh),
                                                            stressFn.evaluate(mesh),
                                                            strainRate.evaluate(mesh))))
        if(uw.mpi.rank == 0):
            np.savez(var_dir / f'vars_{flag}{step}.npz', loc=pt_loc, vars=pt_vars)
    else:
        pt_vars = getGlobalSwarmVar(np.hstack((swarm.data,
                                               materialIndex.data,
                                               slipVelocity.evaluate(swarm),
                                               diffViscosity.evaluate(swarm),
                                               dislViscosity.evaluate(swarm),
                                               viscosityFn.evaluate(swarm),
                                               elastViscosity.evaluate(swarm),
                                               viscosityEff.evaluate(swarm),
                                               viscosityType.evaluate(swarm),
                                               frictionFn.evaluate(swarm),
                                               yieldStress.evaluate(swarm),
                                               stressFn.evaluate(swarm),
                                               strainRate.evaluate(swarm))))
        if(uw.mpi.rank == 0):
            np.savez(var_dir / f'vars_{flag}{step}.npz', loc=pt_vars[:, :mesh.dim], vars=pt_vars[:, mesh.dim:])

checkpoint(0, mesh=True, temperature=True, swarm=True, material=True)
trace(0, swarm=True, material=True, modulus=True, temperature=True)


#################################################################################################
shearZoneFactor = swarm.add_variable( dataType="double", count=1 )
shearZoneFactor.data[:] = 0.

sz_xmax = 200e3 / reference_length * param_gfactor
sz_zmax = 80e3 / reference_length
sz_x = np.linspace(0, sz_xmax, 1001)
sz_z = (slab_top_xzfunc(sz_x)+shear_zone_bottom_xzfunc(sz_x))*0.5
arg_sz_z = np.where(sz_z >= -sz_zmax)[0]
sz_x = sz_x[arg_sz_z]
sz_z = sz_z[arg_sz_z]
sz_xz = np.c_[sz_x, sz_z]

sz_factor = 1.
shear_zone_const_width = 4e3 / reference_length
shear_zone_char_width = 5e3 / reference_length
for idx in range(swarm.data.shape[0]):
    if(materialIndex.data[idx] == shear_zone.index):
        xz = np.array([swarm.data[idx, 0], swarm.data[idx, -1]])
        distance_min = np.min(np.linalg.norm(sz_xz-xz, axis=1))
        if(distance_min <= shear_zone_const_width/2.):
            shearZoneFactor.data[idx] = sz_factor
        elif(distance_min >= 3.*shear_zone_char_width/2.):
            shearZoneFactor.data[idx] = 0.
        else:
            shearZoneFactor.data[idx] = sz_factor * np.exp(-(distance_min-shear_zone_const_width/2.)**2/((shear_zone_char_width-shear_zone_const_width)**2/2.))

friction_st = param_fst
friction_dyn = np.array([param_fstic, param_fst, param_fst])
friction_dyn_lithos = 0.6
if(sz_dependence == 'depth'):
    frictionTransitionUpper = ((depth-friction_trans_depth[0])
                                / (friction_trans_depth[1]-friction_trans_depth[0])
                                * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
    frictionTransitionLower = ((depth-friction_trans_depth[2])
                                / (friction_trans_depth[3]-friction_trans_depth[2])
                                * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
    frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                    (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                    (depth <= friction_trans_depth[0], friction_dyn[0]),
                                                    (depth <= friction_trans_depth[1], frictionTransitionUpper),
                                                    (depth <= friction_trans_depth[2], friction_dyn[1]),
                                                    (depth <= friction_trans_depth[3], frictionTransitionLower),
                                                    (True, friction_dyn[2]) ] )
else:
    frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
    frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
    frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                    (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                    (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                    (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                    (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                    (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                    (True, friction_dyn[2]) ] )

frictionDynCenter = frictionDynCenterTmp*shearZoneFactor + friction_st*(1.-shearZoneFactor)
frictionFn = frictionDynCenter


#################################################################################################
if(restart_step < -1):
    if(restart_viscous):
        flag = 'v'

        yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

        plastViscosity = yieldStress / (2.*strainRate_2nd_Invariant+tiny_strain_rate)
        viscosityType = fn.branching.conditional( [ (plastViscosity < viscosityFn, 0),
                                                    (dislViscosity < diffViscosity, 1), 
                                                    (True, 2) ] )
        viscosityEff = fn.exception.SafeMaths(fn.misc.min(viscosityFn, plastViscosity))

        viscousStress      = 2. * viscosityEff * strainRate
        stressFn           = viscousStress

        stokes = uw.systems.Stokes( velocityField = velocityField,
                                    pressureField = pressureField,
                                    conditions    = [velBC, trBC],
                                    fn_viscosity  = viscosityEff,
                                    fn_bodyforce  = buoyancyFn )

        solver = uw.systems.Solver( stokes )

        # if(yres > 0 and yres < 8):
        #     solver.set_inner_method("mumps")
        # else:    
        #     solver.set_inner_method("mg")
        solver.set_inner_method(inner_method)
        solver.set_penalty(1e3)
        # solver.set_inner_rtol(1e-4)

        if(uw.mpi.rank == 0):
            logger.debug(f'Started solving viscous...')

        step = 0

        # temporaryStress.data[:] = previousStress.data[:]
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-3):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

        # if(yres > 2 and yres < 8):
        #     save_domain_vars(step, flag, save_on='mesh')
        # elif(yres <= 2):
        #     save_domain_vars(step, flag, save_on='swarm')

        trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = stress_data[:]

        checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

    flag = 'l'
    dt_e_ratio = 0.1
    dt_e = west_plate.max_viscosity/west_plate.shear_modulus_high * dt_e_ratio

    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    # if(yres > 0 and yres < 8):
    #     solver.set_inner_method("mumps")
    # else:    
    #     solver.set_inner_method("mg")
    solver.set_inner_method(inner_method)
    solver.set_penalty(1e3)
    # solver.set_inner_rtol(1e-4)

    if(uw.mpi.rank == 0):
        logger.debug(f'Started solving long-term...')

    while(step < nsteps_long):
        # temporaryStress.data[:] = previousStress.data[:]
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-2):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')
        
        dt = dt_e*phi_dt

        # if(step == nsteps_long-1):
        #     if(yres > 2 and yres < 8):
        #         save_domain_vars(step, flag, save_on='mesh')
        #     elif(yres <= 2):
        #         save_domain_vars(step, flag, save_on='swarm')

        if(step == nsteps_long-1):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
        elif(step % 10 == 0):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        if(step == nsteps_long-1):
            checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
        # elif(step % 10 == 0):
            # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

        step = step + 1
        time = time + dt


#################################################################################################
flag = 's'
dt_e = dt_short

if(nsteps_short > 0):
    # checkpoint(-1, flag, velocity=True, pressure=True, prestress=True)

    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    # if(yres > 0 and yres < 8):
    #     solver.set_inner_method("mumps")
    # else:    
    #     solver.set_inner_method("mg")
    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    # solver.set_inner_rtol(1e-4)

    if(uw.mpi.rank == 0):
        logger.debug(f'Started solving short-term...')

    while(step < nsteps_short):
        # temporaryStress.data[:] = previousStress.data[:]
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-2):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')
        
        dt = dt_e*phi_dt

        # if(step == nsteps_short-1):
        #     if(yres > 2 and yres < 8):
        #         save_domain_vars(step, flag, save_on='mesh')
        #     elif(yres <= 2):
        #         save_domain_vars(step, flag, save_on='swarm')

        if(step == nsteps_short-1):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
        elif(step % 1 == 0):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        if(step == nsteps_short-1):
            checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
        # elif(step % 5 == 0):
            # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

        step = step + 1
        time = time + dt


#################################################################################################
flag = 'i'
dt_e = dt_short

if(nsteps_ic > 0):
    friction_st = param_fstic
    friction_dyn = np.array([param_fstic, param_fstic, param_fstic])
    friction_dyn_lithos = 0.6
    if(sz_dependence == 'depth'):
        frictionTransitionUpper = ((depth-friction_trans_depth[0])
                                    / (friction_trans_depth[1]-friction_trans_depth[0])
                                    * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
        frictionTransitionLower = ((depth-friction_trans_depth[2])
                                    / (friction_trans_depth[3]-friction_trans_depth[2])
                                    * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
        frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                        (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                        (depth <= friction_trans_depth[0], friction_dyn[0]),
                                                        (depth <= friction_trans_depth[1], frictionTransitionUpper),
                                                        (depth <= friction_trans_depth[2], friction_dyn[1]),
                                                        (depth <= friction_trans_depth[3], frictionTransitionLower),
                                                        (True, friction_dyn[2]) ] )
    else:
        frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                    / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                    * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
        frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                    / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                    * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
        frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                        (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                        (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                        (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                        (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                        (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                        (True, friction_dyn[2]) ] )
        
    frictionDynCenter = frictionDynCenterTmp*shearZoneFactor + friction_st*(1.-shearZoneFactor)
    frictionFn = frictionDynCenter
    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    # solver.set_inner_rtol(1e-4)

    if(uw.mpi.rank == 0):
        logger.debug(f'Started solving short-term...')

    while(step < nsteps_ic):
        # temporaryStress.data[:] = previousStress.data[:]
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-2):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

        
        dt = dt_e*phi_dt

        # if(step == nsteps_ic-1):
        #     if(yres > 2 and yres < 8):
        #         save_domain_vars(step, flag, save_on='mesh')
        #     elif(yres <= 2):
        #         save_domain_vars(step, flag, save_on='swarm')

        if(step == nsteps_ic-1):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
        elif(step % 1 == 0):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        if(step == nsteps_ic-1):
            checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
        elif(step % 1 == 0):
            checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

        step = step + 1
        time = time + dt


#################################################################################################
if(impose_event):
    flag = 'c'
    dt_e = dt_event
    # checkpoint(-1, flag, velocity=True, pressure=True, prestress=True)

    friction_st = param_fstic
    friction_dyn = np.array([param_fdynup, param_fdynmid, param_fdynup])
    friction_dyn_lithos = 0.6
    if(sz_dependence == 'depth'):
        frictionTransitionUpper = ((depth-friction_trans_depth[0])
                                    / (friction_trans_depth[1]-friction_trans_depth[0])
                                    * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
        frictionTransitionLower = ((depth-friction_trans_depth[2])
                                    / (friction_trans_depth[3]-friction_trans_depth[2])
                                    * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
        frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                        (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                        (depth <= friction_trans_depth[0], friction_dyn[0]),
                                                        (depth <= friction_trans_depth[1], frictionTransitionUpper),
                                                        (depth <= friction_trans_depth[2], friction_dyn[1]),
                                                        (depth <= friction_trans_depth[3], frictionTransitionLower),
                                                        (True, friction_dyn[2]) ] )
    else:
        frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                    / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                    * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
        frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                    / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                    * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
        frictionDynCenterTmp = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                        (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                        (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                        (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                        (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                        (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                        (True, friction_dyn[2]) ] )
        
    frictionDynCenter = frictionDynCenterTmp*shearZoneFactor + friction_st*(1.-shearZoneFactor)
    frictionFn = frictionDynCenter
    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    # solver.set_inner_rtol(1e-4)

    longTermVelocity = velocityField.copy()
    longTermVelocity.data[:] = velocityField.data[:]

    previousVelocity = velocityField.copy()
    temporaryVelocity = velocityField.copy()
    previousTraction = appliedTraction.copy()
    temporaryTraction = appliedTraction.copy()
    expectedTraction = appliedTraction.copy()

    volume_integral = uw.utils.Integral(1., mesh)
    volume = volume_integral.evaluate()[0]
    front_area_integral = uw.utils.Integral(1., mesh, 'surface', frontWall)
    front_area = front_area_integral.evaluate()[0]

    previousVelocity.data[:] = velocityField.data[:]
    previousTraction.data[:] = appliedTraction.data[:]

    resistShearModulus = viscosityEff / dt

    expectedTraction.data[frontWall, 0] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                            ((longTermVelocity.data[frontWall, 0] 
                                            - velocityField.data[frontWall, 0]) / resist_length
                                            + previousTraction.data[frontWall, 0]
                                            / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
    expectedTraction.data[frontWall, -1] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                            ((longTermVelocity.data[frontWall, -1] 
                                            - velocityField.data[frontWall, -1]) / resist_length
                                            + previousTraction.data[frontWall, -1]
                                            / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
    
    if(uw.mpi.rank == 0):
        logger.debug(f'Started solving coseismic...')

    if(Path(cinput_dir / f'velocity_ci0.h5').is_file() and Path(cinput_dir / f'pressure_ci0.h5').is_file()):
        if(uw.mpi.rank == 0):
            logger.debug(f'Iteration 0 data exist...')
        velocityField.load(str(cinput_dir / f'velocity_ci0.h5'))
        pressureField.load(str(cinput_dir / f'pressure_ci0.h5'))
    
        stress_data = stressFn.evaluate(swarm)
        temporaryStress.data[:] = stress_data[:]
    else:
        if(uw.mpi.rank == 0):
            logger.debug(f'Solving for iteration 0 data...')
        # temporaryStress.data[:] = previousStress.data[:]
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                    velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-3):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        velocityField.save(str(cdata_dir / f'velocity_ci0.h5'))
        pressureField.save(str(cdata_dir / f'pressure_ci0.h5'))

    diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
    
    ii = 0
    while(ii < nits):
        if(uw.mpi.rank == 0):
            msg = f'Iteration, phi\n'
            msg += f'{ii+1:d} {phi_tr:.6f}'
            logger.debug(msg)
        temporaryVelocity.data[:] = velocityField.data[:]
        temporaryTraction.data[:] = appliedTraction.data[:]
        
        appliedTraction.data[frontWall, 0] = phi_tr * expectedTraction.data[frontWall, 0] + \
                                                    (1.-phi_tr) * appliedTraction.data[frontWall, 0]
        appliedTraction.data[frontWall, -1] = phi_tr * expectedTraction.data[frontWall, -1] + \
                                                    (1.-phi_tr) * appliedTraction.data[frontWall, -1]
 
        converged = False
        for i in range(500):
            temporaryVelocity.data[:] = velocityField.data[:]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3, nonLinearMaxIterations=1 )

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)
            # if(uw.mpi.rank == 0):
            #     logger.debug(f'Iteration {i}, |dv|/|v| = {dv_norm/v_norm:.6e}')

            phi_s = 0.5
            stress_data = stressFn.evaluate(swarm)
            temporaryStress.data[:] = phi_s*temporaryStress.data[:] + (1.-phi_s)*stress_data[:]

            if(i >= 1 and dv_norm/v_norm < 1e-2):
                if(uw.mpi.rank == 0):
                    logger.debug(f'Converged after iteration {i}')
                converged = True
                break
        if(not converged):
            if(uw.mpi.rank == 0):
                logger.debug(f'Not converged')

        diagnostics = diagnose(plot=f'vx_{flag}{step}_it{ii+1}.png')

        dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                velocityField - temporaryVelocity ), mesh )
        dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
        v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
        v_norm = np.sqrt(v_integral.evaluate()[0] / volume)

        dtau_integral = uw.utils.Integral( fn.math.dot( appliedTraction - temporaryTraction,
                                                        appliedTraction - temporaryTraction ), 
                                            mesh, 'surface', frontWall )
        dtau_norm = np.sqrt(dtau_integral.evaluate()[0] / front_area)
        tau_integral = uw.utils.Integral( fn.math.dot( appliedTraction, appliedTraction ), 
                                            mesh, 'surface', frontWall )
        tau_norm = np.sqrt(tau_integral.evaluate()[0] / front_area)
        
        expectedTraction.data[frontWall, 0] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                ((longTermVelocity.data[frontWall, 0] 
                                                - velocityField.data[frontWall, 0]) / resist_length
                                                + previousTraction.data[frontWall, 0]
                                                / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
        expectedTraction.data[frontWall, -1] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                ((longTermVelocity.data[frontWall, -1] 
                                                - velocityField.data[frontWall, -1]) / resist_length
                                                + previousTraction.data[frontWall, -1]
                                                / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
        
        residual_integral = uw.utils.Integral( fn.math.dot( expectedTraction - appliedTraction,
                                                            expectedTraction - appliedTraction ), 
                                                mesh, 'surface', frontWall )
        residual_norm = np.sqrt(residual_integral.evaluate()[0] / front_area)
        traction_integral = uw.utils.Integral( fn.math.dot( expectedTraction, expectedTraction ), 
                                                mesh, 'surface', frontWall )
        traction_norm = np.sqrt(traction_integral.evaluate()[0] / front_area)
    
        if(uw.mpi.rank == 0):
            msg = f'|dv| (mm/yr), |v| (mm/yr), |dv|/|v|\n'
            msg += f'{v_mm_yr(dv_norm):.6e} {v_mm_yr(v_norm):.6e} {dv_norm/v_norm:.6e}\n'
            msg += f'|dtau| (Pa), |tau| (Pa), |dtau|/|tau|\n'
            msg += f'{tau_Pa(dtau_norm):.6e} {tau_Pa(tau_norm):.6e} {dtau_norm/tau_norm:.6e}\n'
            msg += f'|Residual| (Pa), |Expectation| (Pa), |R|/|E|\n'
            msg += f'{tau_Pa(residual_norm):.6e} {tau_Pa(traction_norm):.6e} {residual_norm/traction_norm:.6e}'
            logger.debug(msg)

        if(residual_norm/traction_norm < residual_tol and ii+1 >= nits_min):
            break
        else:
            ii += 1

    dt = dt_e*phi_dt

    # if(yres > 2 and yres < 8):
    #     save_domain_vars(step, flag, save_on='mesh')
    # elif(yres <= 2):
    #     save_domain_vars(step, flag, save_on='swarm')

    trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

    stress_data = stressFn.evaluate(swarm)
    previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

    checkpoint(step, flag, velocity=True, pressure=True, prestress=True)


if(uw.mpi.rank == 0):
    logger.debug(f'Completed using {uw.mpi.size} threads\n')
