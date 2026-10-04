# Subduction-Zone-Multiscale-Dynamics

## Files

`param_impose_model_all.py`: 2D parameter studies and imposed coseismic events.
`tr_param_impose_model_all.py`: Thin 3D model with iterative boundary traction and a resistance length.
`impose_model_all.py`: Full 3D model with an along-strike friction transition.
`loading_time_search.py`, `tr_loading_time_search.py`: Search for a loading duration consistent with the estimated recurrence interval; select results through a `restart` symlink.
`convert_initial_guess.py`: Interpolate 2D velocity and pressure onto a 3D mesh.
`show_loading_time.py`: Report selected restart indices for parameter studies.

## Requirements

Python with Underworld2, NumPy, SciPy, Matplotlib, Shapely, mpi4py, h5py.

## Running

Examples:

```bash
# 2D parameter study
mpirun -n 64 python3 param_impose_model_all.py 0 0.3
mpirun -n 64 python3 param_impose_model_all.py 0 0.3 9

# 2.5D parameter study
mpirun -n 64 python3 tr_param_impose_model_all.py 0 0.3 9 5120e3

# 3D model
mpirun -n 1024 python3 impose_model_all.py 0.0 0.125
mpirun -n 1024 python3 impose_model_all.py 0.0 0.125 6

# Loading-time searches
python3 loading_time_search.py 0 0.3
python3 tr_loading_time_search.py 0 0.3 9 5120e3
```