# Subduction-Zone-Multiscale-Dynamics

## Files

`param_impose_model_all.py`: 2D parameter studies and imposed coseismic events.
`tr_param_impose_model_all.py`: 2.5D model with iterative boundary traction and a resistance length.
`impose_model_all.py`: Full 3D model with an along-strike friction transition.
`loading_time_search.py`, `tr_loading_time_search.py`: Search for a loading duration consistent with the estimated recurrence interval.
`convert_initial_guess.py`: Interpolate 2D velocity and pressure onto a 3D mesh.
`show_loading_time.py`: Report selected restart indices for parameter studies.

## Requirements

Python with underworld2, numpy, scipy, matplotlib, shapely, mpi4py, h5py, pandas, tqdm.

## Example workflow

The first two arguments select a parameter index and value. Index `0` sets the coseismic friction coefficient in the friction-weakening region.

### 2D

```bash
# Generate long-term and interseismic states
mpirun -n 64 python3 param_impose_model_all.py 0 0.3

# Impose an earthquake from interseismic state 9
mpirun -n 64 python3 param_impose_model_all.py 0 0.3 9

# Alternatively, search for a consistent loading duration
python3 loading_time_search.py 0 0.3
```

### 2.5D

```bash
# Generate the required 2.5D base states
mpirun -n 64 python3 tr_param_impose_model_all.py 0 0.3

# Impose an earthquake with resistance length lambda = 5120 km
mpirun -n 64 python3 tr_param_impose_model_all.py 0 0.3 9 5120e3

# Alternatively, search for a consistent loading duration
python3 tr_loading_time_search.py 0 0.3 9 5120e3
```

The total rupture length is `L = 2*sqrt(lambda*D)`, with domain thickness `D = 5 km`; this example gives `L = 320 km`.

### 3D

```bash
# Generate the base states used by earthquake restarts
mpirun -n 1024 python3 impose_model_all.py 1.1 1.1

# Impose an earthquake with an along-strike friction transition
mpirun -n 1024 python3 impose_model_all.py 0.0 0.125 6
```

The first two 3D arguments specify transition positions as fractions of the 640 km domain length.