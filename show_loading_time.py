from pathlib import Path
import re

base_dir = Path("/home/x-jqfang/scratch/model_2610")

param_list = [
    "fdynmid", "fdynup", "fst", "etamax", "ymax",
    "wdisl", "Gm", "Gsz", "Cm", "Csz",
    "wpa", "epa", "gfactor", "Y", "Z",
]

results = []

for subfolder in base_dir.iterdir():
    if not subfolder.is_dir():
        continue

    for param_name in param_list:
        prefix = f"{param_name}_"

        if subfolder.name.startswith(prefix):
            param_value = float(
                subfolder.name.removeprefix(prefix)
            )
            break
    else:
        continue

    restart_dir = subfolder / "restart"

    if not restart_dir.is_symlink():
        continue

    target = restart_dir.resolve(strict=False)
    match = re.fullmatch(r"restart_i(-?\d+)", target.name)

    if match:
        results.append(
            (
                param_name,
                param_value,
                int(match.group(1)),
            )
        )

for param_name, param_value, restart_i in sorted(
    results,
    key=lambda item: (param_list.index(item[0]), item[1]),
):
    print(
        f"{param_name:10s} "
        f"{param_value:12.4e} "
        f"{restart_i:4d}"
    )