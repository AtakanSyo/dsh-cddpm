import os
import numpy as np
import shutil

# ============================================================
# CONFIG
# ============================================================
CAMERA_DISTANCE = 10000        # pc
GRID_SIZE       = 32

# Camera grid
N_CAM_SIDE   = 50
CAM_SPACING  = 0.48            # pc
APERTURE     = 0.24            # pc
FOV_DEG      = 0.6             # deg
IMG_N        = 48

# Dust slab
DISC_RADIUS  = 700.0           # pc
DISC_H       = 1e-1            # pc

# Ranges
X_MIN, X_MAX = 0.05, 0.97
NH_0, NH_1   = 1e21, 1e23

PC_TO_CM = 3.0856775814913673e18

template_dir = "template_params_par"

# ============================================================
# CAMERA GRID
# ============================================================
def camera_coords(n_side, spacing):
    start = -(n_side - 1) * spacing / 2.0
    return [round(start + i * spacing, 4) for i in range(n_side)]


COORDS = camera_coords(N_CAM_SIDE, CAM_SPACING)

assert abs(CAM_SPACING - 2 * APERTURE) < 1e-9, \
    "spacing must be 2*aperture for tangent/non-overlapping cameras"


def build_repeat_tail(coords):
    out = []
    first = True

    for x in coords:
        for y in coords:
            if first:
                first = False
                continue

            out.append(f"IMAGE REPEAT {x} {y} {CAMERA_DISTANCE}\n")

    return "".join(out)


REPEAT_TAIL = build_repeat_tail(COORDS)

FIRST_X = COORDS[0]
FIRST_Y = COORDS[0]


def image_header(fits_path):
    return (
        f"IMAGE NEW {fits_path} {FIRST_X} {FIRST_Y} {CAMERA_DISTANCE} "
        f"{APERTURE} {FOV_DEG} {IMG_N}\n"
    )


ALLPHOT_BLOCK = (
    image_header("fits_images/allphotons_{dir}.fits")
    + REPEAT_TAIL
)

DUST_BLOCK = (
    image_header("fits_images/dust_{dir}.fits")
    + "IMAGE DUSTSCAT > 0\n"
    + REPEAT_TAIL
)

print(
    f"Camera grid built ONCE: {N_CAM_SIDE}x{N_CAM_SIDE} = {N_CAM_SIDE**2} "
    f"cameras, corner {COORDS[0]}..{COORDS[-1]}, "
    f"REPEAT lines/block = {REPEAT_TAIL.count(chr(10))}"
)

# ============================================================
# TEMPLATE
# ============================================================
with open(os.path.join(template_dir, "params.par")) as f:
    tmpl_lines = f.readlines()

first_img = next(
    i for i, line in enumerate(tmpl_lines)
    if line.lstrip().startswith("IMAGE NEW")
)

last_img = max(
    i for i, line in enumerate(tmpl_lines)
    if line.lstrip().startswith("IMAGE")
)

PREFIX_LINES = tmpl_lines[:first_img]
TAIL_LINES = tmpl_lines[last_img + 1:]


def set_directive(lines, key, new_line):
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(key):
            lines[idx] = new_line + "\n"
            return True

    return False


# ============================================================
# GRIDS
# ============================================================
x_vals = (np.linspace(X_MIN, X_MAX, GRID_SIZE) * CAMERA_DISTANCE).astype(int)
nH_vals = np.geomspace(NH_0, NH_1, GRID_SIZE)

print("z (pc):", x_vals)
print("nH:", nH_vals)

# ============================================================
# COPY STATIC FILES
# ============================================================
other_files = [
    f for f in os.listdir(template_dir)
    if f != "params.par"
    and os.path.isfile(os.path.join(template_dir, f))
]

# ============================================================
# MAIN LOOP
# ============================================================
for x in x_vals:
    for nH in nH_vals:
        x_str = f"{x}".replace(".", "_")
        nH_str = f"{nH:.2e}".replace("+", "_").replace(".", "_")

        dir_name = f"x_{x_str}_0_nH_{nH_str}"

        os.makedirs(os.path.join(dir_name, "fits_images"), exist_ok=True)

        for fn in other_files:
            shutil.copy(
                os.path.join(template_dir, fn),
                os.path.join(dir_name, fn)
            )

        # Remove old DENSITY / COLDENS lines from the template prefix.
        # Also remove LENGTH lines only around the old density block.
        prefix = []
        skip_next_density_block = False

        i = 0
        while i < len(PREFIX_LINES):
            line = PREFIX_LINES[i]
            stripped = line.lstrip()

            if stripped.startswith("COLDENS"):
                i += 1
                continue

            if stripped.startswith("DENSITY"):
                i += 1
                continue

            prefix.append(line)
            i += 1

        z = x

        # Convert column density NH [cm^-2] to volume density [cm^-3]
        # using slab thickness DISC_H [pc].
        disc_h_cm = DISC_H * PC_TO_CM
        density_cm3 = nH / disc_h_cm

        # No scientific notation for DENSITY.
        density_str = f"{density_cm3:.3f}"

        set_directive(
            prefix,
            "OBJECT DISC",
            f"OBJECT DISC DUST_CLOUD 0.0 0.0 {z} {DISC_RADIUS} {DISC_H}"
        )

        # Insert density block immediately before OBJECT DISC.
        inserted_density = False

        for idx, line in enumerate(prefix):
            if line.lstrip().startswith("OBJECT DISC"):
                prefix.insert(idx, "LENGTH Centimeter\n")
                prefix.insert(idx + 1, f"DENSITY {density_str}\n")
                prefix.insert(idx + 2, "LENGTH Parsec\n")
                inserted_density = True
                break

        if not inserted_density:
            raise RuntimeError("Could not find OBJECT DISC line in template prefix.")

        body = (
            ALLPHOT_BLOCK.format(dir=dir_name)
            + "\n"
            + DUST_BLOCK.format(dir=dir_name)
            + "\n"
        )

        with open(os.path.join(dir_name, "params.par"), "w") as f:
            f.writelines(prefix)
            f.write(body)
            f.writelines(TAIL_LINES)

        for exe in ("reflex_3.0_Linux_x86_64_2.17",):
            p = os.path.join(dir_name, exe)

            if os.path.exists(p):
                os.chmod(p, 0o755)

print("Done.")