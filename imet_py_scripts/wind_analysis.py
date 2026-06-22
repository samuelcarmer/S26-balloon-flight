import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from process_imet import load_log, df_wind
from analyze_imet import temp_profile

def normalize(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

df,_ = load_log()
df = df_wind(df)

# choose ascent only
idx_max = df["altitude_ba"].idxmax()
df_asc = df.loc[:idx_max].copy().sort_values("altitude_ba")


z = df_asc["altitude_ba"].to_numpy(dtype=float)        # m
lat = df_asc["latitude"].to_numpy(dtype=float)         # deg
lon = df_asc["longitude"].to_numpy(dtype=float)        # deg

V = df_asc["speed_knot"].to_numpy(dtype=float)         # m/s


# wind components
# u > 0 eastward, v > 0 northward
# ----------------------------
theta = df_asc["wind_dir_rad"].to_numpy(dtype=float) 
u = -V * np.sin(theta)
v = -V * np.cos(theta)



# shear vectors
# magnitude of shear vs


du_dz = np.gradient(u, z)
dv_dz = np.gradient(v, z)

shear_mag = np.sqrt(du_dz**2 + dv_dz**2)
#shear_mag = gaussian_filter1d(shear_mag, sigma=14)


T = df_asc["temp_C"].to_numpy() 

    #only take ascent data
idx = np.argsort(z)
z = z[idx]
T = T[idx]

dz = 100
z_grid = np.arange(z.min(),z.max(),dz)
T_grid = np.interp(z_grid, z, T)
shr_grid = np.interp(z_grid, z, shear_mag)
shr_grid = gaussian_filter1d(shr_grid,sigma=3)

shr_grid = normalize(shr_grid)
T_grid = normalize(T_grid)




# horizontal displacement from launch point
# local tangent-plane approximation
# good for balloon-scale distances
R_earth = 6.371e6  # m

lat0 = np.deg2rad(lat[0])
lon0 = np.deg2rad(lon[0])

lat_r = np.deg2rad(lat)
lon_r = np.deg2rad(lon)

dy = R_earth * (lat_r - lat0)                          # north displacement (m)
dx = R_earth * np.cos(lat0) * (lon_r - lon0)          # east displacement (m)

disp_h = np.sqrt(dx**2 + dy**2)                       # horizontal distance from launch (m)

z_rel = z - z[0]

# plots
out_filepath = "/Users/samuelcarmer/Documents/Balloon_SRS/iMet/Plots/"

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
})

# plot shear
plt.figure(figsize=(6,10))

plt.plot(shr_grid, z_grid/1000, color="#004dc0", linewidth=2)
plt.plot(T_grid, z_grid/1000, color="#38bc04", linewidth=2)

plt.xlabel(r"Shear magnitude $\left(\mathrm{s^{-1}}\right)$")
plt.ylabel("Altitude (km)")
plt.title("Vertical Wind Shear")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_filepath+"shr_tmp_comparison.pdf",bbox_inches='tight',transparent=True)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18,6), sharey=True)



plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "lines.linewidth": 3,

    "axes.grid": True,
    "grid.alpha": 0.3,

    "figure.dpi": 150
})

colors = {
    "u": "#0e0078",   # blue
    "v": "#780505",   # red
    "disp": "#863905" # green
}
# u(z)
axes[0].plot(u, z_rel, color=colors["u"])
axes[0].set_xlabel("u (m/s)")
axes[0].set_ylabel("Altitude (m)")
axes[0].set_yticks(np.arange(0, 15000, 2000))
axes[0].set_title("Zonal Wind")


# v(z)
axes[1].plot(v, z_rel, color=colors["v"])
axes[1].set_xlabel("v (m/s)")
axes[1].set_title("Meridional Wind")

# displacement
axes[2].plot(disp_h / 1000.0, z_rel, color=colors["disp"])
axes[2].set_xlabel("Displacement (km)")
axes[2].set_title("Horizontal Drift")



# clean spines 
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



plt.tight_layout()
#plt.savefig(out_filepath+"mer_zon_wind.pdf", bbox_inches='tight')

'''
plt.figure(figsize=(6, 5))
plt.plot(dx / 1000.0, z_rel, label="x east")
plt.plot(dy / 1000.0, z_rel, label="y north")
plt.xlabel("Displacement (km)")
plt.ylabel("Altitude above launch (m)")
plt.title("Horizontal displacement components")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(6, 6))
plt.plot(dx / 1000.0, dy / 1000.0)
plt.scatter(0, 0, label="launch")
plt.xlabel("East displacement (km)")
plt.ylabel("North displacement (km)")
plt.title("Balloon horizontal track")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
'''
plt.show()