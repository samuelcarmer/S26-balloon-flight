import numpy as np
from process_imet import load_log, df_wind

import matplotlib.pyplot as plt

df,_ = load_log()
df = df_wind(df)
idx_max = df["altitude_ba"].idxmax()
df = df.loc[:idx_max].copy()

V = df["speed_knot"].to_numpy(dtype=float)
theta = df["wind_dir_rad"].to_numpy(dtype=float)


u = -V * np.sin(theta)
v = -V * np.cos(theta)
z = df["altitude_ba"].to_numpy(dtype=float)

idx_max = np.argmax(z)
u = u[:idx_max]
v = v[:idx_max]
z = z[:idx_max]

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


plt.figure(figsize=(8,8))

sc = plt.scatter(u, v, c=z/1000, cmap="viridis", s=10)
plt.plot(u, v, linewidth=1)

plt.xlabel("u, East-West (m/s)")
plt.ylabel("v, North-South (m/s)")
plt.title("Hodograph")

plt.colorbar(sc, label="Altitude (km)")
plt.axhline(0)
plt.axvline(0)

plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.savefig(out_filepath+"hodograph.pdf", bbox_inches='tight',transparent=True) 

plt.show()