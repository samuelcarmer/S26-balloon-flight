import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d




df = pd.read_csv(inpath,header=0)

idx_max = df["altitude_ba"].idxmax()
df = df.loc[:idx_max].copy()

sg = 10 # smooth amount
z = df["altitude_ba"].to_numpy()
z_diff = 1619 - z.min()
print(z_diff)
z_a = z + z_diff
rh_a = df["abq_rh"].to_numpy()
rh_maybe = 100 - rh_a
rh = df["rh_percent"]

temp_d = df["temp_C"]
temp_a = df["temp_C_abq"]

idx = df.sort_values(by=["altitude_ba"],ascending = True)
temp_d = temp_d[idx]
temp_a = temp_a[idx]


rh_a = gaussian_filter1d(rh_a,sigma=sg)
rh = gaussian_filter1d(rh,sigma=sg)

temp_a = gaussian_filter1d(temp_a,sigma=sg)
temp_d = gaussian_filter1d(temp_d,sigma=sg)


nan_count = np.isnan(temp_d/temp_a).sum()

print(f" nan_count: {nan_count}")
print(f"d length {len(temp_d)}")
print(f"abq length {len(temp_a)}")


print(df.head())

plt.figure()
plt.plot(rh,z,label="Daedelus sounding")
plt.plot(rh_a,z_a, label="ABQ sounding")
plt.plot(rh_maybe,z_a, label="100 - RH (abq)")
plt.xlabel("rh %")
plt.ylabel("z")
plt.text(25, 12500,
         f"Date: 03-12-26",
         fontsize=10)
plt.title(f"ABQ vs Team Daedelus RH profile comparison (smoothed, sigma = {sg})")
plt.legend()


plt.figure()
plt.plot(temp_d,z)
plt.plot(temp_a,z)
plt.xlabel("T")
plt.ylabel("z")
plt.title(f"ABQ vs Team Daedelus Temp profile comparison (smoothed, sigma = {sg})")


plt.show()
