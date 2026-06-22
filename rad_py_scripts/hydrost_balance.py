import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from process_imet import load_log, clean_df
from add_radia_counts_data import find_counts



df, _ = load_log()
idx_max = df["altitude_ba"].idxmax()
df = df.loc[:idx_max].copy()


# constants
g = 9.81          # m/s^2
R = 287.0         # J/(kg K)

#
z = df["altitude_gps"].to_numpy(dtype=float)      # m
P_hPa = df["pressure_hPa"].to_numpy(dtype=float)  # hPa
T_C = df["temp_C"].to_numpy(dtype=float)          # deg C
T_v = df["temp_virt"].to_numpy(dtype=float)          # deg K



# convert units
P = P_hPa * 100.0         # Pa
T_K = T_C + 273.15        # K

# -----------------------------
mask = (
    np.isfinite(z) &
    np.isfinite(P) & (P > 0) &
    np.isfinite(T_v) & (T_v > 0)
)

z = z[mask]
P = P[mask]
T_v = T_v[mask]

# sort by altitude
idx = np.argsort(z)
z = z[idx]
P = P[idx]
T_v = T_v[idx]

# shift altitude so bottom point is z=0
z0 = z[0]
z_rel = z - z0
P0 = P[0]

# 1) isothermal fit from measured P(z)

lnP = np.log(P)
slope, intercept, r_value, p_value, std_err = linregress(z_rel, lnP)

T_eff = -g / (R * slope)

P_iso = P0 * np.exp(-g * z_rel / (R * T_eff))

print("Isothermal fit:")
print(f"  slope = {slope:.6e} 1/m")
print(f"  R^2   = {r_value**2:.6f}")
print(f"  T_eff = {T_eff:.2f} K  = {T_eff - 273.15:.2f} C")

# -----------------------------
#  hydrostatic integration using measured T(z)


# build X(z) = integral of g/(R T) dz

integrand = g / (R * T_v)

X = np.zeros_like(z_rel)
dz= np.diff(z_rel)
X[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)

# test linearity: ln(P) vs X
lnP = np.log(P)

slope_X, intercept_X, r_X, _, _ = linregress(X, lnP)


P_temp_model = P0 * np.exp(-X)

# residuals and errors
res_iso = P - P_iso
res_temp = P - P_temp_model

rmse_iso = np.sqrt(np.mean(res_iso**2))
rmse_temp = np.sqrt(np.mean(res_temp**2))

frac_rmse_iso = np.sqrt(np.mean(((P - P_iso)/P)**2))
frac_rmse_temp = np.sqrt(np.mean(((P - P_temp_model)/P)**2))






print("\nModel comparison:")
print(f"  RMSE isothermal         = {rmse_iso:.2f} Pa")
print(f"  RMSE measured-T model   = {rmse_temp:.2f} Pa")
print(f"  fractional RMSE iso     = {frac_rmse_iso:.5f}")
print(f"  fractional RMSE T(z)    = {frac_rmse_temp:.5f}")




plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 18,   # title size
    "axes.labelsize": 11,   # axis label size
    "font.weight": "bold",          # global text
    "axes.labelweight": "bold",     # axis labels
    "axes.titleweight": "bold"
})

out_filepath = "/Users/samuelcarmer/Documents/Balloon_SRS/iMet/Plots/"
# plots
'''
plt.figure(figsize=(7,5))
plt.plot(P/100.0, z_rel, label="Measured pressure",color="#c20000")
plt.plot(P_iso/100.0, z_rel, label=f"Isothermal model ($T_{{eff}}={T_eff:.1f} $K)",color="#4acb00ff")
plt.plot(P_temp_model/100.0, z_rel, label="Model using measured T(z)",color='#4285f4ff')
plt.xlabel("Pressure (hPa)")
plt.ylabel("Altitude above launch (m)")
plt.title("Pressure profile comparison")
plt.grid()
plt.legend()
plt.savefig(out_filepath+"p_comp.png", dpi=400,bbox_inches='tight',transparent=True)

plt.tight_layout()


plt.figure(figsize=(7,5))
plt.plot((P - P_iso)/100.0, z_rel, label="Measured - isothermal")
plt.plot((P - P_temp_model)/100.0, z_rel, label="Measured - measured-T model")
plt.axvline(0, color="k", linewidth=1)
plt.xlabel("Pressure residual (hPa)")
plt.ylabel("Altitude above launch (m)")
plt.title("Pressure residuals")
plt.grid()
plt.legend()
plt.tight_layout()


plt.figure(figsize=(7,5))
plt.plot(100*(P - P_iso)/P, z_rel, label="Isothermal")
plt.plot(100*(P - P_temp_model)/P, z_rel, label="Measured T(z)")
plt.axvline(0, color="k", linewidth=1)
plt.xlabel("Percent residual (%)")
plt.ylabel("Altitude above launch (m)")
plt.title("Percent residuals")
plt.grid()
plt.legend()
plt.tight_layout()





print("\nLinearity test:")
print(f"R^2 (lnP vs z)  = {r_value**2:.6f}")
print(f"R^2 (lnP vs X)  = {r_X**2:.6f}")

# plot comparison
plt.figure(figsize=(6,5))
plt.scatter(z_rel, lnP, s=5, label="ln(P) vs z")
plt.xlabel("z (m)")
plt.ylabel("ln(P)")
plt.title("Isothermal assumption")
plt.grid()
plt.legend()




plt.figure(figsize=(6,5))
plt.scatter(X, lnP, s=5, label="ln(P) vs X(z)")


X_fit = np.linspace(X.min(), X.max(), 500)
lnP_fit_X = slope_X * X_fit + intercept_X
plt.plot(X_fit, lnP_fit_X, color='red', label="Linear fit")


plt.xlabel(r"X = $\int \frac{g}{R T_{virt}} dz $")
plt.ylabel("ln(P)")
text_label = fr"Model RMSE = ${rmse_temp:.4f}$"
plt.text(1.1, 11.0, text_label, 
         fontsize=10, 
         
         bbox=dict(facecolor='white', alpha=0.5,edgecolor='gray', # Light grey outline
                   linewidth=1))

plt.title("Using measured temperature")
plt.grid()
plt.legend()
plt.savefig(out_filepath+"lnP_vs_x.png", dpi=400,bbox_inches='tight',transparent=True)
plt.show()

# fit line for second plot

plt.show()
'''



dfrad,_ = find_counts(df)

print(dfrad.head())

zrad = dfrad["altitude_ba"]
prad = dfrad["pressure_hPa"]
tvrad = dfrad["temp_virt"]

maskrad =  (
    np.isfinite(zrad) &
    np.isfinite(prad) & (prad > 0)&
    np.isfinite(tvrad) & (tvrad > 0)
)
prad = prad[maskrad]
prad = 100 * prad #converts from hPa to Pa
zrad = zrad[maskrad]
tvrad = tvrad[maskrad]
zrad0 = zrad.min()
zrad_rel = zrad - zrad0
zrad_rel = zrad_rel.to_numpy()
#print(zrad[0:10])
#print(zrad_rel[0:10])
#print(zrad0)
print(prad[-10:-1])

rho = (prad / (R * tvrad)).to_numpy()

dz = np.diff(zrad_rel)
#dz = np.insert(dz, len(dz), 0)
#print(np.shape(dz),np.shape(zrad_rel))
# rho_mid offsets rho indices by 1 and take avg - trapezoidal rule
#rho_mid = 0.5 * (rho[1:] + rho[:-1])


print("prad shape:", np.shape(prad))
print("tvrad shape:", np.shape(tvrad))
print("rho shape:", np.shape(rho))
print("zrad_rel shape:", np.shape(zrad_rel))
print("dz shape:", np.shape(np.diff(zrad_rel)))
print("rho_mid shape:", np.shape(0.5 * (rho[1:] + rho[:-1])))



print("rho shape:", rho.shape)
print("rho[:-1] shape:", rho[:-1].shape)
print("rho[1:] shape:", rho[1:].shape)


rho_mid = 0.5 * (rho[1:] + rho[:-1])

#print("rho_mid shape:", rho_mid.shape)
#print("dz shape:", np.diff(zrad_rel).shape)

column_mass = np.zeros_like(zrad_rel)
#print("col mass shape:", column_mass.shape)
column_mass[:-1] = np.cumsum((rho_mid * dz)[::-1])[::-1]
#print("col mass shape 2:", column_mass.shape)
print(column_mass[0:10], column_mass[-1])







from scipy.ndimage import gaussian_filter1d

counts = dfrad["counts_per_s"]
counts = counts[maskrad]
counts_smooth = gaussian_filter1d(counts,sigma = 2)

#print(counts.head())

plt.figure()
plt.plot(column_mass,zrad)
plt.xlabel("Column mass above (kg/m2)")
plt.ylabel("Altitude (m)")
plt.gca().invert_xaxis()



plt.figure(figsize=(12,8))
plt.scatter(column_mass, counts, marker='o', color="#0074f1")
plt.plot(column_mass, counts_smooth,color="#2600cf")

plt.xlabel("Column mass above (kg/m2)")
plt.ylabel("Radiation (counts per second)")
plt.title("Radiation vs Atmospheric Column Mass")

plt.gca().invert_xaxis()  #  top of atmosphere on right
plt.grid()
plt.savefig(out_filepath+"mass_vs_cps.png", dpi=400,bbox_inches='tight',transparent=True)

plt.show()

