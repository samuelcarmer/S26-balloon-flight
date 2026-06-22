import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from process_imet import load_log, clean_df
from analyze_imet import find_lapse_rate, temp_profile, BV_analysis, find_se, temp_hum_prof, compare_virt,ascent_rate

from scipy.ndimage import gaussian_filter1d




df, _ = load_log()
#df = clean_df(df)
#print("in:  " + in_filepath+"001_001.SIG")

dpi = 600

T = True
F = False
#Easily turn on/off plots
PvG = F
AP = F
AG = F
DP = F
DG = F
PvT = F
OV = F
LAPSE = T
BV = F
SE = F
Hu = T
Tvirt = F
asc_r = F



def find_element(array,element):
    import numpy as np
    idx  = (np.abs(array-element)).argmin()
    return idx



ba_alt_name = "altitude_ba"
gps_alt_name = "altitude_gps"
time_name = "time_s"

idx_max = df[ba_alt_name].idxmax()

df_ascent = df.loc[:idx_max].copy()
df_descent = df.loc[idx_max:].copy()

df_sorted_press = df_ascent.sort_values(ba_alt_name)
df_sorted_gps = df_ascent.sort_values(gps_alt_name)

df_sorted_descent_press = df_descent.sort_values(ba_alt_name)
df_sorted_descent_gps = df_descent.sort_values(gps_alt_name)

#print(f" Columns: {df.columns}")

# Compare pressure and GPS altitude directly

'''
plt.figure()
plt.plot(df["time_s"], gaussian_filter1d(df["temp_C"],sigma = 2), label="temp vs time =")
plt.plot(df["altitude_ba"], gaussian_filter1d(df["temp_C"],sigma = 2), label="temp vs alt =")
plt.legend()
plt.xlabel("time")
plt.ylabel("temp (C)")
plt.title("temp vs time =")
'''


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



if PvG:
    plt.figure()
    plt.plot(df[ba_alt_name], label="pressure altitude")
    plt.plot(df[ba_alt_name], label="GPS altitude")
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Altitude (m)")
    plt.title("Pressure vs GPS Altitude")
    plt.savefig(out_filepath + "gps_alt_vs_pressure_alt.png", dpi=300)

    

# Plot 1: temperature vs pressure altitude ASCENT
if AP:
    z, T, T_p, T_bg, time = temp_profile(df)

    plt.figure()
    plt.plot(T, z)
    plt.plot(T_bg, z)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure Altitude (m)")
    plt.title("Temperature vs Pressure Altitude (Ascent Only)")
    plt.grid()
    plt.savefig(out_filepath + "temp_vs_pressure_alt_ascent.png", dpi=300)


    plt.figure()
    plt.plot(T_p, time)
    plt.xlabel("Temperature  Oscillations(°C)")
    plt.ylabel("Pressure Altitude (m)")
    plt.title("Temperature Oscillations vs Altitude ")
    plt.grid()
    plt.savefig(out_filepath + "temp_osc_vs_alt.png", dpi=300)


# Plot 5: Pressure altitude vs elapsed time
if PvT:
    plt.figure()


    #need to figure this out. column that i thought was column was time elapsed, but actually is slant range. Figure out way to get time data
    plt.plot(df[time_name], df[ba_alt_name])
    plt.xlabel("Time")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time")
    plt.grid()
    plt.savefig(out_filepath + "press_alt_vs_time.png", dpi=300)


if OV:
    plt.figure()
    plt.plot(df_sorted_press["temp_C"], df_sorted_press[ba_alt_name], label="Ascent")
    plt.plot(df_sorted_descent_press["temp_C"], df_sorted_descent_press[ba_alt_name], label="Descent")

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Altitude (m)")
    plt.title("Temperature vs Barometric Altitude (Ascent vs Descent)")
    plt.legend()
    plt.grid()

    plt.savefig(out_filepath + "temp_vs_alt_overlay.png", dpi=300)

if LAPSE:

    lapse_rate_km, z = find_lapse_rate(df)

    lapse_rate_km = gaussian_filter1d(lapse_rate_km,sigma=4)
    plt.figure(figsize=(8,6))
    plt.plot(lapse_rate_km, z/1000, color="#00a423")

    plt.axvline(9.8, linestyle="--", label="Dry Adiabatic", color="#c81801")
    plt.axvline(6.0, linestyle="--", label="Moist Adiabatic", color="#006ad4")

    plt.xlim(-20, 20)

    plt.xlabel("Lapse Rate (°C/km)")
    plt.ylabel("Altitude (km)")
    plt.title("Environmental Lapse Rate")
    plt.legend()
    plt.grid()

    plt.savefig(out_filepath+"env_lapse_rate_vs_alt.pdf",bbox_inches='tight',transparent=True)



if BV:

    d = BV_analysis(df)
    z,theta,N2,N = d[:]

    st_idx = find_element(z,12000)
    end_idx = z.argmax()

    N2 = gaussian_filter1d(N2,sigma=1.2)
    N = gaussian_filter1d(N,sigma=1.2)

    plt.figure()
    plt.plot(theta, z)
    plt.xlabel("Potential Temperature (K)")
    plt.ylabel("Altitude (m)")
    plt.title("Potential Temperature vs Altitude")
    plt.grid()
    plt.savefig(out_filepath + "pot_temp_vs_alt_overlay.png", dpi=300)

    plt.figure()
    plt.plot(N2, z)
    plt.axvline(0, linestyle="--")
    plt.xlabel(r"$N^2$ (s$^{-2}$)")
    plt.ylabel("Altitude (m)")
    plt.title(r"Brunt-Väisälä Frequency Squared")
    plt.grid()
    plt.savefig(out_filepath + "n2_vs_alt.png", dpi=300)

    plt.figure()
    plt.plot(N[:], z[:])
    plt.xlabel(r"$N$ (s$^{-1}$)")
    plt.ylabel("Altitude (m)")
    plt.title("Brunt-Väisälä Frequency")
    plt.grid()
    plt.savefig(out_filepath + "n_vs_alt.png", dpi=300)
    plt.show()

if SE:

    _,z,_,_,_,_,dry,moist,mse0,z0 = find_se(df)[:]

    
   

    plt.figure()
    plt.plot(dry,z/1000, label = "Dry")
    plt.plot(moist,z/1000, label = "Moist")
    

    #parcel MSE computed with mean parameters for 0-50m AGL
    #plt.axvline(mse0, linestyle="--", label="Parcel MSE", color='r')
    

    plt.xlabel("J/kg")
    plt.ylabel("Altitude (km)")
    plt.title("Static energy")
    plt.legend()
    plt.grid()

    plt.savefig(out_filepath + "compare_se.png", dpi=300)


if Hu:

    d = temp_hum_prof(df)
    RH,q,T,z = d[:]


    delta_T, z_rel = compare_virt(df)[:]

    plt.figure(figsize=(8,8))

    plt.plot(delta_T, z_rel/1000, label=r"$T - T_{\mathrm{v}}$")


    zk = z/1000

    plt.plot(q,zk, label="Specific Humidity")
    plt.plot(RH,zk, label="Relative Humidity")
    plt.plot(T,zk, label="Temperature")
    plt.xlabel("")
    plt.ylabel("Altitude (km)")
    plt.title("Normalized Humidity and Temperature Profiles")
    plt.legend()
    plt.grid()
    plt.savefig(out_filepath + "compare_temp_hum_prof.pdf",bbox_inches='tight',transparent=True)
    #plt.savefig(out_filepath+"temp_virt_cmp.png",dpi=300)


if Tvirt:

    delta_T, z_rel = compare_virt(df)[:]

    plt.figure(figsize=(6,8))

    plt.plot(delta_T, z_rel, label="T - T_virt")

    plt.xlabel("T - T_virt (°C)")
    plt.ylabel("Altitude (m)")
    plt.grid()
    plt.legend()
    plt.savefig(out_filepath+"temp_virt_cmp.png",dpi=300)


if asc_r:

    z,rate, z_hi, rate_hi, peaks, troughs = ascent_rate(df,window =5)[:]

    rate = gaussian_filter1d(rate, sigma = 7)

    

    plt.figure(figsize=(10,4))
    plt.plot(z, rate, label="Ascent rate")

    plt.xlabel("Ascent rate (m per sample)")
    plt.ylabel("Altitude (m)")
    plt.grid()
    plt.legend()

    plt.savefig(out_filepath+"asc_rate_osc.png",dpi=300)

    plt.figure(figsize=(6,8))
    plt.plot(z_hi, rate_hi, label="T'")

    plt.scatter(z_hi[peaks], rate_hi[peaks], color='red', label='Peaks')
    plt.scatter(z_hi[troughs], rate_hi[troughs], color='blue', label='Troughs')

    plt.xlabel("Altitude (m)")
    plt.ylabel("Ascent rate (m/s)")
    plt.legend()
    plt.grid()




plt.show()










"""
# Plot 2: temperature vs GPS altitude ASCENT
if AG:
    plt.figure()
    plt.plot(df_sorted_gps["temp_C"], df_sorted_gps[gps_alt_name])
    plt.xlabel("Temperature (°C)")
    plt.ylabel("GPS Altitude (m)")
    plt.title("Temperature vs GPS Altitude (Ascent Only)")
    plt.grid()
    plt.savefig(out_filepath + "temp_vs_gps_alt_ascent.png", dpi=300)



# Plot 3: temperature vs pressure altitude DESCENT
if DP:
    plt.figure()
    plt.plot(df_sorted_descent_press["temp_C"], df_sorted_descent_press[ba_alt_name])
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure Altitude (m)")
    plt.title("Temperature vs Pressure Altitude (Descent)")
    plt.grid()
    plt.savefig(out_filepath + "temp_vs_pressure_alt_descent.png", dpi=300)


# Plot 4: temperature vs GPS altitude DESCENT
if DG:
    plt.figure()
    plt.plot(df_sorted_descent_gps["temp_C"], df_sorted_descent_gps[gps_alt_name])
    plt.xlabel("Temperature (°C)")
    plt.ylabel("GPS Altitude (m)")
    plt.title("Temperature vs GPS Altitude (Descent)")
    plt.grid()
    plt.savefig(out_filepath + "temp_vs_gps_alt_descent.png", dpi=300)


"""
