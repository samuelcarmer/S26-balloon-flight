import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from process_imet import load_log
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress


df,_ = load_log()


def find_element(array,element):
    import numpy as np
    idx  = (np.abs(array-element)).argmin()
    return idx



def find_se(df):

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()

    g = 9.81 # m s^-2
    cp = 1004.0 # J/ (kg K)
    Lv = 2.5e6  # J/kg

    T_C = df_asc["temp_C"].to_numpy() 
    T_K = T_C + 273.15
    z = df_asc["altitude_ba"].to_numpy(dtype=float)
    #Use AGL for potential energy
    z0 = np.min(z)
    z_rel = z - z0

    p = df_asc["pressure_hPa"].to_numpy()
    RH = df_asc["rh_percent"].to_numpy()


    # saturation vapor pressure in hPa using tetens formula
    e_s = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    e = (RH/100) * e_s

    r = 0.622 * e / (p - e)
    q = r / (1.0 + r)

    dry_se = cp * T_K + g * z_rel   # J/kg
    dry_se = gaussian_filter1d(dry_se,sigma=4)
    moist_se = dry_se + Lv * q
    moist_se = gaussian_filter1d(moist_se,sigma=4)

    # Df with only near surface values
    df_surface = df[df["altitude_ba"] < z.min() + 50]

    #Near ground values for computing parcel MSE
    T0 = df_surface["temp_C"].mean()
    T0_K = T0 + 273.15
    RH0 = df_surface["rh_percent"].mean()
    p0 = df_surface["pressure_hPa"].mean()
    z0 = df_surface["altitude_ba"].mean() - z.min()
    
   

    es0 = 6.112 * np.exp((17.67 * T0) / (T0 + 243.5))  
    e0 = (RH0/100) * es0   
    r0 = 0.622 * e0 / (p0 - e0)
    q0 = r0 / (1.0 + r0)
    
    mse0 = g*z0 + cp*T0_K + Lv*q0

    d = [z_rel,z,e, e_s, RH, q,dry_se,moist_se, mse0, z0]

    return d






def BV_analysis(df,dz):

    #Constants

    g = 9.81              # m/s^2
    p0 = 1000.0           # hPa
    kappa = 0.286         # Rd/cp
    cp = 1004.0 # J/ (kg K)
    Lv = 2.5e6 # J/kg
    #b = Lv/cp



    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()

    #print(df["altitude_ba"].head())

    z = df_asc["altitude_ba"].to_numpy()      # meters
    T = df_asc["temp_C"].to_numpy() 
    p = df_asc["pressure_hPa"].to_numpy()
    RH = df_asc["rh_percent"].to_numpy()
    
    #only take ascent data
    idx = np.argsort(z)
    z = z[idx]
    T = T[idx]
    p = p[idx]
    RH = RH[idx]





    z_grid = np.arange(z.min(),z.max(),dz)

    T_grid = np.interp(z_grid, z, T) + 273.15
    p_grid = np.interp(z_grid, z, p)
    RH_grid = np.interp(z_grid, z, RH)

    #print(p_grid[0:5])
    
    e_s = 6.112 * np.exp((17.67 * T_grid) / (T_grid + 243.5))
    e = (RH_grid/100) * e_s
    r = 0.622 * e / (p_grid - e)
    #print(r)

    # potential temperature
    theta = (T_grid) * (p0 / p_grid)**kappa
    #theta = gaussian_filter1d(theta, sigma=2)

    # derivative
    dtheta_dz = np.gradient(theta, z_grid)

    # Brunt-Vaisala frequency squared
    N2 = (g / theta) * dtheta_dz   # s^-2

    # Brunt-Vaisala frequency
    N = np.sqrt(np.maximum(N2, 0))  # avoid sqrt of negative values

    mask = (z_grid > 13000) & (z_grid < 15500)
    N_hi = np.mean(N[mask])
    N_hi_med = np.median(N[mask])
    N_eff = np.sqrt(np.mean(N2[mask]))


    d = [z_grid,theta, dtheta_dz, N2,N,N_hi,N_hi_med,N_eff]

    return d





def find_lapse_rate(df,dz):

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()



    z = df_asc["altitude_ba"].to_numpy()      # meters
    T = df_asc["temp_C"].to_numpy()  

    #only take ascent data
    idx = np.argsort(z)
    z = z[idx]
    T = T[idx]

    z_grid = np.arange(z.min(),z.max(),dz)

    T_grid = np.interp(z_grid, z, T)

    #smooth noisy data, not needed with grid approach
    if False:
        T = gaussian_filter1d(T,sigma = 13)
        z = gaussian_filter1d(z, sigma=5)
    
    

    #lapse_rate = -dT / dz   # °C per meter
    lapse_rate = -np.gradient(T_grid, z_grid) 
    lapse_rate_km = lapse_rate * 1000  # °C/km
    z_mid = 0.5 * (z[:-1] + z[1:])

    return lapse_rate_km, z_grid


def temp_profile_hi(df, zmin=12000, dz=100, poly_deg=2, sigma=4, sigma_bg_m = 1000):
    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()
    df_asc = df_asc.sort_values("altitude_ba")

    z = df_asc["altitude_ba"].to_numpy(dtype=float)
    T = df_asc["temp_C"].to_numpy(dtype=float)
    time = df_asc["time_s"].to_numpy(dtype=float)

    # restrict to upper layer first
    mask = z >= zmin
    z = z[mask]
    T = T[mask]
    time = time[mask]

    # interpolate to uniform altitude grid
    z_grid = np.arange(z.min(), z.max(), dz)
    T_grid = np.interp(z_grid, z, T)
    time_grid = np.interp(z_grid, z, time)

       # smoothing scale 10m
    sigma_bg = sigma_bg_m / dz

    T_bg = gaussian_filter1d(T_grid, sigma=sigma_bg)

    # perturbation
    T_prime = T_grid - T_bg

    # light smoothing of perturbation 
    T_prime = gaussian_filter1d(T_prime, sigma=sigma)

    return z_grid, T_grid, T_bg, T_prime, time_grid



def ascent_rate(z, time, dz=5,smooth=False,smooth_sg=8):

    # use index as time (same as Excel)
    '''t = np.arange(len(z))

    rate = np.full_like(z, np.nan)

    half = dz // 2

    for i in range(half, len(z) - half):
        z_win = z[i-half:i+half+1]
        t_win = t[i-half:i+half+1]

        t_mean = t_win.mean()
        z_mean = z_win.mean()

        slope = np.sum((t_win - t_mean)*(z_win - z_mean)) / np.sum((t_win - t_mean)**2)

        rate[i] = slope   # m per sample


    rate[:half] = rate[half]
    rate[-half:] = rate[-half-1]
    '''

    z_grid = np.arange(z.min(), z.max(), dz)
    time_grid = np.interp(z_grid, z, time)

    # then compute ascent speed from z_coarse vs time_coarse
    asc_rate = np.gradient(z_grid, time_grid)
  
    rate = np.clip(asc_rate, 0, 7)
    if smooth:
        rate = gaussian_filter1d(rate, sigma=smooth_sg)
    # 1s/sample so this is effectively in units of ms^-1

    return rate,z_grid,time_grid





def peaks(df,z,f,t,dz,prom=0.1,dist = 5,fd = None):
    from scipy.signal import find_peaks

    dist = max(2, int(300 / dz))

    if fd is None:
        fd = f
    #smoothed fd is used for finding peak location, then amplitude is based off original function f
    peaks, prop_p = find_peaks(fd, prominence=prom, distance = dist,height=0)   # ~500 m separation)
    #(invert signal)
    troughs, prop_t = find_peaks(-fd, prominence=prom, distance = dist)   # ~500 m separation)

    

    z_peaks = z[peaks]

    z_troughs = z[troughs]

    fp = f[peaks]
    ft = f[troughs]

    amp_p = fp
    amp_t = ft


    #if len(z_peaks) < 2 or len(z_troughs) < 2:
        #return [np.nan, np.nan, np.nan, np.nan]

    t_peaks = t[peaks]
    t_troughs = t[troughs]


    lambda_zp = np.mean(np.diff(z_peaks))
    lambda_zt = np.mean(np.diff(z_troughs))
    T_p = np.mean(np.diff(t_peaks))
    T_t = np.mean(np.diff(t_troughs))

    d = [lambda_zp, lambda_zt, T_p,T_t,z_peaks,z_troughs,t_peaks,t_troughs, fp, ft,amp_p,amp_t]

    return d






def plot():
    plt.figure()
    plt.plot(lapse_rate_km, z)

    plt.axvline(10.0, linestyle="--", label="Dry Adiabatic", color='r')
    plt.axvline(6.0, linestyle="--", label="Moist Adiabatic", color='g')

    plt.xlim(-20, 20)

    plt.xlabel("Lapse Rate (°C/km)")
    plt.ylabel("Altitude (m)")
    plt.title("Environmental Lapse Rate")
    plt.legend()
    plt.grid()

    plt.show()



lapse_rate_km, z = find_lapse_rate(df,200)



# ascent only
idx_max = df["altitude_ba"].idxmax()
df_asc = df.loc[:idx_max].copy()

# sort
df_asc = df_asc.sort_values("altitude_ba")

# only look above 12km
df_hi = df_asc[df_asc["altitude_ba"] > 12000].copy()
z = df_hi["altitude_ba"].to_numpy()
time, timestamp = df_hi["time_s"].to_numpy(), df_hi["time_dt"].to_numpy()

dz = 100

# [z_grid,theta, dtheta_dz, N2,N,N_hi,N_hi_med,N_eff]
BVz0, theta, _, N2, N, _, _, N_eff = BV_analysis(df, dz)

# temp
zg, T_grid, T_bg, T_osc, tg = temp_profile_hi(df, dz=dz, sigma_bg_m=1000)

#[lambda_zp, lambda_zt, T_p,T_t,z_peaks,z_troughs,t_peaks,t_troughs, fp, ft,amp_p,amp_t]
_, _, Tp_t, Tt_t, _, _, tp_t, tt_t, fpT, ftT,_,_ = peaks(df, zg, T_osc, tg, dz)

omega_p_temp = 2*np.pi / Tp_t
omega_t_temp = 2*np.pi / Tt_t
omega_temp = 0.5 * (omega_p_temp + omega_t_temp)

pct_diff_temp = np.abs(omega_p_temp - omega_t_temp) / omega_temp * 100


# ascent rate
asc_rate, zc, tc = ascent_rate(z, time, dz=dz, smooth=True, smooth_sg=2)


_, _, Tp, Tt, _, _, tp, tt, a_p, a_t,_,_ = peaks(df, zc, asc_rate, tc, dz)

omega_p_asc = 2*np.pi / Tp
omega_t_asc = 2*np.pi / Tt
omega_asc = 0.5 * (omega_p_asc + omega_t_asc)

pct_diff_asc = np.abs(omega_p_asc - omega_t_asc) / omega_asc * 100










# print
print("\n--- FINAL VALUES (dz = 100 m) ---\n")

print("Temperature:")
print(f"  omega_p   = {omega_p_temp:.5f} s^-1")
print(f"  omega_t   = {omega_t_temp:.5f} s^-1")
print(f"  omega_avg = {omega_temp:.5f} s^-1")
print(f"  % diff    = {pct_diff_temp:.2f}%\n")

print("Ascent rate:")
print(f"  omega_p   = {omega_p_asc:.5f} s^-1")
print(f"  omega_t   = {omega_t_asc:.5f} s^-1")
print(f"  omega_avg = {omega_asc:.5f} s^-1")
print(f"  % diff    = {pct_diff_asc:.2f}%\n")

print(f"N_eff = {N_eff:.5f} s^-1")





#grav wave oscillations


zmin = 12000
mask = (BVz0 >= zmin)
BVz = BVz0[mask]
theta = theta[mask]
#N = N[mask]
N = gaussian_filter1d(N, sigma=5)
#amp_the = prop['peak_heights']

sigma_bg = 7

the_bg = gaussian_filter1d(theta, sigma=sigma_bg)

# perturbation
the_p = theta - the_bg
coeff_res = np.polyfit(BVz, the_p, deg=1)
theta_res_bg = np.polyval(coeff_res, BVz)
the_p_o = the_p - theta_res_bg
# light smoothing of perturbation 
the_p_o_smooth = gaussian_filter1d(the_p_o, sigma=1)

#[lambda_zp, lambda_zt, T_p,T_t,z_peaks,z_troughs,t_peaks,t_troughs, fp, ft,ampp,ampt]
#
the_lam_p, the_lam_t, the_per_p,the_per_p,z_the_p,z_the_t,_,_,the_peaks_p,the_peaks_t,amp_p,amp_t = peaks(df,BVz,the_p_o,time,dz,dist=10,fd=the_p_o_smooth,prom=0.2)
print(amp_p,amp_t)

ap_avg = np.mean(amp_p)
at_avg = np.mean(amp_t)
A = (ap_avg-at_avg)/2
print(A)




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


plt.figure(figsize=(10,4))

plt.plot(BVz0/1000, N, label="Theta (measured)")

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

plt.ylim(bottom=0) 
plt.ylabel(r"$N\ (\mathrm{s}^{-1})$")
plt.xlabel("Altitude (km)")
plt.title(r"Brunt-Väisälä Frequency")
#plt.legend()
plt.grid()
plt.savefig(out_filepath+"bv_N_poster.pdf",bbox_inches='tight',transparent=True)






'''
plt.figure(figsize=(6,8))

plt.plot(theta, BVz, label="Theta (measured)")
plt.plot(the_bg, BVz, label="Theta (background)", linestyle="--")

plt.xlabel("Temperature (°C)")
plt.ylabel("Altitude (m)")
plt.title(f"Temperature Profile with Background Fit (sigma = {sigma_bg})")
plt.legend()
plt.grid()
plt.savefig(out_filepath+"theta_vs_bg.png",dpi=300)
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



plt.figure(figsize=(4,24))
#plt.plot(the_p, BVz, label="Theta (oscillation), not corrected")
#plt.plot(theta_res_bg, BVz, label="Theta (oscillation lin fit)")
plt.plot(the_p_o, BVz/1000, color="#00a489",label=r'$\theta^{\prime}$')

plt.plot(the_p_o_smooth, BVz/1000, color="#005be3",label=r'$\theta^{\prime}$, smoothed')
plt.scatter(amp_p,z_the_p/1000, color="#004dc0")
plt.scatter(amp_t,z_the_t/1000, color="#004dc0")


plt.xlabel(r'$\theta$ (K)')
plt.ylabel("Altitude (km)")
plt.title(r'$\theta$ Oscillations')
plt.legend()
plt.grid()
#plt.savefig(out_filepath+"theta__osc.pdf",bbox_inches='tight',transparent=True)
plt.show()
'''
plt.figure(figsize=(6,6))
plt.plot(tc, asc_rate, color="#9700e3",label="Ascent rate")

plt.scatter(tp, a_p, color="#dd0202", label='Peaks')
plt.scatter(tt, a_t, color="#005be3", label='Troughs')
print(f"lent: {len(a_t)}")
plt.xlabel("Time (s)")
plt.ylabel("Ascent rate (m/s)")
plt.title(r'Ascent rate above 12 km')
plt.legend()
plt.grid()
plt.savefig(out_filepath+"asc_rate_osc_w_crp.pdf",bbox_inches='tight',transparent=True)

plt.show()










asc_rate = ascent_rate(z,5,smooth=True,smooth_sg=12)
#d = [lambda_zp, lambda_zt, T_p,T_t,z_peaks,z_troughs,t_peaks,t_troughs]
_,_,Tp,Tt,zp,zt,tp,tt,ap,at = peaks(df,z,asc_rate,time)


_,theta, dtheta_dz, _,N,N_hi, N_med,N_eff = BV_analysis(df,100)



#z_grid, T_grid, T_bg, T_prime, time_grid
sbg = 800
zg,T_grid,T_bg,T_osc,tg = temp_profile_hi(df,sigma_bg_m=sbg)
print("T_prime range:", np.min(T_osc), np.max(T_osc))
_,_,temp_freq_p,temp_freq_t,_,_,_,_,_,_ = peaks(df,zg,T_osc,tg)


print(2* np.pi / temp_freq_p,2* np.pi / temp_freq_t)


print(f"N : {N_hi:.4f}")
print(f"N with med: {N_med:.4f}")
print(f"N with N2 method: {N_eff:.4f}")

omega_p = 2*np.pi / Tp
omega_t = 2*np.pi / Tt

print("omega_p =", omega_p)
print("omega_t =", omega_t)
print("N_eff   =", N_eff)
'''










'''
dz_vals = np.arange(50, 201, 25)

omega_asc_list = []
omega_temp_list = []
N_eff_list = []



for dz in dz_vals:

    # --- BV ---
    _, _, _, N2, N, _, _, N_eff = BV_analysis(df, dz)

    # --- temperature ---
    zg, T_grid, T_bg, T_osc, tg = temp_profile_hi(df, dz=dz, sigma_bg_m=1000)

    try:
        _, _, Tp_t, Tt_t, _, _, tp_t, tt_t, _, _ = peaks(df, zg, T_osc, tg,dz)

        if len(tp_t) < 2 or len(tt_t) < 2:
            raise ValueError("Not enough peaks")
        
        T_all = np.concatenate([np.diff(tp_t), np.diff(tt_t)])
        T_mean = np.mean(T_all)

        

        omega_temp = 2*np.pi / T_mean

    except:
        omega_temp = np.nan
        print(f"dz={dz}: temp peak fail")

    try:
        asc_rate, zc, tc = ascent_rate(z, time, dz=dz,smooth=True,smooth_sg=2)

        _, _, Tp, Tt, _, _, tp, tt, _, _ = peaks(df, zc, asc_rate, tc, dz)

        if len(tp) < 2 or len(tt) < 2:
            raise ValueError("Not enough peaks")
        omega_asc = 0.5 * (2*np.pi/Tp + 2*np.pi/Tt)
    except:
        omega_asc = np.nan
        print(f"dz={dz}: asc peak fail")

    omega_temp_list.append(omega_temp)
    omega_asc_list.append(omega_asc)
    N_eff_list.append(N_eff)


'''


'''


plt.figure(figsize=(8,5))

plt.plot(dz_vals, omega_temp_list, label='Temp ω')
plt.plot(dz_vals, omega_asc_list, label='Ascent ω')
plt.plot(dz_vals, N_eff_list, label='N_eff')

plt.xlabel("dz (m)")
plt.ylabel("Angular frequency (s$^{-1}$)")
plt.title("Frequency vs Vertical Resolution (dz)")
plt.legend()
plt.grid()
plt.savefig(out_filepath+"omeg_resolution.png",dpi=300)

plt.show()










'''






'''




plt.figure()
plt.plot(T_osc, zg)
plt.title(f"T' vs altitude (sigma = {sbg})")
plt.xlabel("T'")
plt.ylabel("z")
plt.grid()
plt.savefig(out_filepath+"temp_osc.png",dpi=300)
#plt.show()

plt.figure(figsize=(6,8))

plt.plot(T_grid, zg, label="T (measured)")
plt.plot(T_bg, zg, label="T_bg (background)", linestyle="--")

plt.xlabel("Temperature (°C)")
plt.ylabel("Altitude (m)")
plt.title(f"Temperature Profile with Background Fit (sigma = {sbg})")
plt.legend()
plt.grid()
plt.savefig(out_filepath+"tempvsbg.png",dpi=300)

plt.show()


def temp_profile(df):

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()



    z = df_asc["altitude_ba"].to_numpy()      # meters
    T = df_asc["temp_C"].to_numpy() 

      #only take ascent data
    idx = np.argsort(z)
    z = z[idx]
    T = T[idx]

    dz = 100
    z_grid = np.arange(z.min(),z.max(),dz)

    T_grid = np.interp(z_grid, z, T)


    z_break = 11500  # try 11500, 12000, or 12500

    low = z_grid < z_break
    high = z_grid >= z_break

    coeff_low = np.polyfit(z_grid[low], T_grid[low], deg=2)
    coeff_high = np.polyfit(z_grid[high], T_grid[high], deg=2)

    T_bg = np.empty_like(T_grid)
    T_bg[low] = np.polyval(coeff_low, z_grid[low])
    T_bg[high] = np.polyval(coeff_high, z_grid[high])

    T_prime = T_grid - T_bg
    T_prime = gaussian_filter1d(T_prime,sigma=5)


    return z_grid, T_grid, T_prime, T_bg
'''
