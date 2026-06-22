import numpy as np
import matplotlib.pyplot as plt
from process_imet import load_log
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

df,_ = load_log()

def find_lapse_rate(df):

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

    #smooth noisy data, not needed with grid approach
    if False:
        T = gaussian_filter1d(T,sigma = 13)
        z = gaussian_filter1d(z, sigma=5)
    
    

    #lapse_rate = -dT / dz   # °C per meter
    lapse_rate = -np.gradient(T_grid, z_grid) 
    lapse_rate_km = lapse_rate * 1000  # °C/km
    z_mid = 0.5 * (z[:-1] + z[1:])

    return lapse_rate_km, z_grid

def temp_profile(df):

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()



    z = df_asc["altitude_ba"].to_numpy()      # meters
    T = df_asc["temp_C"].to_numpy() 
    time = df_asc["time_s"].to_numpy()

      #only take ascent data
    idx = np.argsort(z)
    z = z[idx]
    T = T[idx]
    time = time[idx]

    dz = 100
    z_grid = np.arange(z.min(),z.max(),dz)

    T_grid = np.interp(z_grid, z, T)
    time_grid = T_grid = np.interp(z_grid, z, time)


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


    return z_grid, T_grid, T_prime, T_bg, time_grid

def alt_diff(df):
    diff = df["altitude_ba"] - df["altitude_gps"]
    return diff





def BV_analysis(df):

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





    dz = 100
    z_grid = np.arange(z.min(),z.max(),dz)

    T_grid = np.interp(z_grid, z, T) + 273.15
    p_grid = np.interp(z_grid, z, p)
    RH_grid = np.interp(z_grid, z, RH)

    print(p_grid[0:5])
    
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

    d = [z_grid,theta,N2,N]

    return d

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


def temp_hum_prof(df):

    def normalize(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()

    
    

    g = 9.81 # m s^-2
    cp = 1004.0 # J/ (kg K)
    Lv = 2.5e6  # J/kg

    T_C = df_asc["temp_C"].to_numpy() 
    T_C = gaussian_filter1d(T_C,sigma = 4)
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


    T_norm = normalize(T_C)
    RH_norm = normalize(RH)
    q_norm = normalize(q)

    d = [RH_norm,q_norm,T_norm,z]
    return d

def compare_virt(df):


    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()

    T_C = df_asc["temp_C"].to_numpy() 
    T_V = df_asc["temp_virt"].to_numpy() -273.15

    z = df_asc["altitude_ba"].to_numpy(dtype=float)
    #Use AGL for potential energy
    z0 = np.min(z)
    z_rel = z - z0

    delta_T = np.abs(T_C - T_V)
    delta_T = gaussian_filter1d(delta_T,sigma=4)

    d = [delta_T,z_rel]

    return d




def ascent_rate(df, window=5):

    idx_max = df["altitude_ba"].idxmax()
    df_asc = df.loc[:idx_max].copy()

    z = df_asc["altitude_ba"].to_numpy(dtype=float)

    # use index as time (same as Excel)
    t = np.arange(len(z))

    rate = np.full_like(z, np.nan)

    half = window // 2

    for i in range(half, len(z) - half):
        z_win = z[i-half:i+half+1]
        t_win = t[i-half:i+half+1]

        slope, _, _, _, _ = linregress(t_win, z_win)
        rate[i] = slope   # m per sample

    mask = (rate > 0) & (rate < 7)
    rate = np.where(mask, rate, np.nan) 


    # convert to m/s if you know sampling rate
    # e.g. if 1 sample per second:
    # rate = rate * 1


    from scipy.signal import find_peaks

    rate = gaussian_filter1d(rate, sigma = 7)

    # --- restrict to upper region ---
    mask = z > 13000
    z_hi = z[mask]
    rate_hi = rate[mask]

    # --- find peaks ---
    peaks, _ = find_peaks(rate_hi, prominence=0.3, distance=5)

    # --- find troughs (invert signal) ---
    troughs, _ = find_peaks(-rate_hi, prominence=0.3, distance=5)

    z_peaks = z_hi[peaks]
    lambda_z = np.diff(z_peaks)

    print("Mean wavelength (m):", np.mean(lambda_z))

    d = [z, rate, z_hi, rate_hi, peaks, troughs]

    return d
    
d = find_lapse_rate(df)
gamma, zg = d[:]
mask = (zg>=12500)
zg = zg[mask]
gamma = gamma[mask]

print(np.shape(zg),np.shape(gamma))
print(f"Avg env lapse rate above 12.5km: {np.mean(gamma)} C/km")

#print(d)
