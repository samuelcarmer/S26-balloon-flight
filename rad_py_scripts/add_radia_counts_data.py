import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


from process_imet import load_log, clean_df

in_file_path = "/Users/samuelcarmer/Documents/Balloon_SRS/Radiacode/"
out_filepath = "/Users/samuelcarmer/Documents/Balloon_SRS/iMet/Plots/"

df_imet, start_time = load_log()
#df_imet = clean_df(df_imet)


def find_element(array,element):
    import numpy as np
    idx  = (np.abs(array-element)).argmin()
    return idx




def find_counts(df_imet):
    dfrad = pd.read_html(in_file_path+"Balloon.html")[0]

    


    dfrad.columns = [ 
        "counts_per_s",
        "realtime", 
        "counts",
        "cr_error",
        "meas_interval",
        "temp_C",
        "charge",
        "comment"
    ]


    #Change error to numeric
    dfrad["cr_error"] = (
        dfrad["cr_error"]
        .astype(str)
        .str.replace("±", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    dfrad["cr_error"] = pd.to_numeric(dfrad["cr_error"], errors="coerce")





    #Fix html formatting
    dfrad["realtime"] = dfrad["realtime"].str.replace(
        r":(\d{3})$", r".\1", regex=True
    )
    dfrad = dfrad[
        dfrad["realtime"].str.contains("Mar", na=False)
    ].copy() #Only read time stamps with "March" to avoid headers


    # Parse "realtime" with specific headers and convert it to stand datetime format
    dfrad["time_dt"] = pd.to_datetime(
        dfrad["realtime"],
        format="%b %d, %Y %H:%M:%S.%f",
        errors="coerce"
    )
    #print("rows after parsing:", len(dfrad))

    # drop failed parses
    dfrad = dfrad.dropna(subset=["time_dt"]).copy()
    #print("rows after parsing 2:", len(dfrad))



    #Convert from MST to UTC 
    dfrad["time_dt"] = dfrad["time_dt"].dt.tz_localize("America/Denver")
    dfrad["time_dt"] = dfrad["time_dt"].dt.tz_convert("UTC")
    dfrad["time_dt"] = dfrad["time_dt"].dt.tz_localize(None)


    launch_time = start_time

    #Filter out data before launch
    dfrad = dfrad[dfrad["time_dt"] >= launch_time].copy()
    #print("rows after filtering:", len(dfrad))


    dfrad["time_s"] = (dfrad["time_dt"] - launch_time).dt.total_seconds()




    #print(dfrad[["realtime", "time_dt"]].head())
    #print(dfrad[["realtime", "time_dt"]].tail())
    #print("min time:", dfrad["time_dt"].min())
    #print("max time:", dfrad["time_dt"].max())

    #print("launch_time:", launch_time)

    dfrad = dfrad[[
        "counts_per_s",
        "realtime",
        "counts",
        "cr_error",
        "meas_interval",
        "temp_C",
        "time_dt",
        "time_s"
        ]]

    dfrad = dfrad.iloc[::-1].reset_index(drop=True)

    dfrad = dfrad.dropna(subset=["counts_per_s"]).copy()

    dfrad = dfrad.sort_values("time_s")
    df_imet = df_imet.sort_values("time_s")

    dfrad["altitude_ba"] = np.interp(
        dfrad["time_s"],
        df_imet["time_s"],
        df_imet["altitude_ba"]
    )
    dfrad["pressure_hPa"] = np.interp(
        dfrad["time_s"],
        df_imet["time_s"],
        df_imet["pressure_hPa"]
    )
    dfrad["temp_virt"] = np.interp(
        dfrad["time_s"],
        df_imet["time_s"],
        df_imet["temp_virt"]
    )



    dfrad["counts_per_s"] = (
        dfrad["counts_per_s"]
        .astype(str)
        .str.replace(" cps", "", regex=False)
        .str.strip()
    )




    dfrad["counts_per_s"] = pd.to_numeric(dfrad["counts_per_s"], errors="coerce")
    dfrad = dfrad.dropna(subset=["counts_per_s"]).copy()




    #print(dfrad["counts"].head(20))

    #print(dfrad["counts"].astype(str).unique()[:20])


    #print("Before dropna:", len(dfrad))
    dfrad["counts"] = pd.to_numeric(dfrad["counts"], errors="coerce")
    #print("NaNs in cps:", dfrad["counts"].isna().sum())
    #print(dfrad[dfrad["counts"].isna()])
    dfrad = dfrad.dropna(subset=["counts"]).copy()

    #print("After dropna:", len(dfrad))

    #print(dfrad[["altitude_ba", "counts_per_s","counts"]].dtypes)

    # convert measurement interval strings like 0:01:36 to seconds
    dfrad["meas_s"] = pd.to_timedelta(dfrad["meas_interval"]).dt.total_seconds()

    #Drop nan error rows; currently are none so not effective
    #dfrad = dfrad.dropna(subset=["cr_error"]).copy()



    idx_max = dfrad["altitude_ba"].idxmax()

    

    #dfrad = dfrad.sort_values("altitude_ba")

    dfrad_ascent = dfrad.loc[:idx_max].copy()
    #print(f"time at max: {dfrad['time_dt'].max()}")
    #dfrad_ascent = dfrad

    #dfrad_ascent = dfrad_ascent[dfrad_ascent["cr_error"] <= 9].copy()


    #print("Max error:", dfrad_ascent["cr_error"].max())

    dfrad_ascent["cps_err"] = (
        dfrad_ascent["cr_error"] / 100 * dfrad_ascent["counts_per_s"]
    )
    return dfrad_ascent, dfrad



dfrad_ascent, dfrad = find_counts(df_imet)


from scipy.signal import find_peaks


def fix_offest(df,sigma = 2 ):

    from scipy.ndimage import gaussian_filter1d

    def find_element(array,element):
        import numpy as np
        idx  = (np.abs(array-element)).argmin()
        return idx




    df["counts_per_s"] = pd.to_numeric(dfrad["counts_per_s"], errors="coerce")

    cps = df["counts_per_s"]
    cps = gaussian_filter1d(cps,sigma)

    z = df["altitude_ba"].to_numpy()

    searchi,searchf = find_element(z,8950),find_element(z,12000)
    zs = z[searchi:searchf]
    cps_search = cps[searchi:searchf]
    steep = np.gradient(cps_search,zs)
    print(steep)
    double = np.gradient(steep,zs)

    doubleg = np.abs(double) * 10**6

    crp,_ = find_peaks(doubleg)

    c = cps_search

    ctrp = c[crp]
    zsc  = zs[crp]


    c[crp[0]-2:crp[1]+3] = np.nan

    off = np.nanmax(c) - np.nanmin(c)
    cps[(searchi+crp[1]+3):] = cps[(searchi+crp[1]+3):] + off
    print(f"off: {off}")

    cps1 = df["counts_per_s"]



    #print(cps_search[0:10])
    return c, steep,zs,double,cps,z,cps1


c,dc,zs,double,cpsall,zall,sca_cps = fix_offest(dfrad_ascent)
dcp = np.abs(dc) * 10**3
doubleg = np.abs(double) * 10**6
crp,_ = find_peaks(doubleg)

ctrp = c[crp]
zsc  = zs[crp]


c[crp[0]-2:crp[1]+3] = np.nan




#off = c.max() - c.min()

'''
print(f" dcp: {dcp[0:10]}")
print(f" doubl: {double[0:10]}")
print(f" ctrp len: {len(ctrp)}")
'''
from scipy.ndimage import gaussian_filter1d

#not offset counts: 

dfrad_error = dfrad_ascent[dfrad_ascent["cr_error"] <= 9].copy()
dfrad_error["counts_per_s"] = pd.to_numeric(dfrad_error["counts_per_s"], errors="coerce")
cne = dfrad_error["counts_per_s"]
cne = gaussian_filter1d(cne,sigma=2)
zerr = dfrad_error["altitude_ba"]

#plot it

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 18,
    "axes.labelsize": 11,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

out_filepath = "/Users/samuelcarmer/Documents/Balloon_SRS/iMet/Plots/"

fig, ax = plt.subplots()

# --- original plots (unchanged) ---
ax.scatter(zall/1000, sca_cps, marker="o", color='#4285f4ff', s=10)

ax.plot(zall/1000, cpsall, label='Profile with Offset',
        color="#dd0000", linewidth=2)

ax.plot(zerr/1000, cne, label="Profile with Error Removed",
        color="#6700d4", linewidth=2)

ax.set_xlabel("Altitude above launch (km)")
ax.set_ylabel("Counts per second")
ax.set_title("Radiation vs Altitude")
ax.grid()
ax.legend()

# --- second x-axis BELOW (kft) ---
ax2 = ax.twiny()

# move second axis to bottom and offset it downward
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 40))

# match limits (in km), then relabel as kft
xmin, xmax = ax.get_xlim()
ax2.set_xlim(xmin, xmax)

# convert tick labels km -> kft
ticks = ax.get_xticks()
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{t * 3.28084:.0f}" for t in ticks])

ax2.set_xlabel("Altitude above launch (kft)")

plt.tight_layout()
plt.savefig(out_filepath+"two_rad_prof.png", dpi=400,bbox_inches='tight',transparent=True)
plt.show()




# keep only rows with reasonably long integrations
#dfrad_long = dfrad_ascent[dfrad_ascent["meas_s"] >= 90].copy()

#dfrad_short = dfrad_ascent[dfrad_ascent["meas_s"] <= 90].copy()
#print(len(dfrad_short["meas_s"]))

#print(dfrad_ascent["cr_error"].dtype)



#dfrad["time"] = pd.to_datetime(dfrad["time"])

#print(df_imet.head())

#print(dfrad.head())
#print(dfrad.tail())

#print(dfrad["time_dt"].head())
#print(dfrad["time_dt"].tail())


#dfrad["Real Time"] = pd.to_datetime(dfrad["Real Time"]).dt.total_seconds()




#dff = pd.concat([df,dfrad],axis=1)


'''
print("Max altitude:", dfrad_ascent["altitude_ba"].max())
print("Last few rows:")
print(dfrad_ascent.tail())
print("time of max alt",dfrad["time_dt"][idx_max])
print("max error",dfrad["cr_error"].max())



plt.figure()
plt.scatter(
    dfrad_ascent["altitude_ba"],
    dfrad_ascent["counts_per_s"],
    s=15,
    
)

# error bars (lighter)
plt.errorbar(
    dfrad_ascent["altitude_ba"],
    dfrad_ascent["counts_per_s"],
    yerr=dfrad_ascent["cps_err"],
    fmt='none',
    alpha=0.4
)

plt.xlabel("Altitude (m)")
plt.ylabel("Counts per second")
plt.title("Radiation vs Altitude with Error Bars")
#plt.legend()
plt.savefig(out_filepath+"rad_vs_alt_errorbars.png",dpi=300)

''
plt.figure()
#plt.plot(dfrad_ascent["altitude_ba"], dfrad_ascent["cr_error"], label="error")
plt.scatter(dfrad_ascent["altitude_ba"], dfrad_ascent["counts_per_s"], s=10, label="cps")
#plt.scatter(dfrad_long["altitude_ba"], dfrad_long["counts_per_s"], s=10, label="meas >= 90 s")
plt.legend()
plt.savefig(out_filepath+"rad_vs_alt_w_error.png",dpi=300)
plt.show()



plt.figure()
plt.plot(dfrad["time_s"], dfrad["altitude_ba"])
plt.axvline(dfrad["time_s"][idx_max], color="red")
plt.title("Altitude vs index (check peak)")


plt.figure()
plt.scatter(dfrad_ascent["altitude_ba"], dfrad_ascent["counts_per_s"], s=10)
plt.xlabel("Altitude (m)")
plt.ylabel("Counts per second")
plt.title("Radiation vs Altitude")
plt.savefig(out_filepath+"rad_vs_alt.png",dpi=300)

plt.show()


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ---------------- PARAMETERS ----------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 18,   # title size
    "axes.labelsize": 11,   # axis label size
    "font.weight": "bold",          # global text
    "axes.labelweight": "bold",     # axis labels
    "axes.titleweight": "bold"
})

sigma = 2  # smoothing strength

# extract arrays
x = dfrad_ascent["altitude_ba"].to_numpy()
y = dfrad_ascent["counts_per_s"].to_numpy()
yerr = dfrad_ascent["cps_err"].to_numpy()

# sort by x for clean line + smoothing
idx = np.argsort(x)
x, y, yerr = x[idx], y[idx], yerr[idx]

# smoothed curve
y_smooth = gaussian_filter1d(y, sigma=sigma)

# ---------------- PLOTTING ----------------
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=False)

# faint background line + error band
ax.plot(x, y, color="#0063a9", alpha=0.3, linewidth=1)
plt.subplots_adjust(bottom=0.18)  # increase bottom margin
#ax.fill_between(x, y - yerr, y + yerr, color='lightblue', alpha=0.15)

# scatter with error bars (foreground)
ax.errorbar(
    x, y, yerr=yerr,
    fmt='o',
    markersize=3,
    color="#0063a9",
    ecolor="#00a5b7",
    elinewidth=0.8,
    alpha=0.8,
    capsize=2
)

# smoothed curve (dark blue)
ax.plot(x, y_smooth, color="#1e00c7", linewidth=2)

# labels
ax.set_xlabel("Altitude (km)")
ax.set_ylabel("Counts per second")
ax.set_title("Radiation vs Altitude")

# -------- PRIMARY AXIS (km) --------
ax.set_xlabel("Altitude (km)", labelpad=10)
ax.set_xlim(x.min(), x.max())

xticks_m = ax.get_xticks()
ax.set_xticks(xticks_m)
ax.set_xticklabels((xticks_m / 1000).round(1))

# -------- SECONDARY AXIS (kft) BELOW --------
def m_to_kft(m): return m / 304.8
def kft_to_m(kft): return kft * 304.8

secax = ax.secondary_xaxis('bottom', functions=(m_to_kft, kft_to_m))
secax.spines['bottom'].set_position(('outward', 40))  # push kft axis down
secax.set_xlabel("Altitude (kft)", labelpad=10)

plt.savefig(out_filepath + "rad_vs_alt_errorbarspdf.png", dpi=400, bbox_inches='tight',transparent=True)
plt.show()
'''