import pandas as pd

import os
#print(os.getcwd())

in_filepath = "/Users/samuelcarmer/Documents/Balloon_SRS/iMet/Data_files/"



def load_log():

    # read whitespace-delimited .LOG
    df = pd.read_csv(
        in_filepath+"001_001.LOG",
        sep=r"\s+",
        header=None,
        engine="python"
    )

    # assign provisional names
    df.columns = [      
        "temp_C",                
        "temp_virt",                
        "rh_percent",
        "pressure_hPa",
        "altitude_ba",
        "col6",
        "col7",
        "wind_dir_rad",
        "speed_knot",
        "slant_range",
        "status",
        "latitude",
        "longitude",
        "temp_C_raw",
        "altitude_gps",
        "record_time",
        "height_status"
    ]

    start_time = pd.to_datetime("2026-03-12 16:26:00")

    df["time_dt"] = start_time + pd.to_timedelta(df.index, unit="s")
    df["time_str"] = df["time_dt"].dt.strftime("%H:%M:%S")

    df["time_s"] = pd.to_timedelta(df.index, unit="s")
    df["time_s"] = df["time_s"].dt.total_seconds()


    return df, start_time


def clean_df(df):
    return df[[
    "temp_C",
    "rh_percent",
    "pressure_hPa",
    "altitude_ba",
    "latitude",
    "longitude",
    "altitude_gps",
    "time_str",
    "time_s", 
    "time_dt"
    ]]

def df_wind(df):
    return df[[
        "temp_C",
        "pressure_hPa",
        "altitude_ba",
        "wind_dir_rad",
        "speed_knot",
        "slant_range",
        "latitude",
        "longitude",
        "altitude_gps",
        "record_time",
        "height_status"
    ]]





