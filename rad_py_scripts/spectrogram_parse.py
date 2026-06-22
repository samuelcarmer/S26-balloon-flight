import xml.etree.ElementTree as ET
import pandas as pd


loc = "/Users/samuelcarmer/Documents/Altig_U_spectrums/spectrums/"
name = "pitchbl"

def out_df(loc,name):
    tree = ET.parse(loc+name+".xml")
    root = tree.getroot()


    spectrum = root.find(".//EnergySpectrum")
    n_channels = int(spectrum.find("NumberOfChannels").text)
    measurement_time = float(spectrum.find("MeasurementTime").text)
    coeffs = [
        float(c.text)
        for c in spectrum.findall(".//EnergyCalibration/Coefficients/Coefficient")
    ]


    counts = [
        int(dp.text)
        for dp in root.findall(".//DataPoint")
    ]

    channels = list(range(len(counts)))

    energy_keV = [
        coeffs[0] + coeffs[1]*ch + coeffs[2]*ch**2
        for ch in channels
    ]
    #print(coeffs)

    df = pd.DataFrame({
        "channel": channels,
        "energy_keV": energy_keV,
        "counts": counts,
        "cps": [round(c / measurement_time,2) for c in counts]
    })
    return df

df = out_df(loc,name)
#print(df.head())

#print(len(df["time_index"]))

#root.find(".//PulseCollection/Pulses").text
#print("working")
out = loc+name
df.to_csv(out+".csv", index=False)

#import os

# Opens with the system default (usually Excel or Numbers)
#os.system(out)