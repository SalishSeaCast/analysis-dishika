#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import arrow
import numpy as np
import pandas as pd
import xarray as xr
import os


# In[ ]:


vsmall = 1e-4


# In[ ]:


def gen_solar(hour, day, latitude, longitude, offset):

    hour = hour + offset

    if hour > 24:
        day = day + 1
        hour = hour - 24
    elif hour < 0:
        day = day - 1
        hour = hour + 24

    ST = 24 / 360.0 * (longitude - (-120))
    day_time = ((hour - 8 - ST) % 24) * 3600
    hour_angle_deg = (day_time / 3600.0 - 12.0) * 15.0
    declination = (23.45* np.pi / 180.0* np.sin((284.0 + day) / 365.25 * 2.0 * np.pi))
    lat = np.pi * latitude / 180.0
    a = np.sin(declination) * np.sin(lat)
    b = np.cos(declination) * np.cos(lat)
    cos_Z = (a+ b * np.cos(np.pi / 180.0 * hour_angle_deg))

    hour_angle = np.tan(lat) * np.tan(declination)
    hour_angle_clipped = np.clip(hour_angle, -1, 1)

    day_length = (np.arccos(-hour_angle_clipped)/ 15.0* 2.0* 180.0/ np.pi )
    day_length = np.where(hour_angle > 1,24 + 1e-7,day_length)
    day_length = np.where(hour_angle < -1,-1e-7,day_length)

    sunrise = 12.0 - 0.5 * day_length
    sunset = 12.0 + 0.5 * day_length

    Q_o = 1368.0
    Qso = Q_o * (1.0+ 0.033 * np.cos(day / 365.25 * 2.0 * np.pi))

    I_incident = Qso * cos_Z
    I_incident = np.where(day_time / 3600.0 < sunrise,0.0,I_incident,)
    I_incident = np.where(day_time / 3600.0 > sunset,0.0,I_incident,)

    I_incident = np.maximum(I_incident, 0.0)

    return I_incident


# In[ ]:


def calculate_max_timeseries(year, month, day, lats, lons):
    ny, nx = lats.shape
    starttime = arrow.get(year, month, day, 0, 0, 0)
    endtime = starttime.shift(days=1)
    deltat = (arrow.get(year, month, day, 0, 10, 0)- arrow.get(year, month, day, 0, 0, 0))
    nsteps = int(86400 / (10 * 60))
    solar = np.zeros((nsteps, ny, nx),dtype=np.float32)
    times = np.zeros(nsteps,dtype="datetime64[s]",)

    time = starttime
    ii = 0

    while time < endtime:

        hour = time.hour + time.minute / 60.0
        yearday = int(time.format("DDDD"))

        solar[ii] = gen_solar(hour,yearday,lats,lons,offset=0)
        times[ii] = np.datetime64(time.naive)
        ii += 1
        time = time + deltat

    da_solar = xr.DataArray(
        solar,
        dims=["time_counter", "y", "x"],
        coords={
            "time_counter": times,
        },
        attrs={
            "description": "Maximum Solar",
            "units": "W/m2",
        },
    )

    # Hourly averages
    maxsolar_1h = (da_solar.resample(time_counter="1h").mean())
    maxsolar_1h = maxsolar_1h.assign_coords(
        time_counter=
        maxsolar_1h.time_counter
        + np.timedelta64(30, "m")
    )

    # 3-hour averages
    maxsolar_3h = (da_solar.resample(time_counter="3h").mean())
    maxsolar_3h = maxsolar_3h.assign_coords(
        time_counter=
        maxsolar_3h.time_counter
        + np.timedelta64(90, "m")
    )

    return maxsolar_1h, maxsolar_3h


# In[ ]:


def make_one_hour_series(threehour,maxsolar_3h,maxsolar_1h):
    predicted_3h = np.asarray(threehour["solar"].values)
    theoretical_3h = np.asarray(maxsolar_3h.values[:8])

    ratio = (predicted_3h/ (theoretical_3h + vsmall))
    ratio = np.clip(ratio,0,1)

    theoretical_1h = np.asarray(maxsolar_1h.values)
    myvalues = np.empty_like(theoretical_1h)
    myvalues[0::3] = (ratio * theoretical_1h[0::3])
    myvalues[1::3] = (ratio * theoretical_1h[1::3])
    myvalues[2::3] = (ratio * theoretical_1h[2::3])

    return myvalues


# In[ ]:


def process_one_day(year, month, day):
    input_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/predictions/daily_downscaled_2008_padded"
    output_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/predictions/hourly_downscaled_2008"
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, f"downscaled_y{year}m{month:02d}d{day:02d}.nc")
    print("Opening:", input_file)
    with xr.open_dataset(input_file) as ds_open:
        threehour = ds_open.load()

    print(threehour)

    if "nav_lat" in threehour:
        lats = threehour["nav_lat"].values
    else:
        raise ValueError(
            "nav_lat was not found in the predicted file."
        )

    if "nav_lon" in threehour:
        lons = threehour["nav_lon"].values
    else:
        raise ValueError(
            "nav_lon was not found in the predicted file."
        )

    print("Grid shape:", lats.shape)

    maxsolar_1h, maxsolar_3h = (calculate_max_timeseries(year,month,day,lats,lons))

    hourly_solar = make_one_hour_series(threehour,maxsolar_3h,maxsolar_1h)
    print("3-hour solar shape:",threehour["solar"].shape)
    print("Hourly solar shape:",hourly_solar.shape)

    ds_hourly = xr.Dataset(
    data_vars={"solar": (["time_counter", "y", "x"], hourly_solar)},
    coords={
        "time_counter": maxsolar_1h.time_counter.values,
        "y": threehour["y"].values,
        "x": threehour["x"].values,
        "nav_lat": (["y", "x"], lats),
        "nav_lon": (["y", "x"], lons),
    },
    attrs={
        "description": "Hourly solar derived from statistically downscaled 3-hourly solar",
        "Comment": "Hourly solar based on theoretical maximum solar and predicted 3-hourly solar"
    })

    ds_hourly["solar"].attrs["units"] = "W/m2"

    output_file = os.path.join(output_dir, f"downscaled_y{year}m{month:02d}d{day:02d}.nc")

    encoding = {"solar": {"zlib": True}}

    ds_hourly.to_netcdf(output_file, unlimited_dims=["time_counter"], encoding=encoding)

    print("Saved:", output_file)

    return ds_hourly


# In[ ]:


# hourly_test = process_one_day(
#     2008,
#     1,
#     1,
# )

# print(hourly_test)


# In[ ]:


# test_file = (
#     "/ocean/dtaneja/MOAD/analysis-dishika/"
#     "notebooks/results/daily_downscaled_2008_padded/"
#     "downscaled_y2008m01d01.nc"
# )

# ds = xr.open_dataset(test_file)

# print(ds)
# print()
# print("solar shape:", ds["solar"].shape)
# print("solar times:")
# print(ds.time_counter.values)

# print()
# print("nav_lat:", ds["nav_lat"].shape)
# print("nav_lon:", ds["nav_lon"].shape)

# print()
# print("Solar min:", ds["solar"].min().values)
# print("Solar max:", ds["solar"].max().values)


# In[ ]:


def process_whole_year(year):

    input_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/predictions/daily_downscaled_2008_padded"
    output_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/predictions/hourly_downscaled_2008"

    os.makedirs(output_dir, exist_ok=True)

    startdate = arrow.get(year, 1, 1, 0, 0, 0)
    enddate = startdate.shift(years=1)
    time = startdate

    while time < enddate:

        input_file = os.path.join(input_dir, f"downscaled_y{time.year}m{time.month:02d}d{time.day:02d}.nc")
        output_file = os.path.join(output_dir, f"downscaled_y{time.year}m{time.month:02d}d{time.day:02d}.nc")
        if not os.path.exists(input_file):
            print("Missing file, skipping:", input_file)
            time = time.shift(days=1)
            continue

        print("Processing:", input_file)

        with xr.open_dataset(input_file) as ds_open:
            threehour = ds_open.load()

        lats = threehour["nav_lat"].values
        lons = threehour["nav_lon"].values

        maxsolar_1h, maxsolar_3h = calculate_max_timeseries(time.year, time.month, time.day, lats, lons)

        hourly_solar = make_one_hour_series(threehour, maxsolar_3h, maxsolar_1h)

        ds_hourly = xr.Dataset(
            data_vars={"solar": (["time_counter", "y", "x"], hourly_solar)},
            coords={
                "time_counter": maxsolar_1h.time_counter.values,
                "y": threehour["y"].values,
                "x": threehour["x"].values,
                "nav_lat": (["y", "x"], lats),
                "nav_lon": (["y", "x"], lons),
            },
            attrs={
                "description": "Hourly solar derived from statistically downscaled 3-hourly solar",
                "Comment": "Hourly solar based on theoretical maximum solar and predicted 3-hourly solar",
            },
        )

        ds_hourly["solar"].attrs["units"] = "W/m2"

        encoding = {"solar": {"zlib": True}}

        ds_hourly.to_netcdf(output_file, unlimited_dims=["time_counter"], encoding=encoding)

        ds_hourly.close()

        print("Saved:", output_file)

        time = time.shift(days=1)


# In[ ]:


process_whole_year(2008)

