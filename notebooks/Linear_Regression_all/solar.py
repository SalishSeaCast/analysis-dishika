#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import gc
from pathlib import Path
import netCDF4 as nc
from datetime import datetime
import re


# In[ ]:


gemlam_dir = "/results/forcing/atmospheric/GEM2.5/gemlam"
operational_dir = "/results/forcing/atmospheric/GEM2.5/operational"
time_fixed_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/TimeFixed"

gemlam_start = datetime(2007, 1, 3)
gemlam_end = datetime(2014, 9, 30)
ops_start = datetime(2014, 10, 1)
ops_end = datetime(2021, 12, 31)

years = range(2007, 2022)

fixed_2008_filenames = {
    "gemlam_y2008m07d16.nc",
    "gemlam_y2008m07d17.nc",
    "gemlam_y2008m07d18.nc",
    "gemlam_y2008m07d19.nc",
    "gemlam_y2008m07d20.nc",
    "gemlam_y2008m07d21.nc",
    "gemlam_y2008m07d22.nc",
    "gemlam_y2008m07d23.nc",
    "gemlam_y2008m08d10.nc",
    "gemlam_y2008m08d11.nc",
    "gemlam_y2008m08d12.nc",
}

def get_file_date(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r"_y(\d{4})m(\d{2})d(\d{2})", filename)
    if match is None:
        return None
    year, month, day = map(int, match.groups())
    return datetime(year, month, day)

all_gemlam_files = glob.glob(os.path.join(gemlam_dir, "*.nc"))
all_ops_files = glob.glob(os.path.join(operational_dir, "ops_y????m??d??.nc"))

selected_gemlam_files = {}

for filepath in all_gemlam_files:
    file_date = get_file_date(filepath)
    if file_date is not None and gemlam_start <= file_date <= gemlam_end:
        selected_gemlam_files[file_date] = filepath

for filename in fixed_2008_filenames:
    fixed_filepath = os.path.join(time_fixed_dir, filename)
    if not os.path.exists(fixed_filepath):
        raise FileNotFoundError(f"Corrected file not found: {fixed_filepath}")
    fixed_date = get_file_date(fixed_filepath)
    if fixed_date is None:
        raise ValueError(f"Could not extract date from: {fixed_filepath}")
    selected_gemlam_files[fixed_date] = fixed_filepath

selected_ops_files = {}

for filepath in all_ops_files:
    file_date = get_file_date(filepath)
    if file_date is not None and ops_start <= file_date <= ops_end:
        selected_ops_files[file_date] = filepath

all_selected_files = {**selected_gemlam_files, **selected_ops_files}

hrdps_files_by_year = {}

for year in years:
    files = [filepath for file_date, filepath in sorted(all_selected_files.items()) if file_date.year == year]
    if len(files) == 0:
        print(f"{year}: no files found")
        continue
    hrdps_files_by_year[year] = files
    print(year, ":", files[0], "to", files[-1], f"({len(files)} files)")


# In[ ]:


# Loading datasets

weights_pre_file = "/home/sallen/MEOPAR/grid/weights-gem2.5-gemlam_201702_pre22sep11.nc"
weights_post_file = "/home/sallen/MEOPAR/grid/weights-gem2.5-gemlam_201702_22sep11onward.nc"
mesh_mask_file = "/ocean/dtaneja/MOAD/analysis-dishika/grid/mesh_mask202108.nc"

ds_weights_pre = xr.open_dataset(weights_pre_file).load()
ds_weights_post = xr.open_dataset(weights_post_file).load()

with xr.open_dataset(mesh_mask_file) as ds_mesh:
    nemo_lat = ds_mesh["nav_lat"].load()
    nemo_lon = ds_mesh["nav_lon"].load()


# In[ ]:


# Converts hourly HRDPS temperature from the 266 × 256 HRDPS grid to the 898 × 398 NEMO grid

transition_time = pd.Timestamp("2011-09-22 00:00:00")

def interpolate_solar_with_weights(solar, ds_weights):
    solar = solar.transpose("time_counter", "y", "x")

    n_time = solar.sizes["time_counter"]
    source_shape = (solar.sizes["y"],solar.sizes["x"])
    n_source_cells = source_shape[0] * source_shape[1]

    source_values = (solar.values.reshape(n_time, n_source_cells).astype(np.float32))
    target_shape = ds_weights["src01"].shape

    interpolated = np.zeros((n_time, target_shape[0], target_shape[1]),dtype=np.float32)

    for n in range(1, 5):
        # The source indexes in the weights file are 1-based
        source_index = (ds_weights[f"src{n:02d}"].values.astype(np.int64)- 1)
        weight = (ds_weights[f"wgt{n:02d}"].values.astype(np.float32))

        interpolated += (source_values[:, source_index]* weight[None, :, :])

    solar_nemo = xr.DataArray(
        interpolated,
        dims=("time_counter", "y", "x"),
        coords={
            "time_counter": solar["time_counter"],
            "nav_lat": (("y", "x"), nemo_lat.values),
            "nav_lon": (("y", "x"), nemo_lon.values),
        },
        name="solar"
    )

    solar_nemo.attrs = solar.attrs.copy()
    solar_nemo.attrs["grid"] = "SalishSeaCast NEMO grid"
    solar_nemo.attrs["interpolation"] = (
        "Four-source weighted HRDPS-to-NEMO interpolation"
    )
    return solar_nemo


# In[ ]:


# Open file and interpolates

def extract_and_interpolate_hourly_solar(file):
    with xr.open_dataset(file) as ds:
        solar = ds["solar"].sortby("time_counter")
        time_index = solar.get_index("time_counter")

        if time_index.has_duplicates:
            unique_mask = ~time_index.duplicated()
            solar = solar.isel(time_counter=unique_mask)

        solar = solar.load()

    times = pd.to_datetime(solar["time_counter"].values)

    before_transition = times < transition_time
    after_transition = times >= transition_time

    pieces = []

    if before_transition.any():
        solar_pre = solar.isel(time_counter=np.where(before_transition)[0])
        interpolated_pre = interpolate_solar_with_weights(solar_pre,ds_weights_pre)
        pieces.append(interpolated_pre)

    if after_transition.any():
        solar_post = solar.isel(time_counter=np.where(after_transition)[0])
        interpolated_post = interpolate_solar_with_weights(solar_post,ds_weights_post)
        pieces.append(interpolated_post)

    solar_nemo = xr.concat(pieces,dim="time_counter").sortby("time_counter")
    return solar_nemo


# In[ ]:


sample_pre_file = hrdps_files_by_year[2008][0]
solar_nemo_pre = extract_and_interpolate_hourly_solar(sample_pre_file)

print("Shape:", solar_nemo_pre.shape)
print("Time range:",solar_nemo_pre.time_counter.values[0],"to",solar_nemo_pre.time_counter.values[-1])


# In[ ]:


sample_post_file = hrdps_files_by_year[2012][0]
solar_nemo_post = extract_and_interpolate_hourly_solar(sample_post_file)

print("Shape:", solar_nemo_post.shape)
print("Time range:",solar_nemo_post.time_counter.values[0],"to",solar_nemo_post.time_counter.values[-1])


# In[ ]:


sample_post_file = hrdps_files_by_year[2015][0]
solar_nemo_post = extract_and_interpolate_hourly_solar(sample_post_file)

print("Shape:", solar_nemo_post.shape)
print("Time range:",solar_nemo_post.time_counter.values[0],"to",solar_nemo_post.time_counter.values[-1])


# In[ ]:


sample_post_file = hrdps_files_by_year[2021][-1]
solar_nemo_post = extract_and_interpolate_hourly_solar(sample_post_file)

print("Shape:", solar_nemo_post.shape)
print("Time range:",solar_nemo_post.time_counter.values[0],"to",solar_nemo_post.time_counter.values[-1])


# In[ ]:


with xr.open_dataset(mesh_mask_file) as ds_mesh:
    water_mask = (ds_mesh["tmask"].isel(t=0, z=0).load().values.astype(bool))
    nemo_lat_2d = ds_mesh["nav_lat"].load().values
    nemo_lon_2d = ds_mesh["nav_lon"].load().values
nemo_j, nemo_i = np.where(water_mask)
water_flat_indices = np.flatnonzero(water_mask.reshape(-1))
nemo_water_lat = nemo_lat_2d[water_mask]
nemo_water_lon = nemo_lon_2d[water_mask]
n_water = len(water_flat_indices)
print("NEMO grid shape:", water_mask.shape)
print("Number of surface water cells:", n_water)


# In[ ]:


def process_daily_file_to_3h_water(file):
    solar_nemo_hourly = extract_and_interpolate_hourly_solar(file)
    n_time = solar_nemo_hourly.sizes["time_counter"]

    solar_water_values = (solar_nemo_hourly.values.reshape(n_time, -1)[:, water_flat_indices].astype(np.float32))
    solar_water_hourly = xr.DataArray(solar_water_values,dims=("time_counter", "water_cell"),coords={
            "time_counter": solar_nemo_hourly["time_counter"].values,
            "water_cell": np.arange(n_water, dtype=np.int32),},name="solar",attrs=solar_nemo_hourly.attrs,)

    solar_water_3h = solar_water_hourly.resample(time_counter="3h",label="left",closed="left",origin="start_day",).mean()
    return solar_water_3h.load()


# In[ ]:


test_file = hrdps_files_by_year[2008][0]

test_3h = process_daily_file_to_3h_water(test_file)

print(test_3h)
print("Shape:", test_3h.shape)
print("Times:", test_3h.time_counter.values)


# In[ ]:


test_file = hrdps_files_by_year[2015][0]

test_3h = process_daily_file_to_3h_water(test_file)

print(test_3h)
print("Shape:", test_3h.shape)
print("Times:", test_3h.time_counter.values)


# In[ ]:


# Run only once for the remaining years
output_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/data/hrdps_nemo_3h"
os.makedirs(output_dir, exist_ok=True)
remaining_years = [2007] + list(range(2013, 2022))
hrdps_nemo_processed_files = []

for year in remaining_years:
    files = hrdps_files_by_year[year]
    print(f"\nProcessing {year}: {len(files)} daily files")
    daily_3h_results = []

    for file_number, file in enumerate(files, start=1):
        daily_3h = process_daily_file_to_3h_water(file)
        daily_3h_results.append(daily_3h)

        if file_number == 1 or file_number % 25 == 0 or file_number == len(files):
            print(f"{year}: processed {file_number}/{len(files)} files")

    solar_year = xr.concat(daily_3h_results, dim="time_counter").sortby("time_counter")
    time_index = solar_year.get_index("time_counter")

    if time_index.has_duplicates:
        solar_year = solar_year.isel(time_counter=~time_index.duplicated())

    ds_year = solar_year.to_dataset(name="solar")
    ds_year = ds_year.assign_coords(nemo_j=("water_cell", nemo_j.astype(np.int32)), nemo_i=("water_cell", nemo_i.astype(np.int32)), nav_lat=("water_cell", nemo_water_lat.astype(np.float32)), nav_lon=("water_cell", nemo_water_lon.astype(np.float32)))
    ds_year.attrs["description"] = "Raw hourly HRDPS solar interpolated onto NEMO surface water cells, then resampled to three-hourly means."

    output_file = f"{output_dir}/HRDPS_NEMO_{year}_solar_3h.nc"
    encoding = {"solar": {"dtype": "float32", "zlib": True, "complevel": 4}}
    ds_year.to_netcdf(output_file, engine="netcdf4", encoding=encoding)
    hrdps_nemo_processed_files.append(output_file)

    print("Saved:", output_file)
    print("Shape:", ds_year["solar"].shape)
    print("Time range:", ds_year.time_counter.values[0], "to", ds_year.time_counter.values[-1])

    del daily_3h_results
    del solar_year
    del ds_year
    gc.collect()


# In[ ]:




