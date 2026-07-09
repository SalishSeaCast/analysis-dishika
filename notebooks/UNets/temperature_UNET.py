#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from tensorflow.keras import layers, models


# In[ ]:


hrdps = sorted(glob.glob('/results/forcing/atmospheric/GEM2.5/gemlam/gemlam_y????m??d??.nc'))
hrdps


# In[ ]:


years = [2008, 2009, 2010, 2011, 2012]
hrdps_files_by_year = {}
for year in years:
    files = sorted(glob.glob(f"/results/forcing/atmospheric/GEM2.5/gemlam/"f"gemlam_y{year}m??d??.nc"))
    hrdps_files_by_year[year] = files
    print(year, ":", files[0], files[-1])


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

def interpolate_tair_with_weights(tair, ds_weights):
    tair = tair.transpose("time_counter", "y", "x")

    n_time = tair.sizes["time_counter"]
    source_shape = (tair.sizes["y"],tair.sizes["x"])
    n_source_cells = source_shape[0] * source_shape[1]

    source_values = (tair.values.reshape(n_time, n_source_cells).astype(np.float32))
    target_shape = ds_weights["src01"].shape

    interpolated = np.zeros((n_time, target_shape[0], target_shape[1]),dtype=np.float32)

    for n in range(1, 5):
        # The source indexes in the weights file are 1-based
        source_index = (ds_weights[f"src{n:02d}"].values.astype(np.int64)- 1)
        weight = (ds_weights[f"wgt{n:02d}"].values.astype(np.float32))

        interpolated += (source_values[:, source_index]* weight[None, :, :])

    tair_nemo = xr.DataArray(
        interpolated,
        dims=("time_counter", "y", "x"),
        coords={
            "time_counter": tair["time_counter"],
            "nav_lat": (("y", "x"), nemo_lat.values),
            "nav_lon": (("y", "x"), nemo_lon.values),
        },
        name="tair"
    )

    tair_nemo.attrs = tair.attrs.copy()
    tair_nemo.attrs["grid"] = "SalishSeaCast NEMO grid"
    tair_nemo.attrs["interpolation"] = (
        "Four-source weighted HRDPS-to-NEMO interpolation"
    )
    return tair_nemo


# In[ ]:


# Open file and interpolates

def extract_and_interpolate_hourly_tair(file):
    with xr.open_dataset(file) as ds:
        tair = ds["tair"].sortby("time_counter")
        time_index = tair.get_index("time_counter")

        if time_index.has_duplicates:
            unique_mask = ~time_index.duplicated()
            tair = tair.isel(time_counter=unique_mask)

        tair = tair.load()

    times = pd.to_datetime(tair["time_counter"].values)

    before_transition = times < transition_time
    after_transition = times >= transition_time

    pieces = []

    if before_transition.any():
        tair_pre = tair.isel(time_counter=np.where(before_transition)[0])
        interpolated_pre = interpolate_tair_with_weights(tair_pre,ds_weights_pre)
        pieces.append(interpolated_pre)

    if after_transition.any():
        tair_post = tair.isel(time_counter=np.where(after_transition)[0])
        interpolated_post = interpolate_tair_with_weights(tair_post,ds_weights_post)
        pieces.append(interpolated_post)

    tair_nemo = xr.concat(pieces,dim="time_counter").sortby("time_counter")
    return tair_nemo


# In[ ]:


sample_pre_file = hrdps_files_by_year[2008][0]
tair_nemo_pre = extract_and_interpolate_hourly_tair(sample_pre_file)

print("Shape:", tair_nemo_pre.shape)
print("Time range:",tair_nemo_pre.time_counter.values[0],"to",tair_nemo_pre.time_counter.values[-1])


# In[ ]:


sample_post_file = hrdps_files_by_year[2012][0]
tair_nemo_post = extract_and_interpolate_hourly_tair(sample_post_file)

print("Shape:", tair_nemo_post.shape)
print("Time range:",tair_nemo_post.time_counter.values[0],"to",tair_nemo_post.time_counter.values[-1])


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
    tair_nemo_hourly = extract_and_interpolate_hourly_tair(file)
    n_time = tair_nemo_hourly.sizes["time_counter"]

    tair_water_values = (tair_nemo_hourly.values.reshape(n_time, -1)[:, water_flat_indices].astype(np.float32))
    tair_water_hourly = xr.DataArray(tair_water_values,dims=("time_counter", "water_cell"),coords={
            "time_counter": tair_nemo_hourly["time_counter"].values,
            "water_cell": np.arange(n_water, dtype=np.int32),},name="tair",attrs=tair_nemo_hourly.attrs,)

    tair_water_3h = tair_water_hourly.resample(time_counter="3h",label="left",closed="left",origin="start_day",).mean()
    return tair_water_3h.load()


# In[ ]:


test_file = hrdps_files_by_year[2008][0]

test_3h = process_daily_file_to_3h_water(test_file)

print(test_3h)
print("Shape:", test_3h.shape)
print("Times:", test_3h.time_counter.values)


# In[ ]:


hrdps_nemo_files = sorted(glob.glob("/ocean/dtaneja/MOAD/analysis-dishika/notebooks/data/hrdps_nemo_3h/HRDPS_NEMO_*_tair_3h.nc"))
ds_hrdps_nemo = xr.open_mfdataset(hrdps_nemo_files,combine="by_coords")

ds_hrdps_nemo = ds_hrdps_nemo.sortby("time_counter")

print(ds_hrdps_nemo)
print("First time:", ds_hrdps_nemo.time_counter.values[0])
print("Last time: ", ds_hrdps_nemo.time_counter.values[-1])
print("Shape:", ds_hrdps_nemo["tair"].shape)


# In[ ]:


# Validation: 2008
ds_hrdps_val = ds_hrdps_nemo.sel(time_counter=slice("2008-01-01", "2008-12-31 23:59:59"))
# Training: 2009–2011
ds_hrdps_train = ds_hrdps_nemo.sel(time_counter=slice("2009-01-01", "2011-12-31 23:59:59"))
# Testing: 2012
ds_hrdps_test = ds_hrdps_nemo.sel(time_counter=slice("2012-01-01", "2012-12-31 23:59:59"))


# In[ ]:


canrcm_years = [2008, 2009, 2010, 2011, 2012]
canrcm_tas_files = []

for year in canrcm_years:
    matches = sorted(glob.glob(f"/results/forcing/CanRCM5/*_{year}01_{year}12_3h_tas.nc"))
    canrcm_tas_files.append(matches[0])

print("CanRCM files:")
for file in canrcm_tas_files:
    print(file)


# In[ ]:


from datetime import timedelta
ds_canrcm = xr.open_mfdataset(canrcm_tas_files,combine="by_coords")
ds_canrcm = ds_canrcm.sortby("time")
shifted_time = np.array([time_value - timedelta(hours=3)for time_value in ds_canrcm["time"].values])
ds_canrcm = ds_canrcm.assign_coords(time=("time", shifted_time))

print(ds_canrcm)
print("First timestamp:", ds_canrcm.time.values[0])
print("Last timestamp: ", ds_canrcm.time.values[-1])


# In[ ]:


nemo_water_lat = ds_hrdps_nemo["nav_lat"].values
nemo_water_lon = ds_hrdps_nemo["nav_lon"].values

nemo_water_lon_normalized = ((nemo_water_lon + 180) % 360) - 180

lat_min = float(np.nanmin(nemo_water_lat))
lat_max = float(np.nanmax(nemo_water_lat))

lon_min = float(np.nanmin(nemo_water_lon_normalized))
lon_max = float(np.nanmax(nemo_water_lon_normalized))

print("NEMO latitude range:", lat_min, "to", lat_max)
print("NEMO longitude range:", lon_min, "to", lon_max)


# In[ ]:


lat_canrcm = ds_canrcm["lat"]
lon_canrcm = ds_canrcm["lon"]

lon_canrcm_normalized = ((lon_canrcm + 180) % 360) - 180
canrcm_region_mask = ((lat_canrcm >= lat_min)& (lat_canrcm <= lat_max)
                      & (lon_canrcm_normalized >= lon_min) & (lon_canrcm_normalized <= lon_max))

rlat_indices, rlon_indices = np.where(canrcm_region_mask.values)

rlat_min = rlat_indices.min()
rlat_max = rlat_indices.max()

rlon_min = rlon_indices.min()
rlon_max = rlon_indices.max()

print("CanRCM rlat indices:", rlat_min, "to", rlat_max)
print("CanRCM rlon indices:", rlon_min, "to", rlon_max)

ds_canrcm_cut = (ds_canrcm[["tas"]].isel(rlat=slice(rlat_min, rlat_max + 1),rlon=slice(rlon_min, rlon_max + 1)))
print(ds_canrcm_cut)


# In[ ]:


# Validation: 2008
ds_canrcm_val = ds_canrcm_cut.sel(time=slice("2008-01-01","2008-12-31 23:59:59"))
# Training: 2009–2011
ds_canrcm_train = ds_canrcm_cut.sel(time=slice("2009-01-01","2011-12-31 23:59:59"))
# Testing: 2012
ds_canrcm_test = ds_canrcm_cut.sel(time=slice("2012-01-01","2012-12-31 23:59:59"))


# In[ ]:


def time_to_string(value):
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value,unit="s").replace("T", " ")

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    raise TypeError(f"Unsupported time type: {type(value)}")


def align_time_pair(ds_canrcm_split, ds_hrdps_split):
    canrcm_times = np.array([time_to_string(t)for t in ds_canrcm_split["time"].values])
    hrdps_times = np.array([time_to_string(t)for t in ds_hrdps_split["time_counter"].values])
    common_times, canrcm_idx, hrdps_idx = np.intersect1d(canrcm_times,hrdps_times,return_indices=True)
    ds_canrcm_aligned = ds_canrcm_split.isel(time=canrcm_idx)
    ds_hrdps_aligned = ds_hrdps_split.isel(time_counter=hrdps_idx)
    return ds_canrcm_aligned, ds_hrdps_aligned


# In[ ]:


ds_canrcm_val, ds_hrdps_val = align_time_pair(ds_canrcm_val,ds_hrdps_val)
ds_canrcm_train, ds_hrdps_train = align_time_pair(ds_canrcm_train,ds_hrdps_train)
ds_canrcm_test, ds_hrdps_test = align_time_pair(ds_canrcm_test,ds_hrdps_test)


# In[ ]:


print("Validation:")
print("HRDPS:", ds_hrdps_val.sizes["time_counter"])
print("CanRCM:", ds_canrcm_val.sizes["time"])

print("\nTraining:")
print("HRDPS:", ds_hrdps_train.sizes["time_counter"])
print("CanRCM:", ds_canrcm_train.sizes["time"])

print("\nTesting:")
print("HRDPS:", ds_hrdps_test.sizes["time_counter"])
print("CanRCM:", ds_canrcm_test.sizes["time"])

assert ds_hrdps_val.sizes["time_counter"] == ds_canrcm_val.sizes["time"]
assert ds_hrdps_train.sizes["time_counter"] == ds_canrcm_train.sizes["time"]
assert ds_hrdps_test.sizes["time_counter"] == ds_canrcm_test.sizes["time"]


# In[ ]:


# Confirm that every CanRCM timestamp matches the corresponding HRDPS timestamp

def confirm_time_alignment(ds_canrcm_split, ds_hrdps_split, split_name):
    canrcm_times = np.array([
        time_to_string(time_value)
        for time_value in ds_canrcm_split["time"].values
    ])

    hrdps_times = np.array([
        time_to_string(time_value)
        for time_value in ds_hrdps_split["time_counter"].values
    ])

    print(f"\n{split_name}")
    print("CanRCM shape:", ds_canrcm_split["tas"].shape)
    print("HRDPS shape:", ds_hrdps_split["tair"].shape)
    print("Exact timestamp match:", np.array_equal(canrcm_times, hrdps_times))
    print("First timestamp:", canrcm_times[0])
    print("Last timestamp: ", canrcm_times[-1])

    assert np.array_equal(canrcm_times, hrdps_times)


confirm_time_alignment(
    ds_canrcm_train,
    ds_hrdps_train,
    "Training"
)

confirm_time_alignment(
    ds_canrcm_val,
    ds_hrdps_val,
    "Validation"
)

confirm_time_alignment(
    ds_canrcm_test,
    ds_hrdps_test,
    "Testing"
)


# In[ ]:


ds_hrdps_train["tair"].shape


# In[ ]:


H, W = water_mask.shape
print("NEMO grid size:", H, W)
print("Number of water cells:", water_mask.sum())

H_lr = ds_canrcm_train["tas"].squeeze(drop=True).shape[1]
W_lr = ds_canrcm_train["tas"].squeeze(drop=True).shape[2]
print("CanRCM grid size:", H_lr, W_lr)


# In[ ]:


def next_multiple(value, multiple=16):
    return int(np.ceil(value / multiple) * multiple)

H_pad = next_multiple(H, 16)
W_pad = next_multiple(W, 16)

pad_y = H_pad - H
pad_x = W_pad - W

print("Original HRDPS size:", H, W)
print("Padded U-Net size:", H_pad, W_pad)
print("Padding:", pad_y, pad_x)


# In[ ]:


canrcm_train_tas = ds_canrcm_train["tas"].squeeze(drop=True)

input_mean = float(canrcm_train_tas.mean().compute())
input_std = float(canrcm_train_tas.std().compute())

hrdps_train_tair = ds_hrdps_train["tair"]

output_mean = float(hrdps_train_tair.mean(skipna=True).compute())
output_std = float(hrdps_train_tair.std(skipna=True).compute())

print("Input mean:", input_mean)
print("Input std:", input_std)
print("Output mean:", output_mean)
print("Output std:", output_std)


# In[ ]:


X_train_lr = ds_canrcm_train["tas"].squeeze(drop=True).transpose("time", "rlat", "rlon").values.astype(np.float32)
X_val_lr   = ds_canrcm_val["tas"].squeeze(drop=True).transpose("time", "rlat", "rlon").values.astype(np.float32)
X_test_lr  = ds_canrcm_test["tas"].squeeze(drop=True).transpose("time", "rlat", "rlon").values.astype(np.float32)

Y_train_water = (ds_hrdps_train["tair"].transpose("time_counter", "water_cell").values.astype(np.float32))
Y_val_water = (ds_hrdps_val["tair"].transpose("time_counter", "water_cell").values.astype(np.float32))
Y_test_water = (ds_hrdps_test["tair"].transpose("time_counter", "water_cell").values.astype(np.float32))

print("X_train_lr:", X_train_lr.shape)
print("Y_train_water:", Y_train_water.shape)

print("X_val_lr:", X_val_lr.shape)
print("Y_val_water:", Y_val_water.shape)

print("X_test_lr:", X_test_lr.shape)
print("Y_test_water:", Y_test_water.shape)


# In[ ]:


water_scatter_indices_tf = tf.constant(water_flat_indices[:, np.newaxis],dtype=tf.int32)
water_mask_float_tf = tf.constant(water_mask.astype(np.float32)[..., np.newaxis],dtype=tf.float32)
input_mean_tf = tf.constant(input_mean, dtype=tf.float32)
input_std_tf = tf.constant(input_std, dtype=tf.float32)
output_mean_tf = tf.constant(output_mean, dtype=tf.float32)
output_std_tf = tf.constant(output_std, dtype=tf.float32)


# In[ ]:


def prepare_example(x_lr, y_water):
    x_lr = tf.cast(x_lr, tf.float32)
    y_water = tf.cast(y_water, tf.float32)
    x_lr = tf.where(tf.math.is_finite(x_lr),x_lr,input_mean_tf)
    x_lr = (x_lr - input_mean_tf) / input_std_tf
    x_lr = x_lr[..., tf.newaxis]
    x_nemo = tf.image.resize(x_lr,size=(H, W),method="bilinear")
    x_nemo = x_nemo * water_mask_float_tf
    x_nemo = tf.pad(x_nemo,paddings=[[0, pad_y],[0, pad_x],[0, 0]],constant_values=0.0)
    valid_water = tf.math.is_finite(y_water)
    y_water_norm = (y_water - output_mean_tf) / output_std_tf
    y_water_norm = tf.where(valid_water,y_water_norm,0.0)
    y_flat = tf.scatter_nd(indices=water_scatter_indices_tf,updates=y_water_norm,shape=[H * W])
    valid_flat = tf.scatter_nd(indices=water_scatter_indices_tf,updates=tf.cast(valid_water, tf.float32),shape=[H * W])
    y_grid = tf.reshape(y_flat,(H, W, 1))
    valid_grid = tf.reshape(valid_flat,(H, W, 1))
    y_grid = tf.pad(y_grid,paddings=[[0, pad_y],[0, pad_x],[0, 0]],constant_values=0.0)
    valid_grid = tf.pad(valid_grid,paddings=[[0, pad_y],[0, pad_x],[0, 0]],constant_values=0.0)
    y_with_mask = tf.concat([y_grid, valid_grid],axis=-1)
    return x_nemo, y_with_mask


# In[ ]:


batch_size = 1

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_lr, Y_train_water)
)

train_ds = train_ds.shuffle(
    buffer_size=512,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    prepare_example,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


# In[ ]:


val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val_lr, Y_val_water)
)

val_ds = val_ds.map(
    prepare_example,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(batch_size)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# In[ ]:


test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test_lr, Y_test_water)
)

test_ds = test_ds.map(
    prepare_example,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# In[ ]:


for x_batch, y_batch in train_ds.take(1):
    print("U-Net input shape:", x_batch.shape)
    print("Target and mask shape:", y_batch.shape)
    print(
        "Valid water cells:",
        tf.reduce_sum(y_batch[..., 1]).numpy()
    )


# In[ ]:


def masked_mse(y_true, y_pred):
    target = y_true[..., 0:1]
    mask = y_true[..., 1:2]

    squared_error = tf.square(y_pred - target)
    squared_error = squared_error * mask

    return tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + 1e-8)


def masked_mae(y_true, y_pred):
    target = y_true[..., 0:1]
    mask = y_true[..., 1:2]

    absolute_error = tf.abs(y_pred - target)
    absolute_error = absolute_error * mask

    return tf.reduce_sum(absolute_error) / (tf.reduce_sum(mask) + 1e-8)


def masked_bias(y_true, y_pred):
    target = y_true[..., 0:1]
    mask = y_true[..., 1:2]

    error = y_pred - target
    error = error * mask

    return tf.reduce_sum(error) / (tf.reduce_sum(mask) + 1e-8)


# In[ ]:


def conv_block(x, filters):
    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_unet(input_shape=(272, 256, 1), base_filters=16):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, base_filters * 16)

    # Decoder
    u4 = layers.UpSampling2D(size=(2, 2))(b)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8)

    u3 = layers.UpSampling2D(size=(2, 2))(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = layers.UpSampling2D(size=(2, 2))(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = layers.UpSampling2D(size=(2, 2))(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    # Output: one temperature value per grid cell
    outputs = layers.Conv2D(1, kernel_size=1, padding="same")(c8)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


# In[ ]:


model = build_unet(
    input_shape=(H_pad, W_pad, 1),
    base_filters=8
)
model.summary()


# In[ ]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss=masked_mse,
    metrics=[
        masked_mae,
        masked_bias,
    ],
)


# In[ ]:


output_dir = "/ocean/dtaneja/MOAD/analysis-dishika/notebooks/UNets"
checkpoint_path = (f"{output_dir}/unet_canrcm_to_nemo_best.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]


# In[ ]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
)


# In[ ]:


history_df = pd.DataFrame(history.history)
history_df.to_csv(f"{output_dir}/unet_training_history.csv", index=False)


# In[ ]:


from pathlib import Path

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

epochs_ran = range(
    1,
    len(history.history["loss"]) + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_ran,
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    epochs_ran,
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Masked MSE loss")
plt.title("NEMO U-Net Training and Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    output_dir / "unet_training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


best_model = tf.keras.models.load_model(
    checkpoint_path,
    custom_objects={
        "masked_mse": masked_mse,
        "masked_mae": masked_mae,
        "masked_bias": masked_bias
    }
)


# In[ ]:


n_test_times = Y_test_water.shape[0]

pred_test_file = output_dir / "pred_test_water.npy"

pred_test_water = np.lib.format.open_memmap(
    pred_test_file,
    mode="w+",
    dtype=np.float32,
    shape=Y_test_water.shape
)

start_index = 0

for batch_number, (x_batch, _) in enumerate(test_ds):

    pred_norm_padded = best_model(
        x_batch,
        training=False
    ).numpy()

    # Remove padding and channel
    pred_norm_grid = pred_norm_padded[
        :,
        :H,
        :W,
        0
    ]

    # Convert back to original temperature units
    pred_grid = (
        pred_norm_grid * output_std
        + output_mean
    )

    # Extract only NEMO water cells
    pred_water_batch = pred_grid.reshape(
        pred_grid.shape[0],
        -1
    )[:, water_flat_indices]

    current_batch_size = pred_water_batch.shape[0]

    pred_test_water[
        start_index:
        start_index + current_batch_size
    ] = pred_water_batch

    start_index += current_batch_size

    if batch_number % 100 == 0:
        print(
            f"Predicted {start_index} of "
            f"{n_test_times} timestamps"
        )

pred_test_water.flush()

print("Predicted test-water shape:")
print(pred_test_water.shape)


# In[ ]:


true_test_water = Y_test_water

n_times = true_test_water.shape[0]

actual_mean_time = np.full(n_times, np.nan)
pred_mean_time = np.full(n_times, np.nan)

rmse_each_time = np.full(n_times, np.nan)
mae_each_time = np.full(n_times, np.nan)
bias_each_time = np.full(n_times, np.nan)

total_count = 0
total_squared_error = 0.0
total_absolute_error = 0.0
total_error = 0.0

total_actual = 0.0
total_actual_squared = 0.0

chunk_size = 32

for start in range(0, n_times, chunk_size):

    end = min(start + chunk_size, n_times)

    actual_chunk = np.asarray(
        true_test_water[start:end],
        dtype=np.float64
    )

    predicted_chunk = np.asarray(
        pred_test_water[start:end],
        dtype=np.float64
    )

    valid_chunk = (
        np.isfinite(actual_chunk)
        & np.isfinite(predicted_chunk)
    )

    error_chunk = np.where(
        valid_chunk,
        predicted_chunk - actual_chunk,
        np.nan
    )

    actual_mean_time[start:end] = np.nanmean(
        np.where(valid_chunk, actual_chunk, np.nan),
        axis=1
    )

    pred_mean_time[start:end] = np.nanmean(
        np.where(valid_chunk, predicted_chunk, np.nan),
        axis=1
    )

    rmse_each_time[start:end] = np.sqrt(
        np.nanmean(error_chunk**2, axis=1)
    )

    mae_each_time[start:end] = np.nanmean(
        np.abs(error_chunk),
        axis=1
    )

    bias_each_time[start:end] = np.nanmean(
        error_chunk,
        axis=1
    )

    valid_errors = error_chunk[valid_chunk]
    valid_actual = actual_chunk[valid_chunk]

    total_count += valid_errors.size
    total_squared_error += np.sum(valid_errors**2)
    total_absolute_error += np.sum(np.abs(valid_errors))
    total_error += np.sum(valid_errors)

    total_actual += np.sum(valid_actual)
    total_actual_squared += np.sum(valid_actual**2)


# In[ ]:


rmse = np.sqrt(
    total_squared_error / total_count
)

mae = (
    total_absolute_error / total_count
)

bias = (
    total_error / total_count
)

ss_tot = (
    total_actual_squared
    - (total_actual**2 / total_count)
)

r2 = 1 - total_squared_error / ss_tot

print("Final 2012 NEMO Water-Cell Test Metrics:")
print("RMSE:", rmse)
print("MAE:", mae)
print("Bias:", bias)
print("R²:", r2)


# In[ ]:


test_metrics = pd.DataFrame({
    "year": [2012],
    "rmse": [rmse],
    "mae": [mae],
    "bias": [bias],
    "r2": [r2],
    "number_valid_values": [total_count]
})

test_metrics.to_csv(
    output_dir / "unet_nemo_test_metrics_2012.csv",
    index=False
)


# In[ ]:


def water_vector_to_nemo_map(water_values):

    full_flat = np.full(
        H * W,
        np.nan,
        dtype=np.float32
    )

    full_flat[water_flat_indices] = water_values

    return full_flat.reshape(H, W)


# In[ ]:


lon_plot = nemo_lon_2d
lat_plot = nemo_lat_2d

print("Longitude shape:", lon_plot.shape)
print("Latitude shape:", lat_plot.shape)


# In[ ]:


test_times = pd.to_datetime(
    ds_hrdps_test["time_counter"].values
)

time_index = 0

actual_map = water_vector_to_nemo_map(
    true_test_water[time_index]
)

predicted_map = water_vector_to_nemo_map(
    pred_test_water[time_index]
)

error_map = predicted_map - actual_map

plot_time = test_times[time_index]

vmin = min(
    np.nanmin(actual_map),
    np.nanmin(predicted_map)
)

vmax = max(
    np.nanmax(actual_map),
    np.nanmax(predicted_map)
)

err_abs = np.nanmax(
    np.abs(error_map)
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

p0 = axes[0].pcolormesh(
    lon_plot,
    lat_plot,
    actual_map,
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

axes[0].set_title("Actual HRDPS tair on NEMO grid")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

plt.colorbar(
    p0,
    ax=axes[0],
    label="tair"
)

p1 = axes[1].pcolormesh(
    lon_plot,
    lat_plot,
    predicted_map,
    shading="auto",
    vmin=vmin,
    vmax=vmax
)

axes[1].set_title("U-Net predicted tair")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.colorbar(
    p1,
    ax=axes[1],
    label="tair"
)

p2 = axes[2].pcolormesh(
    lon_plot,
    lat_plot,
    error_map,
    shading="auto",
    cmap="coolwarm",
    vmin=-err_abs,
    vmax=err_abs
)

axes[2].set_title("Error: prediction − actual")
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")

plt.colorbar(
    p2,
    ax=axes[2],
    label="prediction − actual"
)

plt.suptitle(
    f"NEMO U-Net prediction comparison: {plot_time}",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    output_dir
    / f"unet_nemo_spatial_{plot_time:%Y%m%d_%H%M}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


spatial_results = pd.DataFrame({
    "water_cell": np.arange(len(water_flat_indices)),
    "nemo_j": nemo_j,
    "nemo_i": nemo_i,
    "latitude": nemo_lat_2d[water_mask],
    "longitude": nemo_lon_2d[water_mask],
    "actual": true_test_water[time_index],
    "predicted": pred_test_water[time_index],
    "error": (
        pred_test_water[time_index]
        - true_test_water[time_index]
    )
})

spatial_results.to_csv(
    output_dir
    / f"unet_nemo_spatial_{plot_time:%Y%m%d_%H%M}.csv",
    index=False
)


# In[ ]:


plt.figure(figsize=(14, 5))

plt.plot(
    test_times,
    actual_mean_time,
    label="Actual HRDPS on NEMO grid"
)

plt.plot(
    test_times,
    pred_mean_time,
    label="U-Net prediction",
    alpha=0.6
)

plt.xlabel("Time")
plt.ylabel("NEMO water-cell mean tair")

plt.title(
    "NEMO Water-Cell Mean Actual vs Predicted Temperature, 2012"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    output_dir / "unet_nemo_domain_mean_timeseries_2012.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


domain_timeseries = pd.DataFrame({
    "time": test_times,
    "actual_water_mean": actual_mean_time,
    "predicted_water_mean": pred_mean_time,
    "error_predicted_minus_actual": (
        pred_mean_time - actual_mean_time
    ),
    "rmse": rmse_each_time,
    "mae": mae_each_time,
    "bias": bias_each_time
})

domain_timeseries.to_csv(
    output_dir
    / "unet_nemo_domain_timeseries_2012.csv",
    index=False
)


# In[ ]:


total_values = (
    true_test_water.shape[0]
    * true_test_water.shape[1]
)

sample_size = min(
    200000,
    total_values
)

rng = np.random.default_rng(42)

sample_idx = rng.choice(
    total_values,
    size=sample_size,
    replace=False
)

actual_flat = true_test_water.reshape(-1)
predicted_flat = pred_test_water.reshape(-1)

true_sample = actual_flat[sample_idx]
pred_sample = predicted_flat[sample_idx]

valid_sample = (
    np.isfinite(true_sample)
    & np.isfinite(pred_sample)
)

true_sample = true_sample[valid_sample]
pred_sample = pred_sample[valid_sample]


# In[ ]:


plt.figure(figsize=(6, 6))

plt.scatter(
    true_sample,
    pred_sample,
    s=1,
    alpha=0.2
)

min_val = min(
    np.nanmin(true_sample),
    np.nanmin(pred_sample)
)

max_val = max(
    np.nanmax(true_sample),
    np.nanmax(pred_sample)
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual HRDPS tair on NEMO grid")
plt.ylabel("U-Net predicted tair")
plt.title("Actual vs Predicted NEMO Water Temperature")
plt.grid()
plt.tight_layout()

plt.savefig(
    output_dir / "unet_nemo_actual_vs_predicted_2012.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


daily_errors = pd.DataFrame({
    "time": test_times,
    "date": test_times.normalize(),
    "rmse": rmse_each_time,
    "mae": mae_each_time,
    "bias": bias_each_time
})

daily_errors = (
    daily_errors
    .groupby("date")
    .mean(numeric_only=True)
)

daily_errors.to_csv(
    output_dir / "unet_nemo_daily_errors_2012.csv"
)

daily_errors.head()


# In[ ]:


stations = {
    "Sand Heads": (426, 293),
    "Halibut Bank": (503, 261),
    "Sentry Shoal": (707, 145)
}


# In[ ]:


flat_to_water = np.full(
    H * W,
    -1,
    dtype=np.int32
)

flat_to_water[water_flat_indices] = np.arange(
    len(water_flat_indices),
    dtype=np.int32
)


# In[ ]:


station_results = []
station_timeseries = pd.DataFrame({
    "time": test_times
})

fig, axes = plt.subplots(
    len(stations),
    1,
    figsize=(15, 12),
    sharex=True
)

for ax, (station_name, (j, i)) in zip(
    axes,
    stations.items()
):

    if not (0 <= j < H and 0 <= i < W):
        raise IndexError(
            f"{station_name} ({j}, {i}) is outside "
            f"the NEMO grid ({H}, {W})."
        )

    full_flat_index = j * W + i

    water_cell_index = flat_to_water[
        full_flat_index
    ]

    if water_cell_index == -1:
        raise ValueError(
            f"{station_name} at ({j}, {i}) is not "
            "a NEMO surface-water cell."
        )

    actual_station = true_test_water[
        :,
        water_cell_index
    ]

    predicted_station = pred_test_water[
        :,
        water_cell_index
    ]

    valid_station = (
        np.isfinite(actual_station)
        & np.isfinite(predicted_station)
    )

    station_error = (
        predicted_station[valid_station]
        - actual_station[valid_station]
    )

    station_rmse = np.sqrt(
        np.mean(station_error**2)
    )

    station_mae = np.mean(
        np.abs(station_error)
    )

    station_bias = np.mean(
        station_error
    )

    station_results.append({
        "station": station_name,
        "nemo_j": j,
        "nemo_i": i,
        "water_cell": water_cell_index,
        "rmse": station_rmse,
        "mae": station_mae,
        "bias": station_bias
    })

    column_name = (
        station_name.lower()
        .replace(" ", "_")
    )

    station_timeseries[
        f"{column_name}_actual"
    ] = actual_station

    station_timeseries[
        f"{column_name}_predicted"
    ] = predicted_station

    ax.plot(
        test_times,
        actual_station,
        label="Actual HRDPS",
        linewidth=1
    )

    ax.plot(
        test_times,
        predicted_station,
        label="U-Net prediction",
        linewidth=1,
        alpha=0.7
    )

    ax.set_title(
        f"{station_name}: RMSE = {station_rmse:.3f}"
    )

    ax.set_ylabel("tair")
    ax.grid()
    ax.legend()

axes[-1].set_xlabel("Time")

plt.suptitle(
    "NEMO Station Actual vs Predicted Temperature, 2012",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    output_dir / "unet_nemo_station_timeseries_2012.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


station_metrics = pd.DataFrame(
    station_results
)

station_metrics.to_csv(
    output_dir / "unet_nemo_station_metrics_2012.csv",
    index=False
)

station_timeseries.to_csv(
    output_dir / "unet_nemo_station_timeseries_2012.csv",
    index=False
)

print(station_metrics)


# In[ ]:


average_station_rmse = station_metrics[
    "rmse"
].mean()

print(
    "Average RMSE across Sand Heads, "
    "Halibut Bank, and Sentry Shoal:",
    average_station_rmse
)


# In[ ]:


average_station_rmse_df = pd.DataFrame({
    "metric": ["average_station_rmse"],
    "value": [average_station_rmse]
})

average_station_rmse_df.to_csv(
    output_dir / "unet_nemo_average_station_rmse_2012.csv",
    index=False
)


# In[ ]:


average_row = pd.DataFrame({
    "station": ["Average"],
    "nemo_j": [np.nan],
    "nemo_i": [np.nan],
    "water_cell": [np.nan],
    "rmse": [average_station_rmse],
    "mae": [station_metrics["mae"].mean()],
    "bias": [station_metrics["bias"].mean()]
})

station_metrics_with_average = pd.concat(
    [station_metrics, average_row],
    ignore_index=True
)

station_metrics_with_average.to_csv(
    output_dir / "unet_nemo_station_metrics_2012.csv",
    index=False
)

