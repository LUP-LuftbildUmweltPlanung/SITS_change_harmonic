# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:42:20 2023

@author: LUP
"""

import rasterstats
import geopandas as gpd
import rasterio
import numpy as np
import datetime
import os
from utils.residuals_utils import extract_data, plot_timeseries

raster_tsi = "/uge_mount/Freddy/harmonic_model/process/temp/neg2/tiles_tsi/X0058_Y0056/2016-2018_001-365_HL_UDF_SEN2L_PYP.tif" #"/uge_mount/Freddy/harmonic_model/process/temp/pos/tiles_tsi/X0057_Y0048/2016-2018_001-365_HL_UDF_SEN2L_PYP.tif"
raster_tss = "/uge_mount/Freddy/harmonic_model/process/temp/neg2/tiles_tss/X0058_Y0056/2024-2024_001-365_HL_UDF_SEN2L_PYP.tif" #"/uge_mount/Freddy/harmonic_model/process/temp/pos/tiles_tss/X0057_Y0048/2024-2024_001-365_HL_UDF_SEN2L_PYP.tif"
points = gpd.read_file("/uge_mount/Freddy/harmonic_model/data/points_neg2.shp")
save_fig = "/uge_mount/Freddy/harmonic_model/process/temp/neg2/"
uncertainty = "prc" # number[0,1,2,...] or "std"
id_column = "id"
title = "test_positive_change_visualize"
ylab = "Vitalitätsindex DSWI"

# Open the raster stack
if (uncertainty == "std") or (uncertainty == "prc"):
    with_std = True
else:
    with_std = False

data_tsi, dates_tsi, _, data_std, model = extract_data(raster_tsi, with_std)
data_tss, dates_tss, _, __, model = extract_data(raster_tss, with_std=False)

with rasterio.open(raster_tss) as src:
    # Use the same affine as for the TSI data
    affine = src.transform
# Load the point shapefile
for idx, point in points.iterrows():
    # EXTRACT TSI from point
    tsi_time_series = []
    # Loop through each time step in the stack
    for i, step in enumerate(data_tsi.transpose(2, 0, 1)):
        # Get the time value for this step
        time = datetime.datetime.strptime(dates_tsi[i], '%Y-%m-%d').date()
        values = rasterstats.point_query(point.geometry, step, affine=affine, interpolate='nearest')
        if values[0] == -9999:
            values[0] = np.nan
        # Add the time and values to the time series list
        tsi_time_series.append([time, values])

    # EXTRACT TSS from point
    tss_time_series = []
    # Loop through each time step in the stack
    for i, step in enumerate(data_tss.transpose(2, 0, 1)):
        # Get the time value for this step
        time = datetime.datetime.strptime(dates_tss[i], '%Y-%m-%d').date()

        # Get the TSS values at the points location
        values = rasterstats.point_query(point.geometry, step, affine=affine, interpolate='nearest')
        if values[0] == -9999:
            values[0] = np.nan
        # Add the time and values to the TSS time series list
        tss_time_series.append([time, values])

    if with_std == True:
        threshold = (rasterstats.point_query(point.geometry, data_std, affine=affine, interpolate='nearest'))[0] * 1

    plot_timeseries(tsi_time_series, tss_time_series, threshold, uncertainty, point, with_std, save_fig, ylab, title,
                    id_column)
