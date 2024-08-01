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
from SITS_change_harmonic.utils.residuals_utils import extract_data, plot_timeseries

raster_tsi = "/.../.../tiles_tsi/X0068_Y0038/2017-2019_001-365_HL_UDF_SEN2L_PYP.tif"
raster_tss = "/.../.../tiles_tss/X0068_Y0038/2015-2024_001-365_HL_UDF_SEN2L_PYP.tif"
points = gpd.read_file("/.../.../aoi_points.shp")
save_fig = "/.../.../output_figures"
threshold = "std" # number[0,1,2,...] or "std"
id_column = "ProbeBNr"
title = "Sentinel, Ref: Sentinel 2016-2018, Point ID: "
ylab = "Vitalitätsindex CCCI"




os.makedirs(save_fig, exist_ok=True)
# Open the raster stack
if threshold == "std":
    with_std = True
else:
    with_std = False

data_tsi, dates_tsi, _, data_std = extract_data(raster_tsi, with_std)
data_tss, dates_tss, _, __ = extract_data(raster_tss, with_std=False)

with rasterio.open(raster_tss) as src:
    # Use the same affine as for the TSI data
    affine = src.transform
# Load the point shapefile
for idx, point in points.iterrows():
    #EXTRACT TSI from point
    tsi_time_series = []
    # Loop through each time step in the stack
    for i, step in enumerate(data_tsi.transpose(2,0,1)):
        # Get the time value for this step
        time = datetime.datetime.strptime(dates_tsi[i], '%Y-%m-%d').date()
        values = rasterstats.point_query(point.geometry, step,affine=affine, interpolate='nearest')
        if values[0] == -9999:
            values[0] = np.nan
        # Add the time and values to the time series list
        tsi_time_series.append([time, values])
    
    #EXTRACT TSS from point
    tss_time_series = []
    # Loop through each time step in the stack
    for i, step in enumerate(data_tss.transpose(2,0,1)):
        # Get the time value for this step
        time = datetime.datetime.strptime(dates_tss[i], '%Y-%m-%d').date()
            
        # Get the TSS values at the points location
        values = rasterstats.point_query(point.geometry, step,affine=affine, interpolate='nearest')
        if values[0] == -9999:
            values[0] = np.nan
        # Add the time and values to the TSS time series list
        tss_time_series.append([time, values])
    
    if with_std == True:
        threshold = (rasterstats.point_query(point.geometry, data_std,affine=affine, interpolate='nearest'))[0]*1
    
    plot_timeseries(tsi_time_series, tss_time_series, threshold, point, with_std, save_fig, ylab, title, id_column)



