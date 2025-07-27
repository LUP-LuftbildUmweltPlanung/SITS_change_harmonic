import os
import time
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import rasterio
from rasterio.merge import merge
import glob
from scipy.stats import linregress
import gc
from utils.residuals_utils import extract_data
from utils.residuals_utils import get_output_array_full
from utils.residuals_utils import write_output_raster
from utils.residuals_utils import slice_by_date
from utils.residuals_utils import calculate_residuals
from utils.analysis_utils import *
import fastnanquantile as fnq


def format_time(seconds):
    """Format the time in hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

startzeit = time.time()


# def optimized_a_p10_function(sliced_array):
#     # Ensure sliced_array is a NumPy array
#     sliced_array = np.array(sliced_array)
#
#     # Create an empty array to store the quantile results
#     a_p10 = np.empty(sliced_array.shape[:2], dtype=int)  # Use int directly to avoid the final cast
#
#     # Compute positive and negative values counts
#     positive_values = np.sum(sliced_array > 0, axis=2)
#     negative_values = np.sum(sliced_array <= 0, axis=2)
#
#     # Compute the majority mask (True for majority positive, False for majority negative)
#     majority_positive_mask = positive_values > negative_values
#
#     # Vectorized quantile calculation based on the majority mask
#     for i in range(sliced_array.shape[0]):  # Iterate over rows
#         for j in range(sliced_array.shape[1]):  # Iterate over columns
#             quantile_value = 0.9 if majority_positive_mask[i, j] else 0.1
#             # Calculate the quantile for the current pixel's time series slice
#             quantile_result = np.nanquantile(sliced_array[i, j, :], quantile_value)
#             if np.isnan(quantile_result):  # If quantile result is NaN, replace it with 9999
#                 quantile_result = 9999
#             a_p10[i, j] = quantile_result
#
#     # Return the result as integer array
#     return a_p10.astype(int)


def harmonic(project_name,prc_change,deviation,trend_whole,int10p_whole,firstdate_whole,intp10_period,mosaic,times_std,start_date,end_date,period_length,process_folder,tsi_lst,tss_lst, **kwargs):

    temp_folder = process_folder + "/temp"
    proc_folder = process_folder + "/results"

    if not tsi_lst or not tss_lst:
        tsi_lst = glob.glob(f"{temp_folder}/{project_name}/tiles_tsi/X*/*.tif")
        tss_lst = glob.glob(f"{temp_folder}/{project_name}/tiles_tss/X*/*.tif")

    for raster_tsi, raster_tss in zip(tsi_lst, tss_lst):
        print("###" * 10)
        print(f"TSI:  {raster_tsi}\nTSS:  {raster_tss}")
        output = raster_tss.replace(".tif", "_output")
        if os.path.exists(output):
            print("Output Folder in TSS aleady exists. Skipping Processing ...")
            #continue
        if not os.path.exists(output):
            os.mkdir(output)

        ## force mask / wald = 0, nodata sentinel = -9999
        residuals_nrt = None

        raster_tss_data, dates_nrt, sens, _, __ = extract_data(raster_tss, with_std = False)
        raster_tsi_data, dates_tsi, sens, data_std, model = extract_data(raster_tsi, with_std = True)

        model[np.isnan(model)] = 9999
        write_output_raster(raster_tss, output, model, f"/model.tif", 1)
        model = None

        nrt_raster_data = calculate_residuals(raster_tss_data, raster_tsi_data, dates_nrt, dates_tsi,prc_change)

        raster_tsi_data = None
        raster_tss_data = None

        # nrt_raster_data = raster_tss_data
        print("###" * 10)
        print("finished calculating residuals\n")

        threshold = abs(data_std * times_std)
        # get forest mask lately ... assumption that all values in z dimension are nan
        forest_mask = np.isnan(nrt_raster_data).all(axis=2)

        if "thresholding" in deviation:
            output_array_full, filtered = get_output_array_full(nrt_raster_data, threshold)
            if firstdate_whole == True:
                output_array = calculate_firstdate_whole(output_array_full, dates_nrt, forest_mask)
                write_output_raster(raster_tss, output, output_array, f"/first_date_threshold.tif", 1)
            if int10p_whole == True:
                a_p10 = calculate_int10p_whole(output_array_full, forest_mask)
                write_output_raster(raster_tss, output, a_p10, f"/intensity_threshold.tif", 1)
                print("###" * 10)
                print(f'finished intensity for whole time period (residual related)\n')
            if intp10_period == True:
                a_p10, sm_split, em_split = calculate_intp10_period(start_date, end_date, period_length, dates_nrt, output_array_full, filtered, forest_mask, "thresholding")
                write_output_raster(raster_tss, output, a_p10,
                                    f"/{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_INTp10_threshold.tif", 1)
        if "raw" in deviation:
            output_array_full = nrt_raster_data
            if firstdate_whole == True:
                output_array = calculate_firstdate_whole(output_array_full, dates_nrt, forest_mask)
                write_output_raster(raster_tss, output, output_array, f"/first_date_raw.tif", 1)
            if int10p_whole == True:
                a_p10 = calculate_int10p_whole(output_array_full, forest_mask)
                write_output_raster(raster_tss, output, a_p10, f"/intensity_raw.tif", 1)
                print("###" * 10)
                print(f'finished intensity for whole time period (residual related)\n')
            if intp10_period == True:
                a_p10, sm_split, em_split = calculate_intp10_period(start_date, end_date, period_length, dates_nrt, output_array_full, filtered, forest_mask)
                write_output_raster(raster_tss, output, a_p10,
                                    f"/{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_INTp10_raw.tif", 1)
        if "safe" in deviation:
            output_array_raw = nrt_raster_data
            forest_mask_extended = forest_mask[:, :, np.newaxis]
            missing_values = np.logical_and(np.isnan(output_array_raw), ~forest_mask_extended)
            output_array_raw[missing_values] = 5000
            output_array_raw[np.isnan(output_array_raw)] = 9999
            write_output_raster(raster_tss, output, output_array_raw, f"/residuals.tif", rasterio.open(raster_tss).count)
        if deviation == ["safe"]:
            continue


    if mosaic == True:

        if not os.path.exists(f"{proc_folder}"):
            os.makedirs(f"{proc_folder}")
            print("###" * 10)
            print(f"created new folder: {proc_folder}")
        if not os.path.exists(f"{proc_folder}/{project_name}"):
            os.makedirs(f"{proc_folder}/{project_name}")
            print("###" * 10)
            print(f"created new folder: {proc_folder}/{project_name}")

        file_lst = glob.glob(tss_lst[0].replace(".tif", "_output")+"/*.tif")
        print("###" * 10)
        print(f"starting mosaicing for results")
        for file in file_lst:
            mosaic_files = glob.glob(file.replace(file.split("/")[-3],"X*"))
            #print(file.replace(file.split("/")[-3],"X*"))
            #print(mosaic_files)
            base = os.path.basename(file)
            output_filename = f"{proc_folder}/{project_name}/{base}"
            if "residuals.tif" in base:
                mosaic_rasters(mosaic_files, output_filename, band_descriptions=dates_nrt)
            else:
                mosaic_rasters(mosaic_files, output_filename)
            #mosaic_rasters(mosaic_files, output_filename)




def mosaic_rasters(input_pattern, output_filename, band_descriptions=None):
    """
    Mosaic rasters matching the input pattern and save to output_filename.

    Parameters:
    - input_pattern: str, a wildcard pattern to match input raster files (e.g., "./tiles/*.tif").
    - output_filename: str, the name of the output mosaic raster file.
    """

    # Find all files matching the pattern
    src_files_to_mosaic = [rasterio.open(fp) for fp in input_pattern]

    # Mosaic the rasters
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Get metadata from one of the input files
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update metadata with new dimensions, transform, and compression (optional)
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions, start=1):
                if desc:
                    dest.set_band_description(i, desc)

    # Close the input files
    for src in src_files_to_mosaic:
        src.close()


endzeit = time.time()
print("###" * 10)
print("process finished in "+str((endzeit-startzeit)/60)+" minutes")