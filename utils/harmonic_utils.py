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
import fastnanquantile as fnq

import time
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

        if deviation == "thresholding":
            output_array_full, filtered = get_output_array_full(nrt_raster_data, threshold)
        elif deviation == "raw":
            output_array_full = nrt_raster_data
        elif deviation == "safe":
            output_array_full = nrt_raster_data
            forest_mask_extended = forest_mask[:, :, np.newaxis]
            missing_values = np.logical_and(np.isnan(output_array_full), ~forest_mask_extended)
            output_array_full[missing_values] = 5000
            output_array_full[np.isnan(output_array_full)] = 9999
            write_output_raster(raster_tss, output, output_array_full, f"/residuals.tif", rasterio.open(raster_tss).count)
            continue

        nrt_raster_data = None
        print("###" * 10)
        print("finished calculating anomaly intensities\n")

        ###########FINISHED Disturbance Detection PREPROCESSING ###################

        ###############################################################
        ###########PROCESS WHOLE RASTER################################
        ###############################################################
        if firstdate_whole == True:
            print("###" * 10)
            print(f'calculate first date (residual related)\n')

            # Find the indices of the first non-NaN value along the third dimension
            first_non_nan_indices = np.argmax(~np.isnan(output_array_full), axis=2)
            # Initialize the output array to NaN
            output_array = np.full((3000, 3000), np.nan)

            # Set the values in the output array
            for x in range(output_array.shape[0]):
                for y in range(output_array.shape[1]):
                    index = first_non_nan_indices[x, y]
                    if not np.isnan(output_array_full[x, y, index]):
                        output_array[x, y] = int(dates_nrt[index].replace('-', '')[2:])

            output_array[np.isnan(output_array)] = 9999
            missing_values = np.logical_and(output_array == 9999, ~forest_mask)
            output_array[missing_values] = 5000
            output_array = output_array.astype(int)

            write_output_raster(raster_tss, output, output_array, f"/first_date.tif", 1)
            first_non_nan_indices = None

        if int10p_whole == True:
            ############## Intensity and Count
            print("###" * 10)
            print(f'calculate intensity and count for whole time period (residual related)\n')
            a_p10 = np.nanpercentile(output_array_full, 10, axis=2)
            # Fill NaN values
            a_p10[np.isnan(a_p10)] = 9999
            missing_values = np.logical_and(a_p10 == 9999, ~forest_mask)
            a_p10[missing_values] = 5000
            a_p10 = a_p10.astype(int)
            write_output_raster(raster_tss, output, a_p10, f"/intensity.tif", 1)

            ## uncomment if you want to have information about amount of valid values
            # counts=(~np.isnan(output_array_full)).sum(axis=2)
            # counts[np.isnan(counts)]= 9999
            # counts = counts.astype(int)
            # write_output_raster(residuals_nrt,output, counts, f"\\count.tif",1)
            # a_p10[counts<20] = 9999
            # write_output_raster(residuals_nrt,output, a_p10, f"\\intensity_count_o20.tif",1)

            print("###" * 10)
            print(f'finished intensity for whole time period (residual related)\n')
#################
        if intp10_period == True:
            ###############################################################
            ################ ITERATE OVER TIME PERIODS ####################
            ###############################################################

            # Loop over the date range and slice the data
            date = start_date
            print("###" * 10)
            print(f' starting slicing for time periods! (residual related) \n start date: {start_date} \n end date: {end_date} \n period length: {period_length} month')
            while date < end_date:

                # Print the start and end months for the current slice
                start_month = datetime.strptime(date, '%Y-%m').strftime('%B %Y')
                end_month = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length - 1)).strftime('%B %Y')

                sm_split = start_month.split(" ")
                em_split = end_month.split(" ")

                ## uncomment or adjust if some periods should be skipped
                if sm_split[0] == 'December':
                    print(f"period started with December skipped ...")
                    date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                    continue

                # Slice the data for the current date range
                sliced_array = slice_by_date(output_array_full, dates_nrt, date, period_length)
                #sliced_array_thresh = slice_by_date(output_array_full_nothresh, dates_nrt, date, period_length)
                if deviation == "thresholding":
                    sliced_filter = slice_by_date(filtered, dates_nrt, date, period_length)
                # Move to the next date range
                date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                print("###" * 10)
                print(f'\n Sliced array for {start_month} - {end_month}')
                print(f'array shape: {sliced_array.shape}')

                # do the calculations
                print(f'calculate intensity ...')
                #a_p10 = np.nanpercentile(sliced_array, 10, axis=2)
                #sliced_array[sliced_array > 0] = np.nan
                startzeit_force = time.time()
                # force_harmonic(**params, **advanced_params)

                #######################################################
                sliced_array = np.array(sliced_array)

                # Zähle positive und negative Werte
                positive_values = np.sum(sliced_array > 0, axis=2)
                negative_values = np.sum(sliced_array <= 0, axis=2)

                # Maske: True = mehr positive Werte, False = mehr negative/gleiche
                mask_positive = positive_values > negative_values

                # Initialisiere Ergebnisarray
                a_p10 = np.empty(sliced_array.shape[:2])  # Vorbelegen mit 9999

                # Berechne das 90. Perzentil für alle "positiven" Pixel
                if np.any(mask_positive):
                    a_p10[mask_positive] = np.nanquantile(
                        sliced_array[mask_positive], 0.9, axis=1
                    )

                # Berechne das 10. Perzentil für alle "negativen" Pixel
                mask_negative = ~mask_positive
                if np.any(mask_negative):
                    a_p10[mask_negative] = np.nanquantile(
                        sliced_array[mask_negative], 0.1, axis=1
                    )

                # In Integer umwandeln
                a_p10[np.isnan(a_p10)] = 9999
                a_p10 = a_p10.astype(int)

                endzeit_force = time.time()
                force_harmonic_time_ = endzeit_force - startzeit_force
                print(f"function modified: {format_time(force_harmonic_time_)}")


                ##############################################################

                # sliced_array = np.array(sliced_array)
                #
                # a_p10 = np.empty(sliced_array.shape[:2])  # Create an empty array to store the quantile results
                #
                # # Check the majority of the values in sliced_array
                # positive_values = np.sum(sliced_array > 0, axis=2)
                # negative_values = np.sum(sliced_array <= 0, axis=2)
                #
                # # Apply the logic based on the majority
                # for i in range(sliced_array.shape[0]):  # Iterate over all slices
                #     for j in range(sliced_array.shape[1]):  # Iterate over the second dimension
                #         if positive_values[i, j] > negative_values[i, j]:
                #             quantile_value = 0.9  # Majority are positive, use 0.9 quantile
                #         else:
                #             quantile_value = 0.1  # Majority are negative, use 0.1 quantile
                #
                #         # Calculate the quantile based on the result and store in the a_p10 array
                #         quantile_result = np.nanquantile(sliced_array[i, j, :], quantile_value)
                #         if np.isnan(quantile_result):  # If quantile result is NaN, replace it with 9999
                #             quantile_result = 9999
                #         a_p10[i, j] = quantile_result
                #
                # # Replace NaN values with 9999
                # a_p10[np.isnan(a_p10)] = 9999
                # a_p10 = a_p10.astype(int)

                #a_p10 = optimized_a_p10_function(sliced_array)



                ##############################################

                # a_p10 = fnq.nanquantile(sliced_array, 0.1, axis=2)
                # a_p10[np.isnan(a_p10)] = 9999
                # a_p10 = a_p10.astype(int)


                #a_p10_nothresh = np.nanpercentile(sliced_array_thresh, 10, axis=2)
                #a_p10 = np.nanmedian(sliced_array,axis=2)

                # counts=(sliced_array<threshold).sum(axis=2)
                # Fill NaN values

                #a_p10_nothresh[np.isnan(a_p10_nothresh)] = 9999
                #a_p10_nothresh = a_p10_nothresh.astype(int)

                #counts[np.isnan(counts)]= 9999
                #counts = counts.astype(int)

                if deviation == "thresholding":
                    sliced_filter_2d = np.any(sliced_filter, axis=2)
                    sliced_filter_2d_nan = np.logical_and(a_p10 == 9999, sliced_filter_2d)
                    #print(sum(sliced_filter_2d))
                    a_p10[sliced_filter_2d_nan] = 9999 #Should be set to e.g. 0 if you want to seperate areas outside mask and no disturbance

                missing_values = np.logical_and(a_p10 == 9999, ~forest_mask)
                a_p10[missing_values] = 5000
                a_p10 = a_p10.astype(np.int32)
                write_output_raster(raster_tss, output, a_p10,
                                    f"/{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_INTp10.tif", 1)
                #write_output_raster(raster_tss, output, a_p10_nothresh,
                                    #f"/{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_INTp10_nothresh.tif", 1)

                # print(f'calculate meta ...')
                # counts_disturbance = (~np.isnan(output_array_full)).sum(axis=2)
                # counts_nodisturbance = (~(~np.isnan(output_array_full))).sum(axis=2)
                # counts_all = counts_nodisturbance + counts_disturbance
                # counts_perz = ((counts_disturbance/counts_all)*100).astype(int)

                # counts_disturbance[forest_mask] = 9999
                # counts_all[forest_mask] = 9999
                # counts_perz[forest_mask] = 9999
                # meta = np.stack((counts_disturbance, counts_all, counts_perz))
                # write_output_raster(residuals_nrt,output, meta.transpose(2,1,0), f"\\{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_META.tif",3)

        sliced_array = None
        sliced_filter = None
        output_array_full = None
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
            mosaic_rasters(mosaic_files, output_filename)




def mosaic_rasters(input_pattern, output_filename):
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

    # Close the input files
    for src in src_files_to_mosaic:
        src.close()


endzeit = time.time()
print("###" * 10)
print("process finished in "+str((endzeit-startzeit)/60)+" minutes")