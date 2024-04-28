import os
import time
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import rasterio
from rasterio.merge import merge
import glob
from utils.residuals_utils import extract_data
from utils.residuals_utils import get_output_array_full
from utils.residuals_utils import write_output_raster
from utils.residuals_utils import slice_by_date
from utils.residuals_utils import calculate_residuals


startzeit = time.time()
def harmonic(project_name,residuals,int10p_whole,firstdate_whole,intp10_period,mosaic,times_std,start_date,end_date,period_length,temp_folder,proc_folder,tsi_lst,tss_lst, **kwargs):
    if not tsi_lst or not tss_lst:
        tsi_lst = glob.glob(f"{temp_folder}/{project_name}/tiles_tsi/X*/*.tif")
        tss_lst = glob.glob(f"{temp_folder}/{project_name}/tiles_tss/X*/*.tif")

    for raster_tsi, raster_tss in zip(tsi_lst, tss_lst):
        print("###" * 10)
        print(f"TSI:  {raster_tsi}\nTSS:  {raster_tss}")
        output = raster_tss.replace(".tif", "_output")
        if not os.path.exists(output):
            os.mkdir(output)

        ## force mask / wald = 0, nodata sentinel = -9999
        residuals_nrt = None
        # residuals_nrt = r"E:\++++Promotion\ChangeDetection\Anwendung\harmonic\sachsen\force_reftillend2017\X0058_Y0047_10til23_LandSen_mode3\2010-2023_001-365_HL_TSA_SEN2L_NDM_NRT.tif"   ## Near Real time product ## Residuen zwischen Extrapolated harmonic und real data

        raster_tss_data, dates_nrt, sens, __ = extract_data(raster_tss, with_std = False)
        raster_tsi_data, dates_tsi, sens, data_std = extract_data(raster_tsi, with_std = True)
        nrt_raster_data = calculate_residuals(raster_tss_data, raster_tsi_data, dates_nrt, dates_tsi)
        # nrt_raster_data = raster_tss_data
        print("###" * 10)
        print("finished calculating residuals\n")

        threshold = data_std * times_std
        forest_mask = np.isnan(nrt_raster_data).all(axis=2)

        if residuals == "thresholding":
            output_array_full, filtered = get_output_array_full(nrt_raster_data, threshold)
        elif residuals == "raw":
            output_array_full = nrt_raster_data
        elif residuals == "safe":
            output_array_full = nrt_raster_data
            forest_mask_extended = forest_mask[:, :, np.newaxis]
            missing_values = np.logical_and(np.isnan(output_array_full), ~forest_mask_extended)
            output_array_full[missing_values] = 5000
            output_array_full[np.isnan(output_array_full)] = 9999
            write_output_raster(raster_tss, output, output_array_full, f"/residuals.tif", rasterio.open(raster_tss).count)

        nrt_raster_data = None
        raster_tsi_data = None
        raster_tss_data = None
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
            output_array = output_array.astype(int)

            write_output_raster(raster_tss, output, output_array, f"/first_date.tif", 1)
            first_non_nan_indices = None

        if int10p_whole == True:
            ############## Intensity and Count
            print("###" * 10)
            print(f'calculate intensity and count for whole time period (residual related)\n')
            a_p10 = np.nanpercentile(output_array_full, 10, axis=2)
            # counts=(~np.isnan(output_array_full)).sum(axis=2)
            # Fill NaN values
            a_p10[np.isnan(a_p10)] = 9999
            a_p10 = a_p10.astype(int)
            # counts[np.isnan(counts)]= 9999
            # counts = counts.astype(int)
            # a_p10[counts<20] = 9999
            # write_output_raster(residuals_nrt,output, a_p10, f"\\intensity_count_o20.tif",1)
            write_output_raster(raster_tss, output, a_p10, f"/intensity.tif", 1)
            # write_output_raster(residuals_nrt,output, counts, f"\\count.tif",1)
            print("###" * 10)
            print(f'finished intensity for whole time period (residual related)\n')

        if intp10_period == True:
            ###############################################################
            ################ ITERATE OVER TIME PERIODS ####################
            ###############################################################

            # get forest mask lately ... assumption that all values in z dimension are nan
            #forest_mask = np.isnan(nrt_raster_data).all(axis=2)

            # Loop over the date range and slice the data
            date = start_date
            print("###" * 10)
            print(
                f' starting slicing for time periods! (residual related) \n start date: {start_date} \n end date: {end_date} \n period length: {period_length} month')
            while date < end_date:

                # Print the start and end months for the current slice
                start_month = datetime.strptime(date, '%Y-%m').strftime('%B %Y')
                end_month = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length - 1)).strftime('%B %Y')

                sm_split = start_month.split(" ")
                em_split = end_month.split(" ")

                if sm_split[0] == 'December':
                    print(f"period started with December skipped ...")
                    date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                    continue
                #if sm_split[0] == 'January':
                    #print(f"period started with January skipped ...")
                    #date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                    #continue
                if sm_split[0] == 'February':
                    print(f"period started with February skipped ...")
                    date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                    continue

                # Slice the data for the current date range
                sliced_array = slice_by_date(output_array_full, dates_nrt, date, period_length)
                if residuals == "thresholding":
                    sliced_filter = slice_by_date(filtered, dates_nrt, date, period_length)
                # Move to the next date range
                date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
                print("###" * 10)
                print(f'\n Sliced array for {start_month} - {end_month}')
                print(f'array shape: {sliced_array.shape}')

                # do the calculations
                print(f'calculate intensity ...')
                a_p10 = np.nanpercentile(sliced_array, 10, axis=2)
                # a_p10 = sliced_array
                # counts=(sliced_array<threshold).sum(axis=2)
                # Fill NaN values
                a_p10[np.isnan(a_p10)] = 9999
                a_p10 = a_p10.astype(int)
                # counts[np.isnan(counts)]= 9999
                # counts = counts.astype(int)


                if residuals == "thresholding":
                    sliced_filter_2d = np.any(sliced_filter, axis=2)
                    sliced_filter_2d_nan = np.logical_and(a_p10 == 9999, sliced_filter_2d)
                    #print(sum(sliced_filter_2d))
                    a_p10[sliced_filter_2d_nan] = 0

                missing_values = np.logical_and(a_p10 == 9999, ~forest_mask)
                a_p10[missing_values] = 5000
                a_p10 = a_p10.astype(np.int32)
                write_output_raster(raster_tss, output, a_p10,
                                    f"/{sm_split[0]}_{sm_split[1]}_{em_split[0]}_{em_split[1]}_INTp10.tif", 1)
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
        "compression": "lzw"
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