# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:55:21 2023

@author: benjaminstoeckigt

"""
import time
from force.force_harmonic_utils import *
from utils.harmonic_utils import *

#### Default Index DSWI -> to change adjust .force/skel/dswi_harmonic*.py
#### BNIR USED FOR DSWI, if harmonized with Landsat -> NIR must been changed in .force/skel/dswi_harmonic*.py
params = {
    #########################
    #########Basics##########
    #########################
    "project_name": "HIGH_VALUES2", #Project Name that will be the name of output folder in temp & result subfolder test_full_tile_all_time
    "aoi": "/uge_mount/Freddy/harmonic_model/data/positive_disturbence/high_values_test.shp", #Define Area of Interest as Shapefile

    #TimeSeriesStack (TSS) --> Real Spectral Values
    "TSS_Sensors": "SEN2A SEN2B", #LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B, # Choose between Input Sensors
    "TSS_DATE_RANGE": "2019-01-01 2019-12-31",# TimeRange for ChangeDetection. Will also be Prediction Time Range for TSI 2018-06-01 2025-07-16

    #TimeSeriesInterpolation (TSI) --> Interpolated Spectral Values
    "TSI_Sensors": "SEN2A SEN2B", #"LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B", # "SEN2A SEN2B",Choose between Input Sensors
    "TSI_DATE_RANGE": "2016-01-01 2018-06-01",# Reference Period for Interpolation Model

    ###########################
    ##HARMONIC Postprocessing##
    ###########################
    "prc_change": True, # way of analyse change in spectral value related to harmonic model prediction
    # False --> residual change [threshold --> std of harmonic reference period]
    # True --> relative change in percent [threshold --> coefficient of variation - (std / mean ) * 100]
    "deviation": ["safe", "thresholding"], # "safe", "thresholding", "raw" ## "thresholding": anomaly cleaning (3 times lower/higher threshold) will be applied; "safe": residuals will be safed and further processes skipped; "raw": raw residuals will be used for further processes; it's possible to input multiple options
    "trend_whole": False,
    "int10p_whole": False, # Calculate the 10th Perzentil (negative Devivations for negative Change in Spectral Value)
    "firstdate_whole": False, # Calculate the first Date the Change was detected
    "intp10_period": True, # Calculate the 10th Perzentil for periods specified below
    "mosaic": True, # Mosaic the final results?

    "times_std": -1, # Threshold for ChangeDetection (std * -x | cv * -x)
    # Define start and end dates and period length
    "start_date": "2019-03", # Starting Date for Period Calculation
    "end_date": "2019-05", # End Date for Period Calculation  2025-07
    "period_length": 2, # # Time Range for Period Calculation
    }

advanced_params = {
    #BASIC
    "process_folder": "/uge_mount/Freddy/harmonic_model/process/", # Folder where Data and Results will be processed (will be created if not existing)
    "force_dir": "/force", # mount directory for FORCE-Datacube - should look like /force_mount/FORCE/C1/L2/..
    #"tsi_lst" : glob(".../tiles_tsi/X*/2017-2019_001-365_HL_UDF_SEN2L_PYP.tif"),
    #"tss_lst" : glob(".../tiles_tss/X*/2018-2023_001-365_HL_UDF_SEN2L_PYP.tif"),
    "tsi_lst": None, #tss & tsi will be automatically used from project_folder structure
    "tss_lst": None, #tss & tsi will be automatically used from project_folder structure

    # To disable
    "TSS_ABOVE_NOISE": 3, #noise filtering in spectral values above 3 x std; take care for not filtering real changes
    "TSS_BELOW_NOISE": 1, #get back values from qai masking below single std
    "TSS_SPECTRAL_ADJUST": "FALSE", #spectral adjustment will be necessary by using Sentinel 2 & Landsat together

    "Model": "notrend",  # you can choose between a model including or exluding trend ["notrend","trend"]
    # there are three different complexities for the harmonic model, that will be chosen by the amount of valid spectral values within the reference period --> have a look at UDF - Function

    "TSI_ABOVE_NOISE": 3, #noise filtering in spectral values above 3 x std
    "TSI_BELOW_NOISE": 1, #get back values from qai masking below single std
    "TSI_SPECTRAL_ADJUST": "FALSE", #spectral adjustment will be necessary by using Sentinel 2 & Landsat together

    "hold": False,  # if True, cmd must be closed manually ## recommended for debugging FORCE

    #Streaming Mechnism
    "TSS_NTHREAD_READ": 7,  # 4,
    "TSS_NTHREAD_COMPUTE": 7,  # 11,
    "TSS_NTHREAD_WRITE": 2,  # 2,
    "TSS_BLOCK_SIZE": 1000,
    "TSI_NTHREAD_READ": 7,  # 4,
    "TSI_NTHREAD_COMPUTE": 7,  # 11,
    "TSI_NTHREAD_WRITE": 2,  # 2,
    "TSI_BLOCK_SIZE": 1000,
    }

def format_time(seconds):
    """Format the time in hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

if __name__ == '__main__':
    # Measure time for force_harmonic
    startzeit_force = time.time()
    #force_harmonic(**params, **advanced_params)
    endzeit_force = time.time()
    #force_harmonic_time = endzeit_force - startzeit_force
    #print(f"force_harmonic executed in: {format_time(force_harmonic_time)}")

    # Measure time for harmonic
    startzeit_harmonic = time.time()
    harmonic(**params, **advanced_params)
    endzeit_harmonic = time.time()
    harmonic_time = endzeit_harmonic - startzeit_harmonic
    print(f"harmonic executed in: {format_time(harmonic_time)}")

    # Total time
    #total_time = force_harmonic_time + harmonic_time
    #print(f"Total execution time: {format_time(total_time)}")



