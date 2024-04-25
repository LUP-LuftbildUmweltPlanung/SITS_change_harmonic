# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:55:21 2023

@author: LUP

"""
from sits_change_harmonic.force.force_harmonic_utils import *
from sits_change_harmonic.utils.harmonic_utils import *
from sits_change_harmonic.config_path import path_params

#### Default Index DSWI -> to change adjust .force/skel/force_cube_sceleton_dswi*.py
#### BNIR USED FOR DSWI, if harmonized with Landsat -> NIR must been changed in .force/skel/force_cube_sceleton_dswi*.py
params = {
    #########################
    #########Basics##########
    #########################
    "project_name": "Th_2024",
    "aoi": "/uge_mount/force_sits/process/data/Thueringen/th_25833.shp",

    #TimeSeriesStack (TSS)
    "TSS_Sensors": "SEN2A SEN2B", #LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B,
    "TSS_DATE_RANGE": "2018-01-01 2024-04-25",# Will also be Prediction Time Range for TSI

    #TimeSeriesInterpolation (TSI)
    "TSI_Sensors": "LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B",#"LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B",
    "TSI_DATE_RANGE": "2010-01-01 2018-01-01",# Ref Period for TSI
    "Trend": True,
    ###########################
    ##HARMONIC Postprocessing##
    ###########################
    "residuals": "thresholding", # "safe", "thresholding", "raw" ## "thresholding": anomaly cleaning (3 times lower/higher threshold) will be applied; "safe": residuals will be safed and further processes skipped; "raw": rawa residuals will be used for further processes
    "int10p_whole": True,
    "firstdate_whole": False,
    "intp10_period": False,
    "mosaic": True,

    "times_std": -1.5,
    # Define start and end dates and period length
    "start_date": "2018-01",
    "end_date": "2024-12",
    "period_length": 12,
    }

advanced_params = {
    #BASIC
    #"tsi_lst" : glob(".../tiles_tsi/X*/2017-2019_001-365_HL_UDF_SEN2L_PYP.tif"),
    #"tss_lst" : glob(".../tiles_tss/X*/2018-2023_001-365_HL_UDF_SEN2L_PYP.tif"),
    "tsi_lst": None, #tss & tsi will be automatically used from project_folder structure
    "tss_lst": None, #tss & tsi will be automatically used from project_folder structure

    "TSS_ABOVE_NOISE": 3, #noise filtering in spectral values above 3 x std; take care for not filtering real changes
    "TSS_BELOW_NOISE": 1, #get back values from qai masking below single std
    "TSS_SPECTRAL_ADJUST": "FALSE", #spectral adjustment will be necessary by using Sentinel 2 & Landsat together

    "TSI_ABOVE_NOISE": 3, #noise filtering in spectral values above 3 x std
    "TSI_BELOW_NOISE": 1, #get back values from qai masking below single std
    "TSI_SPECTRAL_ADJUST": "TRUE", #spectral adjustment will be necessary by using Sentinel 2 & Landsat together

    "hold": False,  # if True, cmd must be closed manually

    #Streaming Mechnism
    "NTHREAD_READ": 7,  # 4,
    "NTHREAD_COMPUTE": 7,  # 11,
    "NTHREAD_WRITE": 2,  # 2,
    "BLOCK_SIZE": 1000,
    }


if __name__ == '__main__':
    force_harmonic(**params,**path_params,**advanced_params)
    harmonic(**params,**path_params)




