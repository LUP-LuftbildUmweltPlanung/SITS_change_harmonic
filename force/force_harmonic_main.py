from force_harmonic_utils import *


params = {
    "project_name": "aoi_sbs_test",
    #preprocess
    "aoi" : "/uge_mount/FORCE/new_struc/data/aoi_test_sbs/aoi_sbs_test.shp",
    #analysis TSS
    "Sensors" : "SEN2A SEN2B", #LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B,
    "SPECTRAL_ADJUST" : "FALSE",
    #filter and interpolation
    "ABOVE_NOISE" : "3",
    "BELOW_NOISE" : "1",
    "DATE_RANGE": "2018-01-01 2023-12-31",# Will also be Prediction Time Range for TSI
    #analysis TSI
    "TSI_Sensors": "SEN2A SEN2B",#"LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B"
    "TSI_SPECTRAL_ADJUST": "FALSE",
    "TSI_ABOVE_NOISE": "99",
    "TSI_BELOW_NOISE": "1",
    "TSI_DATE_RANGE": "2017-01-01 2019-01-01", # Ref Period for TSI
    }

advanced_params = {
    "force_dir": "/force:/force",
    "local_dir": "/uge_mount:/uge_mount",
    "force_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel/force_cube_sceleton",
    "scripts_skel": "/uge_mount/FORCE/new_struc/scripts_sits/sits_force/skel",
    "temp_folder": "/uge_mount/FORCE/new_struc/process/temp",
    "mask_folder": "/uge_mount/FORCE/new_struc/process/mask",
    "proc_folder": "/uge_mount/FORCE/new_struc/process/result",
    "data_folder": "/uge_mount/FORCE/new_struc/data",
    ###BASIC PARAMS###
    "hold": True,  # execute cmd
    # compute
    "NTHREAD_READ": 4,  # 4,
    "NTHREAD_COMPUTE": 11,  # 11,
    "NTHREAD_WRITE": 2,  # 2,
    "BLOCK_SIZE": 300,
    # UDF TSS
    "PYTHON_TYPE": "PIXEL",
    "OUTPUT_PYP": "TRUE",
    # UDF TSI
    "TSI_PYTHON_TYPE": "PIXEL",
    "TSI_OUTPUT_PYP": "TRUE",
    }


if __name__ == '__main__':
    force_harmonic(**params,**advanced_params)
