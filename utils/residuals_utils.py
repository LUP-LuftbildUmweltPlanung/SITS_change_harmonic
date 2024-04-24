# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:51:48 2023

@author: Admin
"""
import rasterio
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib as mtplt
from scipy.interpolate import interp1d


def extract_data(raster_file,with_std):
    #raster_file=residuals_nrt
    with rasterio.open(raster_file) as src:
        if with_std == True:
            count = (src.count)-1
        else:
            count = src.count
        
        dates_nrt = []
        sens = []
        bands_to_read = []
        for band in range(count): 
            date_string = src.descriptions[band][:8]
            sen = src.descriptions[band][9:]
            dt = datetime.datetime.strptime(date_string, '%Y%m%d')
            bands_to_read.append(band+1)
            sens.append(sen)
            dates_nrt.append(dt.strftime("%Y-%m-%d"))
        
        nrt_raster_data = src.read(bands_to_read)
        nrt_raster_data = nrt_raster_data.astype(float)
        nrt_raster_data[np.logical_or(nrt_raster_data==-9999,nrt_raster_data==0)] = np.nan
        ## force mask / wald = 0, nodata sentinel = -9999
        
        if with_std == True:
            std_raster_data = src.read(count+1)
            std_raster_data = std_raster_data.astype(float)
            std_raster_data[np.logical_or(std_raster_data==-9999,std_raster_data==0)] = np.nan
        else:
            std_raster_data = []
        
        
        return nrt_raster_data.transpose(1,2,0), dates_nrt, sens, std_raster_data

def calculate_residuals(raster_tss_data,raster_tsi_data,dates_nrt,dates_tsi):
    # Convert date strings to datetime objects
    dates_tsi_dt = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_tsi]
    # Convert datetime objects to timestamps
    timestamps_tsi = [d.timestamp() for d in dates_tsi_dt]
    # Create a function to interpolate along the time axis
    interp_func = interp1d(timestamps_tsi, raster_tsi_data, axis=2, copy = False, kind='linear',fill_value="extrapolate")
    raster_tsi_data = None
    # Convert date strings in dates_tss to datetime objects
    dates_tss_dt = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_nrt]
    # Convert datetime objects to timestamps
    timestamps_tss = [d.timestamp() for d in dates_tss_dt]
    # Use the interpolation function to interpolate raster_tsi_data to the new time axis
    # Define chunk size
    chunk_size = 50
    
    # Interpolate raster_tsi_data in chunks
    interpolated_tsi_chunks = []
    for i in range(0, len(timestamps_tss), chunk_size):
        tss_chunk = timestamps_tss[i:i+chunk_size]
        interpolated_tsi_chunk = interp_func(tss_chunk)
        interpolated_tsi_chunks.append(interpolated_tsi_chunk)
    
    # Concatenate interpolated chunks
    interpolated_tsi_data = np.concatenate(interpolated_tsi_chunks, axis=2)
    
    # Calculate the difference between the interpolated raster_tsi_data and raster_tss_data
    raster_tss_data -= interpolated_tsi_data
    return raster_tss_data

def get_output_array_full(nrt_raster_data, threshold):
    output_array_full = np.zeros((nrt_raster_data.shape),dtype = nrt_raster_data.dtype)
    output_array_full[output_array_full==0] = np.nan
    #unconfirmed_dist_array = np.zeros((nrt_raster_data.shape),dtype = np.int8)
    #unconfirmed_dist_array_test = np.zeros((nrt_raster_data.shape),dtype = np.int8)
    #unconfirmed_release_array = np.zeros((nrt_raster_data.shape),dtype = np.int8)
    #unconfirmed_release_array_test = np.zeros((nrt_raster_data.shape),dtype = np.int8)
    #startend_dieback = np.zeros((nrt_raster_data.shape),dtype = np.int8)

    filtered = np.zeros((nrt_raster_data.shape), dtype=bool)
    anomaly_count = np.zeros((nrt_raster_data.shape[0],nrt_raster_data.shape[1]),dtype = nrt_raster_data.dtype)
    reset_count = np.zeros((nrt_raster_data.shape[0],nrt_raster_data.shape[1]),dtype = nrt_raster_data.dtype)

    for full,layer in enumerate(nrt_raster_data.transpose(2,0,1)):
        print(f"Timestep {full} from {nrt_raster_data.shape[2]} processed ...")
        layer_belowth = layer < threshold ## Cloudmasked & Forestmasked --> false as well
        layer_higherth = layer > threshold ## Cloudmasked & Forestmasked --> false as well
        
        anomaly_prev = np.copy(anomaly_count)
        anomaly_bool = np.logical_and(anomaly_count != 3,layer_belowth)
        anomaly_count[anomaly_bool] = anomaly_count[anomaly_bool]+1
        
        reset_bool = np.logical_and(anomaly_count == 3,layer_higherth) 
        reset_count[reset_bool] = reset_count[reset_bool]+1
        count_up_reset = np.logical_and(anomaly_count != 3,layer_higherth) # for resetting anomaly counter when counting up to 3 and consequitive anomalies are disrupted 
        
        anomaly_count[np.logical_or(reset_count == 3,count_up_reset)] = 0 
        
        #print(np.logical_and(anomaly_prev==anomaly_count,np.logical_and(anomaly_count!=0,anomaly_count!=3)).sum())
        
        
        #startend_dieback[:,:,full][np.logical_and(anomaly_prev == 2,anomaly_count ==3)]=1
        #startend_dieback[:,:,full][reset_count == 3]=2
        #unconfirmed_dist_array[:,:,full]=anomaly_count
        #unconfirmed_release_array[:,:,full]=reset_count

        filtered_bool = np.logical_and(anomaly_count != 3,~np.isnan(layer))
        filtered[filtered_bool,full] = True
        #print(sum(filtered))
        output_bool = anomaly_count == 3
        output_array_full[output_bool,full] = layer[output_bool]

        
        ## New Emergency Solution that is stable - not emergency
        # Identify anomalies that started or ended
        unconf_start = np.logical_and(anomaly_prev == 2, anomaly_count == 3)
        unconf_end = reset_count == 3

        # For each position in unconf_start and unconf_end, identify the last two non-NaN layers
        rows, cols = np.where(unconf_start)
        for r, c in zip(rows, cols):
            non_nan_layers = np.where(~np.isnan(nrt_raster_data[r, c, :full]))[0][-2:]
            output_array_full[r, c, non_nan_layers] = nrt_raster_data[r, c, non_nan_layers]
            filtered[r, c, non_nan_layers] = False
        rows, cols = np.where(unconf_end)
        for r, c in zip(rows, cols):
            non_nan_layers = np.where(~np.isnan(nrt_raster_data[r, c, :full]))[0][-2:]
            output_array_full[r, c, non_nan_layers] = np.nan
            filtered[r, c, non_nan_layers] = True

        # resetting reset_count
        reset_count[np.logical_or(reset_count == 3, layer_belowth)] = 0        
        

        
        ###OLD SOLUTION#####Emergency solution!! Currently, when a disturbance is released/started, the time frame of the last two recordings is adjusted ... NA values ​​in past recordings are not yet taken into account.
        # full_minus = max(0,full-2)

        # unconf_bool = np.logical_and(anomaly_prev == 2,anomaly_count ==3)
        # #unconfirmed_dist_array_test[unconf_bool,full_minus:full] = 5
        # #unconfirmed_release_array_test[reset_count==3,full_minus:full+1]= 5  
        # output_array_full[unconf_bool,full_minus:full] = nrt_raster_data[unconf_bool,full_minus:full]
        # output_array_full[reset_count==3,full_minus:full+1] = np.nan

        # # resetting reset_count
        # reset_count[np.logical_or(reset_count == 3,layer_belowth)] = 0
    

    return output_array_full, filtered



def write_output_raster(nrt_raster,output, array, suffix, nbands):
    nrt_raster = rasterio.open(nrt_raster)
    kwargs = nrt_raster.meta
    kwargs.update(
        dtype='int32',
        count=nbands,
        compress='lzw',
        nodata= -9999)

    if nbands == 1:
        with rasterio.open(output + suffix, 'w', **kwargs) as dst:
            dst.write(array,nbands)
    
    else:
        with rasterio.open(output + suffix, 'w', **kwargs) as dst:
            for bcount in range(nbands):
                dst.write(array[:,:,bcount], bcount+1)



def slice_by_date(output_array_full, dates_nrt, start_month, period_length):
    # Convert dates to datetime objects
    dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_nrt]
    
    # Find the indices of the start and end dates
    start_date = datetime.datetime.strptime(start_month, '%Y-%m')
    end_date = start_date + relativedelta(months=period_length)

    start_index = np.argmax(np.array(dates) >= start_date)
    end_index = np.argmax(np.array(dates) >= end_date)
    # Adjust end_index if it is out of range
    if end_index == 0 and end_date > dates[-1]:
        end_index = len(dates)
    # Slice the output array based on the indices
    sliced_array = output_array_full[:, :, start_index:end_index]
    #sliced_array = output_array_full[:, :, start_index]
    
    return sliced_array



def plot_timeseries(tsi_time_series, tss_time_series, threshold, points, with_std, save_fig, ylab, title, id_column):

    mtplt.rcParams['figure.dpi']= 300
    # Create a list of dates in tss_time_series
    dates = [ts[0] for ts in tss_time_series]

    # Interpolate the tsi_time_series data to match the dates in tss_time_series
    interpolated_tsi = np.interp([d.toordinal() for d in dates], [d.toordinal() for d in [ts[0] for ts in tsi_time_series]], [ts[1][0] for ts in tsi_time_series])
    tsi_time_series = [[d, [tsi]] for d, tsi in zip(dates, interpolated_tsi)]
    plt.figure(figsize=(10, 5))
    # Plot the TSI and TSS time series data
    start_date = datetime.datetime(2018, 1, 1).date()
    end_date = datetime.datetime(2023, 12, 31).date()

    tsi_time_series = [ts for ts in tsi_time_series if start_date <= ts[0] <= end_date]
    tss_time_series = [ts for ts in tss_time_series if start_date <= ts[0] <= end_date]

    # Create the new time series with the values of time_series_2018 + threshold
    time_series_threshold_plus = [[tsi_time_series[i][0], tsi_time_series[i][1][0] + threshold] for i in range(len(tsi_time_series))]
    time_series_threshold_minus = [[tsi_time_series[i][0], tsi_time_series[i][1][0] - threshold] for i in range(len(tsi_time_series))]
    plt.plot([tsi_time_series[j][0] for j in range(len(tsi_time_series))],[tsi_time_series[i][1] for i in range(len(tsi_time_series))], label=f"Erwartungswert", color='black',linewidth=1)
    #plt.plot([tsi_time_series[j][0] for j in range(len(tsi_time_series))], [tsi_time_series[i][1] for i in range(len(tsi_time_series))], label=f"Harmonic", color='black', linewidth=3)
    if with_std == True:
        #plt.plot([time_series_threshold_plus[j][0] for j in range(len(time_series_threshold_plus))], [time_series_threshold_plus[i][1] for i in range(len(time_series_threshold_plus))], color='black', label=f"Threshold (1*std): {threshold}", linewidth=1, linestyle='dashed')
        plt.plot([time_series_threshold_plus[j][0] for j in range(len(time_series_threshold_plus))],[time_series_threshold_plus[i][1] for i in range(len(time_series_threshold_plus))], color='black', label=f"Toleranzbereich", linewidth=0.5, linestyle='dashed')
    else: 
        plt.plot([time_series_threshold_plus[j][0] for j in range(len(time_series_threshold_plus))], [time_series_threshold_plus[i][1] for i in range(len(time_series_threshold_plus))], color='black', label=f"Threshold: {threshold}", linewidth=0.5, linestyle='dashed')
    plt.plot([time_series_threshold_minus[j][0] for j in range(len(time_series_threshold_minus))], [time_series_threshold_minus[i][1] for i in range(len(time_series_threshold_minus))], color='black',linewidth=0.5, linestyle='dashed')

    display_anomaly = False
    display_disturbance = False
    display_tss = False
    y_counter = 0#anomaly yes
    n_counter = 0#anomaly no
    for i in range(len(tss_time_series)):
        #print(tss_time_series)
        #print(tss_time_series[i][0])   #  date
        #print(tss_time_series[i][1][0])# value
        
        
        if np.isnan(tss_time_series[i][1][0]):
            continue
        elif tss_time_series[i][1][0] < time_series_threshold_minus[i][1]:
            y_counter = y_counter +1
            if not display_anomaly:
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='orange', label=f"Anomalie",s=8)
                display_anomaly = True
            else:
                if y_counter >= 3:
                    if not display_disturbance:
                        plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='red', label=f"Störung",s=8)
                        display_disturbance = True
                    else: 
                        plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='red',s=8)
                    
                    
                    
                    ### no stable solution!! if anomaly counter is triggered (3) the past two observations are going to be anomaly as well
                    #But have to check for nan values in past so not sure how many steps has to look back
                    for count in range(8):
                        c = count+1
                        c_stop = 0
                        if np.isnan(tss_time_series[i-c][1][0]):
                            continue
                        elif tss_time_series[i-c][1][0] < time_series_threshold_minus[i-c][1]:
                            plt.scatter(tss_time_series[i-c][0], tss_time_series[i-c][1], color='red',s=8)
                            c_stop += 1
                        else: 
                            c_stop += 1
                        if c_stop == 3:
                            break
                    

                    
                else:
                    plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='orange',s=8)
            if n_counter < 3: 
                n_counter = 0 
        else:
            if not display_tss:
                #plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='green', label=f"Oberservation",s=8)
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='green', label=f"Vitalwert", s=8)
                display_tss = True
            else:
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='green',s=8)

            
            n_counter = n_counter +1
            
            if y_counter <3:
                y_counter = 0
            if n_counter == 3:
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='green',s=8)
                
                ### no stable solution!! if reset counter is triggered (3) the past two observations are going to be resetted as well. 
                #But have to check for nan values in past so not sure how many steps has to look back
                for count in range(8):
                    c = count+1
                    c_stop = 0
                    if np.isnan(tss_time_series[i-c][1][0]):
                        continue
                    elif tss_time_series[i-c][1][0] > time_series_threshold_minus[i-c][1]:
                        plt.scatter(tss_time_series[i-c][0], tss_time_series[i-c][1], color='green',s=8)
                        c_stop += 1
                    else: 
                        c_stop += 1
                    if c_stop == 3:
                        break
                n_counter = 0
                y_counter = 0


            if y_counter >=3:
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='red',s=8)


    pid = str(getattr(points, id_column))
    #pid = str(points.ProbeBNr)

    #plt.title(f"Sentinel, Ref: S&L 2010-2017, Class: {dsc}, Point ID: {pid}")
    plt.title(title+str(pid))
    plt.legend(loc=3, prop={'size': 7})
    #plt.xlabel("Time")
    plt.xlabel("Jahr")
    plt.xticks(fontsize=6)
    #plt.ylabel("DSWI (Scalefactor:100)")
    plt.ylabel(ylab)
    if save_fig:
        plt.savefig(f'{save_fig}/pointID_{pid}.png', dpi=300)
    
    return plt.close()#plt.show()#

