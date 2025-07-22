# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:51:48 2023

@author: benjaminstoeckigt
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
            count = (src.count)-2
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

            ## for faster inference skipping d,j,f, uncomment following lines and comment previous 3 lines
            #month = dt.month
            #if month not in [12,1,2]:
                #bands_to_read.append(band+1)
                #sens.append(sen)
                #dates_nrt.append(dt.strftime("%Y-%m-%d"))
        
        nrt_raster_data = src.read(bands_to_read)
        nrt_raster_data = nrt_raster_data.astype(float)
        nrt_raster_data[np.logical_or(nrt_raster_data==-9999,nrt_raster_data==0)] = np.nan
        ## force mask / wald = 0, nodata sentinel = -9999
        
        if with_std == True:
            std_raster_data = src.read(count+1)
            std_raster_data = std_raster_data.astype(float)
            std_raster_data[np.logical_or(std_raster_data==-9999,std_raster_data==0)] = np.nan

            model = src.read(count+2)
            model = model.astype(float)
            model[np.logical_or(model == -9999, model == 0)] = np.nan
        else:
            std_raster_data = []
            model = []
        
        
        return nrt_raster_data.transpose(1,2,0), dates_nrt, sens, std_raster_data, model

def calculate_residuals(raster_tss_data,raster_tsi_data,dates_nrt,dates_tsi,relativ_prc_change):
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

    for i in range(0, len(timestamps_tss), chunk_size):
        tss_chunk = timestamps_tss[i:i + chunk_size]
        interpolated_tsi_chunk = interp_func(tss_chunk)

        if relativ_prc_change:
            interpolated_tsi_chunk[interpolated_tsi_chunk < 0] = np.nan
            raster_tss_data[:, :, i:i + chunk_size][raster_tss_data[:, :, i:i + chunk_size] < 0] = np.nan
            residual = raster_tss_data[:, :, i:i + chunk_size] - interpolated_tsi_chunk
            np.divide(residual, interpolated_tsi_chunk, out=residual, where=interpolated_tsi_chunk != 0)
            raster_tss_data[:, :, i:i + chunk_size] = residual * 100
        else:
            raster_tss_data[:, :, i:i + chunk_size] -= interpolated_tsi_chunk

    return raster_tss_data


def get_output_array_full(nrt_raster_data, threshold):
    output_array_full = np.full(nrt_raster_data.shape, np.nan, dtype=nrt_raster_data.dtype)
    filtered = np.zeros(nrt_raster_data.shape, dtype=bool)

    # Zwei getrennte Anomalie-Counter (negativ / positiv)
    anomaly_count_neg = np.zeros(nrt_raster_data.shape[:2], dtype=nrt_raster_data.dtype)
    anomaly_count_pos = np.zeros(nrt_raster_data.shape[:2], dtype=nrt_raster_data.dtype)

    reset_count_neg = np.zeros(nrt_raster_data.shape[:2], dtype=nrt_raster_data.dtype)
    reset_count_pos = np.zeros(nrt_raster_data.shape[:2], dtype=nrt_raster_data.dtype)

    for full, layer in enumerate(nrt_raster_data.transpose(2, 0, 1)):
        #print(f"Timestep {full} from {nrt_raster_data.shape[2]} processed ...")

        layer_belowth = layer < -threshold  # negative Anomalien
        layer_aboveth = layer > threshold  # positive Anomalien
        #layer_normal = np.logical_not(np.logical_or(layer_belowth, layer_aboveth))  # innerhalb normal

        # === Negative Anomalien ===
        anomaly_prev_neg = np.copy(anomaly_count_neg)
        anomaly_bool_neg = np.logical_and(anomaly_count_neg != 3, layer_belowth)
        anomaly_count_neg[anomaly_bool_neg] += 1

        reset_bool_neg = np.logical_and(anomaly_count_neg == 3, ~layer_belowth)
        reset_count_neg[reset_bool_neg] += 1

        count_up_reset_neg = np.logical_and(anomaly_count_neg != 3, np.logical_and(~layer_belowth, ~np.isnan(layer)))
        anomaly_count_neg[np.logical_or(reset_count_neg == 3, count_up_reset_neg)] = 0

        # === Positive Anomalien ===
        anomaly_prev_pos = np.copy(anomaly_count_pos)
        anomaly_bool_pos = np.logical_and(anomaly_count_pos != 3, layer_aboveth)
        anomaly_count_pos[anomaly_bool_pos] += 1

        reset_bool_pos = np.logical_and(anomaly_count_pos == 3, ~layer_aboveth)
        reset_count_pos[reset_bool_pos] += 1

        count_up_reset_pos = np.logical_and(anomaly_count_pos != 3, np.logical_and(~layer_aboveth, ~np.isnan(layer)))
        anomaly_count_pos[np.logical_or(reset_count_pos == 3, count_up_reset_pos)] = 0

        # Definieren von positiven und negativen Anomalien
        neg_anomaly = np.logical_and(anomaly_count_neg == 3, layer < -threshold)
        pos_anomaly = np.logical_and(anomaly_count_pos == 3, layer > threshold)

        # === Filter setzen ===
        # Alle anderen Zeitpunkte sind nicht relevant → filtered = True
        not_relevant = ~np.logical_or(neg_anomaly, pos_anomaly)
        filtered[not_relevant, full] = True

        # === Ergebnis schreiben ===
        output_bool = np.logical_or(neg_anomaly, pos_anomaly)
        output_array_full[output_bool, full] = layer[output_bool]

        # === Anomaliebeginn erkennen (negativ) ===
        unconf_start_neg = np.logical_and(anomaly_prev_neg == 2, anomaly_count_neg == 3)
        rows, cols = np.where(unconf_start_neg)
        for r, c in zip(rows, cols):
            non_nan_layers = np.where(~np.isnan(nrt_raster_data[r, c, :full]))[0][-2:]
            output_array_full[r, c, non_nan_layers] = nrt_raster_data[r, c, non_nan_layers]
            filtered[r, c, non_nan_layers] = False

        # === Anomaliebeginn erkennen (positiv) ===
        unconf_start_pos = np.logical_and(anomaly_prev_pos == 2, anomaly_count_pos == 3)
        rows, cols = np.where(unconf_start_pos)
        for r, c in zip(rows, cols):
            non_nan_layers = np.where(~np.isnan(nrt_raster_data[r, c, :full]))[0][-2:]
            output_array_full[r, c, non_nan_layers] = nrt_raster_data[r, c, non_nan_layers]
            filtered[r, c, non_nan_layers] = False

        # === Reset-Count zurücksetzen ===
        reset_count_neg[np.logical_or(reset_count_neg == 3, layer_belowth)] = 0
        reset_count_pos[np.logical_or(reset_count_pos == 3, layer_aboveth)] = 0

    return output_array_full, filtered


def write_output_raster(nrt_raster,output, array, suffix, nbands):
    nrt_raster = rasterio.open(nrt_raster)
    kwargs = nrt_raster.meta
    kwargs.update(
        dtype='int32',
        count=nbands,
        compress='lzw',
        nodata= 9999)

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


def plot_timeseries(tsi_time_series, tss_time_series, threshold, uncertainty, points, with_std, save_fig, ylab, title, id_column):

    mtplt.rcParams['figure.dpi']= 300
    # Create a list of dates in tss_time_series
    dates = [ts[0] for ts in tss_time_series]

    # Interpolate the tsi_time_series data to match the dates in tss_time_series
    interpolated_tsi = np.interp([d.toordinal() for d in dates], [d.toordinal() for d in [ts[0] for ts in tsi_time_series]], [ts[1][0] for ts in tsi_time_series])
    tsi_time_series = [[d, [tsi]] for d, tsi in zip(dates, interpolated_tsi)]
    plt.figure(figsize=(10, 5))
    # Plot the TSI and TSS time series data
    start_date = datetime.datetime(2018, 1, 1).date()
    end_date = datetime.datetime(2024, 12, 31).date()

    tsi_time_series = [ts for ts in tsi_time_series if start_date <= ts[0] <= end_date]
    tss_time_series = [ts for ts in tss_time_series if start_date <= ts[0] <= end_date]

    # Create the new time series with the values of time_series_2018 + threshold
    if uncertainty == "prc":
        time_series_threshold_plus = [[tsi_time_series[i][0], tsi_time_series[i][1][0] * (1 + threshold / 100)] for i in range(len(tsi_time_series))]
        time_series_threshold_minus = [[tsi_time_series[i][0], tsi_time_series[i][1][0] * (1 - threshold / 100)] for i in range(len(tsi_time_series))]
    else:
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
        elif tss_time_series[i][1][0] < time_series_threshold_minus[i][1] or tss_time_series[i][1][0] > time_series_threshold_plus[i][1]:
            y_counter += 1
            if not display_anomaly:
                plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='orange', label="Anomalie", s=8)
                display_anomaly = True
            else:
                if y_counter >= 3:
                    if not display_disturbance:
                        plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='red', label="Störung", s=8)
                        display_disturbance = True
                    else:
                        plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='red', s=8)

                    # auch hier: rückwirkend Punkte einfärben
                    for count in range(8):
                        c = count + 1
                        c_stop = 0
                        if np.isnan(tss_time_series[i - c][1][0]):
                            continue
                        elif tss_time_series[i - c][1][0] < time_series_threshold_minus[i - c][1] or \
                                tss_time_series[i - c][1][0] > time_series_threshold_plus[i - c][1]:
                            plt.scatter(tss_time_series[i - c][0], tss_time_series[i - c][1], color='red', s=8)
                            c_stop += 1
                        else:
                            c_stop += 1
                        if c_stop == 3:
                            break
                else:
                    plt.scatter(tss_time_series[i][0], tss_time_series[i][1], color='orange', s=8)
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

