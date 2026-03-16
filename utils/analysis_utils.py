import os
import time
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import fastnanquantile as fnq
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


def _apply_nodata_masking(a, forest_mask, clip=True):
    """Gemeinsames NaN/NoData-Handling und optionales Clipping."""
    a[np.isnan(a)] = -32768
    missing_values = np.logical_and(a == -32768, ~forest_mask)
    a[missing_values] = 5000

    if clip:
        mask_special = np.isin(a, [5000, -32768])
        a_clipped = np.clip(a, -4000, 4000)
        a_clipped[mask_special] = a[mask_special]
        a = a_clipped

    return a.astype(np.int16)


def _compute_metric(array, metric):
    """
    Berechnet eine einzelne Metrik auf einem 3D-Array (x, y, time).
    Gibt ein 2D-Array (x, y) zurück.
    Hier können künftig weitere Metriken ergänzt werden.
    """
    if metric == 'p10_p90':
        positive_values = np.sum((array > 0), axis=2)
        negative_values = np.sum(array <= 0, axis=2)
        mask_positive = positive_values > negative_values
        result = np.full(array.shape[:2], np.nan)

        if np.any(mask_positive):
            result[mask_positive] = fnq.nanquantile(
                array[mask_positive], 0.9, axis=1
            )
        mask_negative = ~mask_positive
        if np.any(mask_negative):
            result[mask_negative] = fnq.nanquantile(
                array[mask_negative], 0.1, axis=1
            )
        return result

    elif metric == 'median':
        return fnq.nanquantile(array, 0.5, axis=2)

    elif metric == 'mean':
        # check if cumulative sum of negative or positive residuals is bigger
        sum_pos = np.nansum(np.where(array > 0, array, 0), axis=2)
        sum_neg = np.nansum(np.where(array < 0, array, 0), axis=2)
        mask_positive = sum_pos > np.abs(sum_neg)

        result = np.full(array.shape[:2], np.nan)

        # only use positive values if positive values are bigger
        pos_values = np.where(array > 0, array, np.nan)
        result[mask_positive] = np.nanmean(pos_values[mask_positive], axis=1)

        # only use negative values if negative values are bigger
        neg_values = np.where(array < 0, array, np.nan)
        mask_negative = ~mask_positive
        result[mask_negative] = np.nanmean(neg_values[mask_negative], axis=1)

        return result

    # elif metric == 'std':
    #     return np.nanstd(array, axis=2)

    else:
        raise ValueError(f"unknown metric: '{metric}'. "
                         f"available metrics: 'p10_p90', 'median'")


def calculate_firstdate_whole(output_array_full, dates_nrt, forest_mask):
    print("###" * 10)
    print(f'calculate first date (residual related)\n')

    first_non_nan_indices = np.argmax(~np.isnan(output_array_full), axis=2)
    output_array = np.full((3000, 3000), np.nan)

    for x in range(output_array.shape[0]):
        for y in range(output_array.shape[1]):
            index = first_non_nan_indices[x, y]
            if not np.isnan(output_array_full[x, y, index]):
                output_array[x, y] = int(dates_nrt[index].replace('-', '')[2:])

    output_array[np.isnan(output_array)] = -32768
    missing_values = np.logical_and(output_array == -32768, ~forest_mask)
    output_array[missing_values] = 5000
    output_array = output_array.astype(int)

    return output_array


def calculate_vit_whole(output_array_full, forest_mask, metrics=None):
    if metrics is None:
        metrics = ['median']

    print("###" * 10)
    print(f'calculate metrics for whole time period (residual related)')
    print(f'metrics: {metrics}\n')

    results = {}
    for metric in metrics:
        print(f'  computing {metric} ...')
        result = _compute_metric(output_array_full, metric)
        results[metric] = _apply_nodata_masking(result, forest_mask, clip=True)

    return results


def calculate_vit_period(raster_tss, output, start_date, end_date, period_length,
                             dates_nrt, output_array_full, forest_mask, metrics=None, filtered=None, mode=None):
    if metrics is None:
        metrics = ['median']

    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }

    date = start_date
    print("###" * 10)
    print(f'starting slicing for time periods! (residual related)')
    print(f'  start date:     {start_date}')
    print(f'  end date:       {end_date}')
    print(f'  period length:  {period_length} month')
    print(f'  metrics:        {metrics}\n')

    while date < end_date:
        start_month = datetime.strptime(date, '%Y-%m').strftime('%B %Y')
        end_month = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length - 1)).strftime('%B %Y')

        sm_split = start_month.split(" ")
        em_split = end_month.split(" ")

        if sm_split[0] == 'December':
            print(f"period started with December – skipped.")
            date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')
            continue

        sliced_array = np.array(slice_by_date(output_array_full, dates_nrt, date, period_length))

        if mode == "thresholding":
            sliced_filter = slice_by_date(filtered, dates_nrt, date, period_length)
            sliced_filter_2d = np.any(sliced_filter, axis=2)

        date = (datetime.strptime(date, '%Y-%m') + relativedelta(months=period_length)).strftime('%Y-%m')

        print("###" * 10)
        print(f'sliced array for {start_month} - {end_month}')
        print(f'array shape: {sliced_array.shape}')

        sm_month = month_map[sm_split[0].lower()]
        em_month = month_map[em_split[0].lower()]

        for metric in metrics:
            print(f'  computing {metric} ...')
            a = _compute_metric(sliced_array, metric)

            # Thresholding: NaN-Bereiche aus Filter übernehmen
            if mode == "thresholding":
                sliced_filter_2d_nan = np.logical_and(a == -32768, sliced_filter_2d)
                a[sliced_filter_2d_nan] = -32768

            a = _apply_nodata_masking(a, forest_mask, clip=True)

            suffix = f"/vit_{sm_split[1]}{sm_month}_{em_split[1]}{em_month}_{mode}_{metric}_v3.tif"
            write_output_raster(raster_tss, output, a, suffix, 1)