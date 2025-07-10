import numpy as np
import pandas as pd
from typing import Union, List
from numpy.lib.stride_tricks import sliding_window_view # Create sliding windows using np.lib.stride_tricks for efficiency

def compute_reliability(
    df: pd.DataFrame, col: str = 'T_L_mu', threshold: float = 24, quantile: float = 0.05, only_summer_period: bool = True, return_distribution: bool = False
    ) -> float:
    """
    Compute JRel(θ): Average reliability of 10-day moving windows for values below the 0.05 quantile.

    Based on the formula:
    JRel = (1/REL0.05) * Σ(rel∈REL) rel

    Where:
    - REL0.05 = {rel ∈ REL | rel ≤ Quantile(REL, 0.05)}
    - REL = {frel(Y(t:t+9)) | t ∈ [0, T - 9]}
    - Y(t:t+9) = {yt, ..., yt+9}
    - frel(·) = |{y∈Y(t:t+9) | y≤24}| / |Y(t:t+9)|

    Parameters
    ----------
    Y : array-like
        Time series data (e.g., temperature values)
    threshold : float, default=24
        Threshold value for computing reliability (e.g., 24°C)
    quantile : float, default=0.05
        Quantile level to determine the threshold for reliability values
    return_distribution : bool, default=False
        If True, return the distribution of reliability values instead of the average

    Returns
    -------
    float
        JRel reliability metric
    """
    df = df.copy()
    if only_summer_period:
        df.loc[(df.index.month < 6) | (df.index.month > 8), col] = np.nan

    # Create 10-day sliding windows
    windows = sliding_window_view(df[col].values, window_shape=10)

    # Compute reliability for each window
    REL = np.mean(windows <= threshold, axis=1)
    REL = REL[~np.any(np.isnan(windows), axis=1)] # if the roll have nan

    # Compute quantile of REL
    quantile_value = np.quantile(REL, quantile)

    # Find REL0.05: values below or equal to 0.05 quantile
    REL_quantile = REL[REL <= quantile_value]

    # Compute JRel: average reliability of values below quantile
    JRel = np.mean(REL_quantile)

    if return_distribution:
        return REL
    return round(float(JRel), 4)


def compute_max_annual_accumulated_degree_days(
    df: pd.DataFrame, col: str = 'Tavg_L_mu', threshold: float = 20, only_summer_period: bool = True, max_Jadd: float = 132.4373, return_distribution: bool = False
    ) -> float:
    """
    Compute JADD(θ): Maximum annual accumulated degree days.

    Based on the formula:
    JADD = Max{ADD}
    ADD = {addyr | yr ∈ [2018, 2024]}
    addyr = Σt∈{t | year(t)=yr,t∈[0,T]} max(0, yt - 20)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and temperature values
    col : str, default='Tavg_L'
        Column name for temperature values
    threshold : float, default=20
        Base temperature threshold for degree days calculation (e.g., 20°C)
    return_distribution : bool, default=False
        If True, return the distribution of annual accumulated degree days instead of the maximum

    Returns
    -------
    float
        JADD maximum annual accumulated degree days
    """
    df = df.copy()
    df['year'] = df.index.year

    if only_summer_period:
        df.loc[(df.index.month < 6) | (df.index.month > 8), col] = np.nan

    # Compute degree days for each observation
    df['degree_days'] = np.maximum(0, df[col] - threshold)

    # Compute annual sums (ADD)
    annual_sums = df.groupby('year')['degree_days'].sum()

    # Return maximum annual sum (JADD)
    JADD = annual_sums.max()

    if return_distribution:
        return annual_sums.values
    return round(float(JADD/max_Jadd), 4)

def compute_max_thermal_bank_usage_ratio(
    df: pd.DataFrame, col: str = 'remained_bank_amounts', bank_size: float = 1620, return_distribution: bool = False, last_date_of_ctrl: tuple = (8, 31)
    ) -> float:
    """
    Compute JTBUR(θ): Average annual thermal mitigation bank usage ratio.

    Based on the formula:
    JTBUR = max{tburyr}
    TBUR = {tburyr | yr ∈ years}
    tburyr = Σt∈{t | year(t)=yr,t∈[0,T]} ut

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing thermal bank usage data
    col : str, default='ratios'
        Column name for remained bank amounts
    bank_size : float, default=1620
        Size of the thermal mitigation bank (e.g., 1620 MGD)
    return_distribution : bool, default=False
        If True, return the distribution of ratios instead of the average

    Returns
    -------
    float
        JTBUR average annual thermal mitigation bank usage ratio
    """
    df = df.copy()
    df = df[(df.index.month == last_date_of_ctrl[0]) & (df.index.day == last_date_of_ctrl[1])]
    df["ratios"] = 1 - df[col] / bank_size  # Normalize by bank size

    # Compute JTBUR
    JTBUR = np.max(df["ratios"])

    if return_distribution:
        return df["ratios"].values
    return round(float(JTBUR), 4)

def compute_mean_thermal_bank_usage_ratio(
    df: pd.DataFrame, col: str = 'remained_bank_amounts', bank_size: float = 1620, return_distribution: bool = False, last_date_of_ctrl: tuple = (8, 31)
    ) -> float:
    """
    Compute JTBUR(θ): Average annual thermal mitigation bank usage ratio.

    Based on the formula:
    JTBUR = (1/|TBUR|) * Σtbur∈TBUR tbur
    TBUR = {tburyr | yr ∈ years}
    tburyr = Σt∈{t | year(t)=yr,t∈[0,T]} ut

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing thermal bank usage data
    col : str, default='ratios'
        Column name for remained bank amounts
    bank_size : float, default=1620
        Size of the thermal mitigation bank (e.g., 1620 MGD)
    return_distribution : bool, default=False
        If True, return the distribution of ratios instead of the average

    Returns
    -------
    float
        JTBUR average annual thermal mitigation bank usage ratio
    """
    df = df.copy()
    df = df[(df.index.month == last_date_of_ctrl[0]) & (df.index.day == last_date_of_ctrl[1])]
    df["ratios"] = 1 - df[col] / bank_size  # Normalize by bank size

    # Compute JTBUR
    JTBUR = np.mean(df["ratios"])

    if return_distribution:
        return df["ratios"].values
    return round(float(JTBUR), 4)

