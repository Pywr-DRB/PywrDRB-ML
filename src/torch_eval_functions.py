import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def filter_nan_preds(y_true,y_pred):
    """
    Filters out NaN values from the true and predicted values.

    This function removes any entries from the predicted values where the corresponding true values 
    are NaN. This is useful for ensuring that predictions are only considered when there are valid 
    observed values.

    Args:
        y_true (array-like): Observed target values, which may contain NaN entries.
        y_pred (array-like): Predicted target values.

    Returns:
        tuple: A tuple containing two elements:
            - y_true_filtered (array-like): The filtered observed values with NaN entries removed.
            - y_pred_filtered (array-like): The filtered predicted values corresponding to the 
              non-NaN observed values.

    Notes:
        - This function assumes that the input arrays are of the same length.
        - It is important to ensure that the filtering does not lead to mismatched lengths between 
          the true and predicted values.
    """
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    return(y_true, y_pred)

def nse_eval(y_true, y_pred):
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE) between observed and predicted values.

    The NSE is a normalized statistic that determines the relative magnitude of the residual variance
    compared to the variance of the observed data. It ranges from -∞ to 1, where 1 indicates a perfect 
    fit, 0 indicates that the model predictions are as accurate as the mean of the observed data, 
    and negative values indicate worse performance than simply using the mean of the observed data.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The Nash-Sutcliffe Efficiency score.

    Notes:
        - NaN values in the input arrays will be filtered out before calculation.
    """
    y_true, y_pred = filter_nan_preds(y_true,y_pred)
    mean = np.mean(y_true)
    deviation = y_true - mean
    error = y_pred-y_true
    numerator = np.sum(np.square(error))
    denominator = np.sum(np.square(deviation))
    return 1 - numerator / denominator


def rmse_eval(y_true, y_pred):
    """
    Calculates the Root Mean Square Error (RMSE) between observed and predicted values.

    RMSE is a measure of the differences between predicted values and observed values. It is the 
    square root of the average of squared differences, providing a measure of how well the 
    predicted values match the observed values.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The Root Mean Square Error.

    Notes:
        - NaN values in the input arrays will be filtered out before calculation.
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    n = len(y_true)
    sum_squared_error = np.sum(np.square(y_pred-y_true))
    rmse = np.sqrt(sum_squared_error/n)
    return rmse

def bias_eval(y_true,y_pred):
    """
    Calculates the bias between observed and predicted values.

    Bias is the mean difference between predicted values and observed values. A positive bias 
    indicates that predictions are generally higher than observed values, while a negative bias 
    indicates that predictions are generally lower.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The bias value.

    Notes:
        - NaN values in the input arrays will be filtered out before calculation.
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    bias = np.mean(y_pred-y_true)
    return bias

def pbias_eval(y_true,y_pred):
    """
    Calculates the Percent Bias (PBIAS) between observed and predicted values.

    PBIAS is a measure of the average bias expressed as a percentage of the observed data. 
    A PBIAS value of 0 indicates perfect model performance, while positive values indicate 
    overestimation and negative values indicate underestimation.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The Percent Bias value.

    Notes:
        - NaN values in the input arrays will be filtered out before calculation.
    """
    bias = bias_eval(y_true, y_pred)
    pbias = bias/np.mean(y_true)*100
    return pbias

def kge_eval(y_true, y_pred):
    """
    Calculates the Kling-Gupta Efficiency (KGE) between observed and predicted values.

    KGE is a metric that combines three components: correlation, variability, and bias. It is 
    designed to provide a comprehensive measure of how well predicted values match observed values, 
    with a maximum value of 1 indicating perfect agreement.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The Kling-Gupta Efficiency score, or NaN if there are not enough valid observations.

    Notes:
        - NaN values in the input arrays will be filtered out before calculation.
        - KGE requires at least 2 valid observations to compute correlation.
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    #Need to have > 1 observation to compute correlation.
    #This could be < 2 due to percentile filtering
    if len(y_true) > 1:
        r, _ = pearsonr(y_pred, y_true)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        r_component = np.square(r - 1)
        std_component = np.square((std_pred / std_true) - 1)
        bias_component = np.square((mean_pred / mean_true) - 1)
        result = 1 - np.sqrt(r_component + std_component + bias_component)
    else:
        result = np.nan
    return result


def filter_by_percentile(y_true, y_pred, percentile, less_than=True):
    """
    Filters the observed and predicted values based on a specified percentile.

    This function replaces values in the observed and predicted arrays with NaN based on 
    whether they are less than or greater than the specified percentile value of the observed data.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.
        percentile (float): Percentile value between 0 and 100 to filter the data.
        less_than (bool): If True, values less than the percentile will be retained; 
                          if False, values greater than the percentile will be retained.

    Returns:
        tuple: A tuple containing two elements:
            - y_true_filt (array-like): Filtered observed values with values outside the percentile replaced by NaN.
            - y_pred_filt (array-like): Filtered predicted values with values outside the percentile replaced by NaN.
    """
    percentile_val = np.nanpercentile(y_true, percentile)
    if less_than:
        y_true_filt = np.where(y_true < percentile_val, y_true, np.nan)
        y_pred_filt = np.where(y_true < percentile_val, y_pred, np.nan)
    else:
        y_true_filt = np.where(y_true > percentile_val, y_true, np.nan)
        y_pred_filt = np.where(y_true > percentile_val, y_pred, np.nan)
    return y_true_filt, y_pred_filt


def percentile_metric(y_true, y_pred, metric, percentile, less_than=True):
    """
    Computes an evaluation metric for a specified percentile of the observations.

    This function filters the observed and predicted values based on the specified percentile 
    and then applies the provided metric function to the filtered data.

    Args:
        y_true (array-like): Observed target values.
        y_pred (array-like): Predicted target values.
        metric (function): A function that computes a metric (e.g., RMSE, NSE).
        percentile (float): Percentile value between 0 and 100 to filter the data.
        less_than (bool): If True, values less than the percentile will be retained; 
                          if False, values greater than the percentile will be retained.

    Returns:
        float: The computed metric value for the filtered data.
    """
    y_true_filt, y_pred_filt = filter_by_percentile(
        y_true, y_pred, percentile, less_than
    )
    return metric(y_true_filt, y_pred_filt)

def get_confusion_matrix (y_true, y_pred):
    """
    Computes the confusion matrix components for binary classifications.

    This function calculates the number of true positives, true negatives, false positives, 
    and false negatives based on the observed and predicted classifications.

    Args:
        y_true (list-like): Observed binary classifications (0s and 1s).
        y_pred (list-like): Predicted binary classifications (0s and 1s).

    Returns:
        tuple: A tuple containing four integers:
            - True Positives (TP)
            - False Negatives (FN)
            - False Positives (FP)
            - True Negatives (TN)

    Notes:
        - This function assumes that the inputs are binary classifications.
    """
    tp = fn = fp = tn = 0
    for true, pred in zip(y_true, y_pred):
        if true==1:
            if pred==1:
                tp += 1
            else:
                fn += 1
        else:
            if pred==1:
                fp += 1
            else:
                tn += 1
    return (tp, fn, fp, tn)
    

def confusion_matrix_stats(obs, pred, min_obs = 10):
    """
    Computes various statistics from the confusion matrix for binary classifications.

    This function calculates key performance metrics including accuracy, precision, sensitivity, 
    specificity, and the Jaccard index based on the observed and predicted classifications. 
    It also returns the counts of true positives, false negatives, false positives, and true negatives.

    Args:
        obs (list-like): Observed binary classifications (0s and 1s).
        pred (list-like): Predicted binary classifications (0s and 1s).
        min_obs (int, optional): Minimum number of observations required to compute each metric. 
                                 If the number of observations is less than this value, the 
                                 corresponding metric will return NaN. Default is 10.

    Returns:
        dict: A dictionary containing the following statistics:
            - 'Accuracy': The proportion of correct predictions (TP + TN) / total observations.
            - 'Precision': The proportion of true positives among the predicted positives (TP / (TP + FP)).
            - 'Sensitivity': The proportion of true positives among the actual positives (TP / (TP + FN)).
            - 'Specificity': The proportion of true negatives among the actual negatives (TN / (FP + TN)).
            - 'Jaccard': The proportion of true positives among all predicted positives and actual positives 
                         (TP / (TP + FP + FN)).
            - 'tp': Count of true positives.
            - 'fn': Count of false negatives.
            - 'fp': Count of false positives.
            - 'tn': Count of true negatives.

    Notes:
        - This function assumes that the inputs are binary classifications.
        - The performance metrics will return NaN if the minimum number of observations 
          specified by `min_obs` is not met for the respective metric.
    """
    tp, fn, fp, tn = get_confusion_matrix(obs, pred)

    stats_dict = {'Accuracy': (tp+tn)/(len(obs)) if len(obs) >= min_obs else np.nan,
        'Precision': tp/(tp+fp) if (tp + fp) >= min_obs else np.nan,
        'Sensitivity': tp/(tp+fn) if (tp + fn) >= min_obs else np.nan,
        'Specificity': tn/(fp+tn) if (fp + tn) >= min_obs else np.nan,
        'Jaccard': tp/(tp+fp+fn) if (tp+fp+fn) >= min_obs else np.nan,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn
        }
        
    return stats_dict


def calc_metrics(df, min_obs = 10):
    """
    Calculate various evaluation metrics for model predictions against observations.

    This function computes metrics such as RMSE (Root Mean Square Error), NSE (Nash-Sutcliffe Efficiency), 
    and others based on the provided DataFrame containing observed and predicted values. If a "threshold" 
    column is present in the DataFrame, binary exceedance metrics will be calculated.

    Args:
        df (pd.DataFrame): A DataFrame containing the observed and predicted values for one reach. 
                           The DataFrame must have the following columns:
                           - "obs": Observed values.
                           - "pred": Predicted values.
                           - "threshold" (optional): If present, binary exceedance metrics will be calculated.
        min_obs (int, optional): Minimum number of observations required for metrics to be calculated. 
                                  If the number of observations is less than this value, NaN will be returned 
                                  for all metrics. Default is 10.

    Returns:
        pd.Series: A Pandas Series containing various evaluation metrics, including:
            - 'rmse': Root Mean Square Error.
            - 'rmse_top10': RMSE for the top 10% of predictions.
            - 'rmse_bot10': RMSE for the bottom 10% of predictions.
            - 'rmse_bot90': RMSE for the bottom 90% of predictions.
            - 'mean_bias': Mean bias of predictions.
            - 'mean_bias_top10': Mean bias for the top 10% of predictions.
            - 'mean_bias_bot10': Mean bias for the bottom 10% of predictions.
            - 'mean_bias_bot90': Mean bias for the bottom 90% of predictions.
            - 'pbias': Percent Bias.
            - 'pbias_top10': Percent Bias for the top 10% of predictions.
            - 'pbias_bot10': Percent Bias for the bottom 10% of predictions.
            - 'pbias_bot90': Percent Bias for the bottom 90% of predictions.
            - 'nse': Nash-Sutcliffe Efficiency.
            - 'nse_top10': NSE for the top 10% of predictions.
            - 'nse_bot10': NSE for the bottom 10% of predictions.
            - 'nse_bot90': NSE for the bottom 90% of predictions.
            - 'kge': Kling-Gupta Efficiency.
            - 'kge_top10': KGE for the top 10% of predictions.
            - 'kge_bot10': KGE for the bottom 10% of predictions.
            - 'kge_bot90': KGE for the bottom 90% of predictions.
            - 'n_obs': Total number of observations used for metric calculations.

    Notes:
        - The function assumes that the DataFrame contains numerical values in the "obs" and "pred" columns.
        - If the number of observations is less than `min_obs`, all metrics will return NaN.
        - The function requires additional helper functions (e.g., `rmse_eval`, `nse_eval`, etc.) to compute the specific metrics.
    """
    obs = df["obs"].values
    pred = df["pred"].values
    obs, pred = filter_nan_preds(obs, pred)
    
    if "threshold" in df.columns:
        obs_bin = [1 if x != 0 else 0 for x in obs]
        pred_bin = [1 if x != 0 else 0 for x in pred]
        metrics = confusion_matrix_stats(obs_bin, pred_bin, min_obs)
    else:
        if len(obs) >= min_obs:
            metrics = {
                "rmse": rmse_eval(obs, pred),
                "rmse_top10": percentile_metric(
                    obs, pred, rmse_eval, 90, less_than=False
                ),
                "rmse_bot10": percentile_metric(
                    obs, pred, rmse_eval, 10, less_than=True
                ),
                "rmse_bot90": percentile_metric(
                    obs, pred, rmse_eval, 90, less_than=True
                ),
                "mean_bias": bias_eval(obs,pred),
                "mean_bias_top10":percentile_metric(
                    obs, pred, bias_eval, 90, less_than=False
                ),
                "mean_bias_bot10": percentile_metric(
                    obs, pred, bias_eval, 10, less_than=True
                ),
                "mean_bias_bot90": percentile_metric(
                    obs, pred, bias_eval, 90, less_than=True
                ),
                "pbias": pbias_eval(obs, pred),
                "pbias_top10":  percentile_metric(
                    obs, pred, pbias_eval, 90, less_than=False
                ),
                "pbias_bot10": percentile_metric(
                    obs, pred, pbias_eval, 10, less_than=True
                ),
                "pbias_bot90": percentile_metric(
                    obs, pred, pbias_eval, 90, less_than=True
                ),
                "nse": nse_eval(obs, pred),
                "nse_top10": percentile_metric(
                    obs, pred, nse_eval, 90, less_than=False
                ),
                "nse_bot10": percentile_metric(
                    obs, pred, nse_eval, 10, less_than=True
                ),
                "nse_bot90": percentile_metric(
                    obs, pred, nse_eval, 90, less_than=True
                ),
                "kge": kge_eval(obs, pred),
                "kge_top10": percentile_metric(obs, pred, kge_eval, 90, less_than=False),
                "kge_bot10": percentile_metric(obs, pred, kge_eval, 10, less_than=True),
                "kge_bot90": percentile_metric(obs, pred, kge_eval, 90, less_than=True),
                "n_obs": len(obs)
            }
        else:
            metrics = {
                "rmse": np.nan,
                "rmse_top10": np.nan,
                "rmse_bot10": np.nan,
                "rmse_bot90": np.nan,
                "mean_bias": np.nan,
                "mean_bias_top10": np.nan,
                "mean_bias_bot10": np.nan,
                "mean_bias_bot90": np.nan,
                "pbias": np.nan,
                "pbias_top10": np.nan,
                "pbias_bot10": np.nan,
                "pbias_bot90": np.nan,
                "nse": np.nan,
                "nse_top10": np.nan,
                "nse_bot10": np.nan,
                "nse_bot90": np.nan,
                "kge": np.nan,
                "kge_top10": np.nan,
                "kge_bot10": np.nan,
                "kge_bot90": np.nan,
                "n_obs": len(obs)
            }
    return pd.Series(metrics)

