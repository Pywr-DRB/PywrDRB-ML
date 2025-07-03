import pandas as pd
import numpy as np

def calc_crossval_splits(
    date_range: pd.DatetimeIndex = None,
    min_date: str = None,
    max_date: str = None,
    k_outer: int = 5,
    k_inner: int = 4
) -> list:
    """
    Calculates the start and end date lists for each k-outer and k-inner fold in a nested cross-validation setup.

    Args:
        min_date (str): The minimum date in the date range (inclusive).
        max_date (str): The maximum date in the date range (inclusive).
        k_outer (int, optional): The number of outer folds. Defaults to 5.
        k_inner (int, optional): The number of inner folds. Defaults to 4.

    Returns:
        list: A list of dictionaries, where each dictionary represents an outer fold and contains the start and end date lists for the outer test set and the inner folds.

    Notes:
        The function assumes that the input dates are in a format that can be parsed by pandas.
        The function uses a time-based split for the outer folds, where each outer fold represents a contiguous block of time.
        The function uses a time-based split for the inner folds, where each inner fold represents a contiguous block of time within the outer fold's training set.
    """

    # Calculate the date range
    if date_range is None:
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Calculate the total number of days
    total_days = len(date_range)

    # Calculate the outer fold size
    outer_fold_size = total_days // k_outer
    crossval_folds = []

    # Loop through the outer folds
    for i in range(k_outer):
        # Calculate the start and end dates for the outer test fold
        outer_test_start_date = date_range[i * outer_fold_size]
        outer_test_end_date = date_range[(i + 1) * outer_fold_size - 1] if i < k_outer - 1 else date_range[-1]
        # Calculate the outer test date ranges 
        outer_test_dates = pd.date_range(start=outer_test_start_date,
                                              end=outer_test_end_date,
                                              freq='D')
        # All the dates that aren't test are for training  
        outer_train_dates = [date for date in date_range if date not in outer_test_dates]

        outer_test_start_dates, outer_test_end_dates = find_continuous_periods(outer_test_dates)

        # Calculate the inner train/val fold sizes
        inner_fold_size = len(outer_train_dates) // k_inner
        outer_folds = {
            'outer_fold': i,
            'outer_test_start_dates': outer_test_start_dates,
            'outer_test_end_dates': outer_test_end_dates
        }
        inner_folds = []
        # Loop through the inner folds
        for j in range(k_inner):
            # Grab the dates for validation 
            inner_val_dates = outer_train_dates[(j * inner_fold_size) : (j + 1) * (inner_fold_size)] if j < k_inner - 1 else outer_train_dates[(len(outer_train_dates) - inner_fold_size) : ]
            # all the dates that aren't val are for training 
            inner_train_dates = [date for date in outer_train_dates if date not in inner_val_dates]

            inner_val_start_dates, inner_val_end_dates = find_continuous_periods(inner_val_dates)
            inner_train_start_dates, inner_train_end_dates = find_continuous_periods(inner_train_dates)
            inner_folds.append({
                'inner_fold': j,
                'inner_train_start_dates': inner_train_start_dates,
                'inner_train_end_dates': inner_train_end_dates,
                'inner_val_start_dates': inner_val_start_dates,
                'inner_val_end_dates': inner_val_end_dates
            })
        outer_folds['inner_folds'] = inner_folds
        crossval_folds.append(outer_folds)
    return crossval_folds


def find_continuous_periods(dates):

    dates = sorted(set(dates))  # Remove duplicates and sort the dates
    start_dates = []
    end_dates = []
    start_date = dates[0]
    end_date = dates[0]

    for date in dates[1:]:
        if date - end_date == pd.Timedelta(days=1):
            end_date = date
        else:
            if end_date - start_date > pd.Timedelta(days=5):
                start_dates.append(start_date.strftime('%Y-%m-%d'))
                end_dates.append(end_date.strftime('%Y-%m-%d'))
            start_date = date
            end_date = date
    if end_date - start_date > pd.Timedelta(days=5):
        start_dates.append(start_date.strftime('%Y-%m-%d'))
        end_dates.append(end_date.strftime('%Y-%m-%d'))
    return start_dates, end_dates


if __name__ == "__main__":
    crossval_folds = calc_crossval_splits(
        min_date="2010-01-01", 
        max_date="2024-12-31",
        k_outer=5, 
        k_inner=4)
        
    for outer_fold in crossval_folds:
        print("Outer Fold:", outer_fold['outer_fold'])
        print("Outer Test Start Dates:", outer_fold['outer_test_start_dates'])
        print("Outer Test End Dates:", outer_fold['outer_test_end_dates'])
        for inner_fold in outer_fold['inner_folds']:
            print("Inner Fold:", inner_fold['inner_fold'])
            print("Inner Train Start Dates:", inner_fold['inner_train_start_dates'])
            print("Inner Train End Dates:", inner_fold['inner_train_end_dates'])
            print("Inner Val Start Dates:", inner_fold['inner_val_start_dates'])
            print("Inner Val End Dates:", inner_fold['inner_val_end_dates'])

