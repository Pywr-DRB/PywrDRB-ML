import xarray as xr
import numpy as np
import yaml
from yaml.loader import SafeLoader
import sys


def read_config(config_file):
    """
    read in the configuration file and return contents
    """
    # Open the file and load the file
    with open(config_file) as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data

def scale(dataset, std=None, mean=None):
    """
    scale the data so it has a standard deviation of 1 and a mean of zero
    :param dataset: [xr dataset] input or output data
    :param std: [xr dataset] standard deviation if scaling test data with dims
    :param mean: [xr dataset] mean if scaling test data with dims
    :return: scaled data with original dims
    """

    if not isinstance(std, xr.Dataset) or not isinstance(mean, xr.Dataset):
        std = dataset.std(skipna=True)
        mean = dataset.mean(skipna=True)

    # adding small number in case there is a std of zero
    scaled = (dataset - mean) / (std + 1e-10)

    check_if_finite(std)
    check_if_finite(mean)
    return scaled, std, mean

# from river-dl
def split_into_batches(data_array, seq_len=365, offset=1.0,
                       fill_batch=True, fill_nan=False, fill_time=False,
                       fill_pad=False):
    """
    split training data into batches with size of seq_len
    :param data_array: [numpy array] array of training data with dims [nseg,
    ndates, nfeat]
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] How to offset the batches. Values < 1 are taken as fractions, (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547), values > 1 are used as a constant number of
    observations to offset by.
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because
    data_array is observation data that should not contribute to the loss).
    When False, filled in data are replicates of the previous timesteps.
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data
    are replicates of the previous timesteps.
    :param fill_pad: [bool] When True, the returned data are bool indicating
    True when it is padded and False otherwise. When False, filled in data
    are based on other fill_ rules.
    :return: [numpy array] batched data with dims [nbatches, nseg, seq_len
    (batch_size), nfeat]
    """
    if offset>1:
        period = int(offset)
    else:
        period = int(offset*seq_len)

    nsteps = data_array.shape[1]
    num_batches = nsteps//period
    if fill_batch:
        final_batch_shape_check = nsteps - period*num_batches
        if (final_batch_shape_check != seq_len):
            #append timesteps to the data_array
            #Determine how many timesteps to replicate to get a full batch
            num_rep_steps = seq_len - final_batch_shape_check

            if fill_nan:
                #fill in with nan values
                nan_array = np.empty((data_array.shape[0],
                                      num_rep_steps,
                                      data_array.shape[2]))
                nan_array.fill(np.nan)
                data_array = np.concatenate((data_array,
                                             nan_array),
                                            axis = 1)
            elif fill_pad:
                #fill in with True for padded values and False otherwise
                False_array = np.empty((data_array.shape[0],
                                      data_array.shape[1],
                                      data_array.shape[2]),
                                      dtype=bool)
                True_array = np.empty((data_array.shape[0],
                                      num_rep_steps,
                                      data_array.shape[2]),
                                      dtype=bool)
                False_array.fill(False)
                True_array.fill(True)
                data_array = np.concatenate((False_array,
                                             True_array),
                                            axis = 1)
            else:
                #fill in by replicating the previous timesteps in the data_array
                if fill_time:
                    #data are an np.datetime64 object. These must be unique, so
                    #cannot be replicated. Add timesteps sequentially
                    fill_dates_array = data_array[:,(nsteps-num_rep_steps):nsteps,:].copy()
                    #add num_rep_steps to each index.
                    # Sending the smallest possible sample of data_array to the function
                    time_unit = get_time_unit(data_array[0:2,0:2,0:2])
                    fill_dates_array = fill_dates_array[:,:,:] + np.timedelta64(num_rep_steps, time_unit)

                    data_array = np.concatenate((data_array,
                                                 fill_dates_array),
                                                axis = 1)
                else:
                    data_array = np.concatenate((data_array,
                                                 data_array[:,(nsteps-num_rep_steps):nsteps,:]),
                                                axis = 1)

            num_batches = num_batches+1

        elif fill_pad:
            #return array of False - no padded data
            False_array = np.empty((data_array.shape[0],
                                  data_array.shape[1],
                                  data_array.shape[2]),
                                  dtype=bool)
            False_array.fill(False)
            data_array = False_array

    elif fill_pad:
        #return array of False - no padded data
        False_array = np.empty((data_array.shape[0],
                              data_array.shape[1],
                              data_array.shape[2]),
                              dtype=bool)
        False_array.fill(False)
        data_array = False_array


    combined=[]
    for i in range(num_batches+1):
        idx = int(period*i)
        batch = data_array[:,idx:idx+seq_len,...]
        combined.append(batch)
    combined = [b for b in combined if b.shape[1]==seq_len]
    combined = np.asarray(combined)
    return combined

def convert_batch_reshape(
    dataset,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    offset=1.0,
    fill_batch=True,
    fill_nan=False,
    fill_time=False,
    fill_pad=False
):
    """
    convert xarray dataset into numpy array, swap the axes, batch the array and
    reshape for training
    :param dataset: [xr dataset] data to be batched
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because
    data_array is observation data that should not contribute to the loss).
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data
    are replicates of the previous timesteps.
    :param fill_pad: [bool] When True, the returned data are bool indicating
    True when it is padded and False otherwise. When False, filled in data
    are based on other fill_ rules.
    :return: [numpy array] batched and reshaped dataset
    """
    # If there is no dataset (like if a test or validation set is not supplied)
    # just return None
    if not dataset:
        return None

    if fill_batch:
        continuous_start_inds = []
        #Check if there are gaps in the timeseries as a result of using a
        # discontinuous partition.
        #identify all gaps greater than the 1st timestep
        time_diff = np.diff(dataset[time_idx_name])
        timestep_1 = time_diff[0]
        if any(time_diff != timestep_1):
            #fill those gaps based on the sequence length.
            gap_timesteps = np.delete(np.unique(time_diff),
                                      np.where(np.unique(time_diff) == timestep_1))
            for t in gap_timesteps:
                #using [0] to return an array
                gap_ind = np.where(time_diff == t)[0]
                #there can be multiple gaps of the same length
                for i in gap_ind:
                    #determine if the gap is longer than the sequence length
                    date_before_gap = dataset[time_idx_name][i].values
                    next_date = dataset[time_idx_name][i+1].values
                    #gap length in the same unit as the timestep
                    gap_length = int((next_date - date_before_gap)/timestep_1)
                    if gap_length < seq_len:
                        #I originally had this as a sys.exit, but did not want
                        # to force this condition.
                        print("The gap between this partition's continuous time periods is less than the sequence length")

                    #get the start date indices. These are used to split the
                    # dataset before creating batches
                    continuous_start_inds.append(i+1)

            continuous_start_inds = np.sort(continuous_start_inds)

    # convert xr.dataset to numpy array
    dataset = dataset.transpose(spatial_idx_name, time_idx_name)

    arr = dataset.to_array().values

    # if the dataset is empty, just return it as is
    if dataset[time_idx_name].size == 0:
        return arr

    # before [nfeat, nseg, ndates]; after [nseg, ndates, nfeat]
    # this is the order that the split into batches expects
    arr = np.moveaxis(arr, 0, -1)

    # batch the data
    # after [nbatch, nseg, seq_len, nfeat]
    if fill_batch:
        if len(continuous_start_inds) != 0:
            #using a discontinuous partition. create a set of batches for each
            # continuous period in this partion and join
            for g in range(len(continuous_start_inds)+1):
                if g == 0:
                    arr_g = arr[:,0:continuous_start_inds[g],:].copy()

                    batched = split_into_batches(arr_g, seq_len=seq_len, offset=offset,
                                                 fill_batch=fill_batch, fill_nan=fill_nan,
                                                 fill_time=fill_time, fill_pad=fill_pad)

                else:
                    if g == len(continuous_start_inds):
                        arr_g = arr[:,continuous_start_inds[g-1]:arr.shape[1],:].copy()
                    else:
                        arr_g = arr[:,continuous_start_inds[g-1]:continuous_start_inds[g],:].copy()

                    batched_g = split_into_batches(arr_g, seq_len=seq_len, offset=offset,
                                                   fill_batch=fill_batch, fill_nan=fill_nan,
                                                   fill_time=fill_time, fill_pad=fill_pad)
                    batched = np.append(batched, batched_g, axis = 0)
        else:
            batched = split_into_batches(arr, seq_len=seq_len, offset=offset,
                                         fill_batch=fill_batch, fill_nan=fill_nan,
                                         fill_time=fill_time, fill_pad=fill_pad)
    else:
        batched = split_into_batches(arr, seq_len=seq_len, offset=offset,
                                     fill_batch=fill_batch, fill_nan=fill_nan,
                                     fill_time=fill_time, fill_pad=fill_pad)

    # reshape data
    # after [nbatch * nseg, seq_len, nfeat]
    reshaped = reshape_for_training(batched)
    return reshaped

def separate_trn_tst(
    dataset,
    time_idx_name,
    train_start_date,
    train_end_date,
    val_start_date=None,
    val_end_date=None,
    test_start_date=None,
    test_end_date=None,
    spatial_idx_name="seg_id_nat",
    y_vars=None,
    withheld_ids=None,
):
    """
    separate the train data from the test data according to the start and end
    dates. This assumes your training data is in one continuous block. Be aware,
    if your train/test/val partitions are discontinuous (composed of multiple
    periods), depending on your sequence length and how the data line up, you
    could end up with sequences starting in one period and ending in another.
    The breaking up of sequences would happen in the `convert_batch_reshape`
    function
    :param dataset: [xr dataset] input or output data with dims
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param train_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    train period (can have multiple discontinuous periods)
    :param train_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end train
     period (can have multiple discontinuous periods)
    :param val_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
     validation period (can have multiple discontinuous periods)
    :param val_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end
    validation period (can have multiple discontinuous periods)
    :param test_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    test period (can have multiple discontinuous periods)
    :param test_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end test
    period (can have multiple discontinuous periods)
    :param withheld_ids: [str or list] id(s) to withhold from training and validation
     setting the observations to NA's
    :return: [tuple] separated data
    """
    train = sel_partition_data(
        dataset, time_idx_name, train_start_date, train_end_date, spatial_idx_name, y_vars, withheld_ids
    )

    if val_start_date and val_end_date:
        val = sel_partition_data(
            dataset, time_idx_name, val_start_date, val_end_date, spatial_idx_name, y_vars, withheld_ids
        )

    elif val_start_date and not val_end_date:
        raise ValueError("With a val_start_date a val_end_date must be given")
    elif val_end_date and not val_start_date:
        raise ValueError("With a val_end_date a val_start_date must be given")
    else:
        val = None

    if test_start_date and test_end_date:
        test = sel_partition_data(
            dataset, time_idx_name, test_start_date, test_end_date
        )
    elif test_start_date and not test_end_date:
        raise ValueError("With a test_start_date a test_end_date must be given")
    elif test_end_date and not test_start_date:
        raise ValueError("With a test_end_date a test_start_date must be given")
    else:
        test = None

    return train, val, test

def sel_partition_data(dataset,
                       time_idx_name,
                       start_dates,
                       end_dates,
                       spatial_idx_name="seg_id_nat",
                       y_vars=None,
                       withheld_ids=None
):
    """
    select the data from a date range or a set of date ranges
    :param dataset: [xr dataset] input or output data with date dimension
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param start_dates: [str or list] fmt: "YYYY-MM-DD"; date(s) to start period
    (can have multiple discontinuos periods)
    :param end_dates: [str or list] fmt: "YYYY-MM-DD"; date(s) to end period
    (can have multiple discontinuos periods)
    :return: dataset of just those dates
    """
   # if it just one date range
    if isinstance(start_dates, str):
        if isinstance(end_dates, str):
            out = dataset.sel({time_idx_name: slice(start_dates, end_dates)})
        else:
            raise ValueError("start_dates is str but not end_date")
    # if it's a list of date ranges
    elif isinstance(start_dates, list) or isinstance(start_dates, tuple):
        if len(start_dates) == len(end_dates):
            data_list = []
            for i in range(len(start_dates)):
                date_slice = slice(start_dates[i], end_dates[i])
                data_list.append(dataset.sel({time_idx_name: date_slice}))
            out = xr.concat(data_list, dim=time_idx_name)
        else:
            raise ValueError("start_dates and end_dates must have same length")
    else:
        raise ValueError("start_dates must be either str, list, or tuple")
    # if there are withheld ids, then set those observations to NA's
    if withheld_ids:
        out = withhold_sites(out, spatial_idx_name, y_vars, withheld_ids)

    return out


def withhold_sites(dataset,
                  spatial_idx_name,
                  y_vars,
                  withheld_ids
):
    """
    Sets y_vars to NA's for sites we want to withhold during training.
    :param dataset: [xr dataset] input dataset
    :param spatial_idx_name:
    :param y_vars:
    :param withheld_ids: [str or list] id(s) to withhold from training and validation
     setting the observations to NA's
    """
    spatial_mask = np.logical_not(np.isin(dataset[spatial_idx_name], withheld_ids))
    len_dates = dataset.sizes['Date']
    spatial_mask_rep = np.tile(spatial_mask, (len_dates, 1))
    spatial_mask_rep = spatial_mask_rep.swapaxes(0,1)

    masked_y_vars = dataset[y_vars].where(spatial_mask_rep, np.nan)

    dataset.update(masked_y_vars) # update y_vars in the original xarray with the masked y_vars for withheld sites
    return dataset


def reshape_for_training(data):
    """
    reshape the data for training
    :param data: training data (either x or y_dataset or mask) dims: [nbatch, nseg,
    len_seq, nfeat/nout]
    :return: reshaped data [nbatch * nseg, len_seq, nfeat/nout]
    """
    n_batch, n_seg, seq_len, n_feat = data.shape
    return np.reshape(data, [n_batch * n_seg, seq_len, n_feat])

def check_if_finite(xarr):
    assert np.isfinite(xarr.to_array().values).all()


def coord_as_reshaped_array(
    dataset,
    coord_name,
    spatial_idx_name="StaID",
    time_idx_name="Date",
    seq_len=365,
    offset=1.0,
    fill_batch=True,
    fill_nan=False,
    fill_time=False,
    fill_pad=False
):
    """
    convert an xarray coordinate to an xarray data array and reshape that array
    :param dataset:
    :param coord_name: [str] the name of the coordinate to convert/reshape
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'StaID')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'Date')
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because
    data_array is observation data that should not contribute to the loss).
    When False, filled in data are replicates of the previous timesteps.
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data
    are replicates of the previous timesteps.
    :param fill_pad: [bool] When True, the returned data are bool indicating
    True when it is padded and False otherwise. When False, filled in data
    are based on other fill_ rules.
    :return:
    """
    # If there is no dataset (like if a test or validation set is not supplied)
    # just return None
    if not dataset:
        return None

    # I need one variable name. It can be any in the dataset, but I'll use the
    # first
    first_var = next(iter(dataset.data_vars.keys()))
    coord_array = xr.broadcast(dataset[coord_name], dataset[first_var])[0]
    new_var_name = coord_name + "1"
    dataset[new_var_name] = coord_array
    reshaped_np_arr = convert_batch_reshape(
        dataset[[new_var_name]],
        spatial_idx_name,
        time_idx_name,
        seq_len=seq_len,
        offset=offset,
        fill_batch=fill_batch,
        fill_nan=fill_nan,
        fill_time=fill_time,
        fill_pad=fill_pad
    )
    return reshaped_np_arr

def get_time_unit(data_array):
    '''
    Function to get the timestep unit from a numpy.datetime64 array column

    :param data_array: [np 3D array] the array's second axis must be time
    in np.datetime64 format with nanoseconds specified (default).

    returns the unit of the timestep (day, month, etc.)
    '''
    time_delta = data_array[0,1,0] - data_array[0,0,0]
    time_unit = np.datetime_data(time_delta)[0]
    if time_unit != 'ns':
        sys.exit('time unit must be provided as YYYY-MM-DDT:HH:MM:SS.000000000 nanoseconds')

    if time_delta == 86400000000000:
        time_unit = 'D'
    elif time_delta == 24000000000:
        time_unit = 'h'
    else:
        sys.exit('time_delta does not correspond to 1 h or D')

    return(time_unit)
