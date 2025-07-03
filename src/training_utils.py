import torch
import numpy as np
import time
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


def rmse_masked(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values,
    ignoring NaN values in the true values.

    This function computes the RMSE by masking out the NaN values in the `y_true` tensor.
    The RMSE is calculated only on the non-NaN elements of `y_true`.

    Args:
        y_true (torch.Tensor): A tensor containing the true values. NaN values will be ignored.
        y_pred (torch.DataFrame): A DataFrame or similar structure containing the predicted values.
                                  It should have a column named 'y_hat'.

    Returns:
        torch.Tensor: The calculated RMSE value as a tensor.

    Example:
        >>> y_true = torch.tensor([3.0, 5.0, float('nan'), 7.0])
        >>> y_pred = pd.DataFrame({'y_hat': torch.tensor([2.5, 5.5, 6.0, 7.5])})
        >>> rmse_masked(y_true, y_pred)
        tensor(0.6124)
    """
    y_pred = y_pred['y_hat']
    num_y_true = torch.count_nonzero(
        ~torch.isnan(y_true)
    )
    zero_or_error = torch.where(
        torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
    )
    sum_squared_errors = torch.sum(torch.square(zero_or_error))
    rmse_loss = torch.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss

# GPT to solve zero seg_result issue
def MaskedGMMLoss(y, prediction, eps=1e-10):
    """
    Numerically stable average negative log-likelihood for a GMM with missing value masking.
    """

    m = prediction['mu']         # (B, T, K)
    s = prediction['sigma']      # (B, T, K)
    p = prediction['pi']         # (B, T, K)

    # Clamp sigma to avoid divide-by-zero or log(0)
    s = torch.clamp(s, min=1e-3, max=1e2)

    # In case y is (B, T, 1), match shape
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    mask = ~torch.isnan(y)       # (B, T, 1)
    losses = []

    for i in range(y.shape[0]):  # Loop over batch
        seg_mask = mask[[i]].any(0).any(-1)  # Select valid time steps for this batch sample

        if torch.sum(seg_mask) > 0:
            seg_y = y[[i]][:, seg_mask, :]        # (1, T, 1)
            seg_m = m[[i]][:, seg_mask, :]        # (1, T, K)
            seg_s = s[[i]][:, seg_mask, :]        # (1, T, K)
            seg_p = p[[i]][:, seg_mask, :]        # (1, T, K)

            # Broadcast y to match GMM shape (1, T, K)
            seg_y_exp = seg_y.expand_as(seg_m)

            # log(pi) + log N(y | mu, sigma)
            log_coeff = torch.log(seg_p + eps)  # (1, T, K)
            log_normal = -0.5 * ((seg_y_exp - seg_m) / seg_s)**2 \
                         - torch.log(seg_s + eps) \
                         - 0.5 * np.log(2 * np.pi)

            log_prob = log_coeff + log_normal  # (1, T, K)

            # log-sum-exp over K components
            log_sum_exp = torch.logsumexp(log_prob, dim=-1)  # (1, T)

            losses.append(-log_sum_exp)  # Negative log-likelihood

    # Concatenate all valid batches and time steps
    if len(losses) > 0:
        loss_tensor = torch.cat(losses, dim=1)  # (1, total_valid_T)
        final_loss = torch.mean(loss_tensor)
    else:
        final_loss = torch.tensor(0.0, device=y.device)

    return final_loss

def MaskedGMMLoss_org(y, prediction, eps = 1e-10):
    """
    Calculate the average negative log-likelihood for a Gaussian Mixture Model (GMM).

    This loss function computes the negative log-likelihood for GMMs, which is commonly used
    in mixture density networks. The implementation handles missing values in the target
    variable `y` by applying a mask, ensuring that only valid (non-NaN) values contribute
    to the loss calculation.

    Parameters
    ----------
    y : torch.Tensor
        The target values (observations) for which the GMM predicts probabilities.
        It should be a tensor of shape (n_samples, n_features).

    prediction : dict
        A dictionary containing the parameters of the GMM:
        - 'mu' (torch.Tensor): The means of the Gaussian components, shape (n_samples, n_components, n_features).
        - 'sigma' (torch.Tensor): The standard deviations of the Gaussian components, shape (n_samples, n_components, n_features).
        - 'pi' (torch.Tensor): The mixture weights (prior probabilities) for each Gaussian component, shape (n_samples, n_components).

    eps : float, optional
        A small constant for numerical stability to prevent log(0). Default is 1e-10.

    Returns
    -------
    torch.Tensor
        The average negative log-likelihood loss across all valid samples.

    References
    ----------
    .. [#] Ha, D. (2015). Mixture density networks with TensorFlow.
           Retrieved from http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow

    Example
    -------
    >>> y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float('nan'), 6.0]])
    >>> prediction = {
    ...     'mu': torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]], [[1.5, 2.5], [3.5, 4.5]]]),
    ...     'sigma': torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
    ...     'pi': torch.tensor([[[0.3, 0.7], [0.6, 0.4]], [[0.4, 0.6], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    ... }
    >>> loss = MaskedGMMLoss(y, prediction)
    >>> print(loss)
    """

    ONE_OVER_2PI_SQRT = 1.0 / np.sqrt(2.0 * np.pi)

    m = prediction['mu']
    s = prediction['sigma']
    p = prediction['pi']

    mask = ~torch.isnan(y)
    for i in range(y.shape[0]):
        seg_mask = mask[[i]].any(0).any(-1)
        if torch.sum(seg_mask) > 0:

            seg_y = y[[i]][:, seg_mask, :]
            seg_m = m[[i]][:, seg_mask, :]
            seg_s = s[[i]][:, seg_mask, :]
            seg_p = p[[i]][:, seg_mask, :]

            # likelihood calculation
            seg_error = seg_y - seg_m
            seg_result = seg_error * torch.reciprocal(seg_s)
            seg_result = -0.5 * (seg_result * seg_result)
            seg_result = seg_p * ((torch.exp(seg_result) * torch.reciprocal(seg_s)) * ONE_OVER_2PI_SQRT)

            # concatenate all the likelihoods
            try:
                total_result = torch.cat([total_result, seg_result], dim = 1)
            except NameError as e:
                total_result = seg_result

    # Sum across n distribution
    result = torch.sum(total_result, dim=-1)
    # Take the negative log
    result = -torch.log(result + eps)
    # Take the average
    result = torch.sum(result) / torch.sum(mask)
    return(result)

# Numerically stable
def MaskedGMMLoss_weighted(y, prediction, weights, eps=1e-10):
    """
    Numerically stable, weighted negative log-likelihood loss for a Gaussian Mixture Model (GMM)
    with masking for missing values.

    Parameters
    ----------
    y : torch.Tensor
        Target tensor of shape (B, T, 1)

    prediction : dict
        Dictionary containing:
            - 'mu': (B, T, K)
            - 'sigma': (B, T, K)
            - 'pi': (B, T, K)

    weights : torch.Tensor
        Tensor of weights with shape (B, T, 1) or (B, T, K)

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    mu = prediction['mu']       # (B, T, K)
    sigma = prediction['sigma'] # (B, T, K)
    pi = prediction['pi']       # (B, T, K)

    B, T, K = mu.shape

    # Clamp sigma to avoid divide-by-zero or log(0)
    sigma = torch.clamp(sigma, min=1e-3, max=1e2)

    if y.ndim == 2:
        y = y.unsqueeze(-1)  # Ensure y is (B, T, 1)

    # Create mask for non-NaN entries
    mask = ~torch.isnan(y)     # (B, T, 1)

    total_log_probs = []
    total_weights = []

    for i in range(B):
        seg_mask = mask[i].squeeze(-1)  # (T,)

        if seg_mask.any():
            seg_y = y[i, seg_mask, :]          # (T_valid, 1)
            seg_mu = mu[i, seg_mask, :]        # (T_valid, K)
            seg_sigma = sigma[i, seg_mask, :]  # (T_valid, K)
            seg_pi = pi[i, seg_mask, :]        # (T_valid, K)
            seg_weights = weights[i, seg_mask, :]  # (T_valid, 1) or (T_valid, K)

            seg_y_exp = seg_y.expand_as(seg_mu)  # (T_valid, K)

            # log(pi) + log N(y | mu, sigma)
            log_pi = torch.log(seg_pi + eps)  # (T_valid, K)
            log_normal = -0.5 * ((seg_y_exp - seg_mu) / seg_sigma) ** 2 \
                         - torch.log(seg_sigma + eps) \
                         - 0.5 * np.log(2 * np.pi)

            log_prob = log_pi + log_normal     # (T_valid, K)
            log_sum_exp = torch.logsumexp(log_prob, dim=-1)  # (T_valid,)

            total_log_probs.append(-log_sum_exp)             # (T_valid,)
            total_weights.append(seg_weights.squeeze(-1))    # (T_valid,)

    # Concatenate and compute weighted average
    if total_log_probs:
        log_probs_all = torch.cat(total_log_probs, dim=0)   # (N,)
        weights_all = torch.cat(total_weights, dim=0)       # (N,)
        weighted_loss = torch.sum(weights_all * log_probs_all) / (torch.sum(weights_all) + eps)
    else:
        weighted_loss = torch.tensor(0.0, device=y.device)

    return weighted_loss

def MaskedGMMLoss_weighted_org(y, prediction, weights, eps = 1e-10):
    """
    Calculate the average negative log-likelihood for a Gaussian Mixture Model (GMM) with weighted contributions.

    This loss function computes the negative log-likelihood for GMMs, which is commonly used
    in mixture density networks. The implementation handles missing values in the target
    variable `y` by applying a mask, ensuring that only valid (non-NaN) values contribute
    to the loss calculation. Additionally, it incorporates weights for each sample, allowing
    for flexible loss contributions based on the importance of each observation.

    Parameters
    ----------
    y : torch.Tensor
        The target values (observations) for which the GMM predicts probabilities.
        It should be a tensor of shape (n_samples, n_features).

    prediction : dict
        A dictionary containing the parameters of the GMM:
        - 'mu' (torch.Tensor): The means of the Gaussian components, shape (n_samples, n_components, n_features).
        - 'sigma' (torch.Tensor): The standard deviations of the Gaussian components, shape (n_samples, n_components, n_features).
        - 'pi' (torch.Tensor): The mixture weights (prior probabilities) for each Gaussian component, shape (n_samples, n_components).

    weights : torch.Tensor
        A tensor of shape (n_samples, n_components, n_features) representing the weights for each sample.
        These weights determine the contribution of each sample to the overall loss.

    eps : float, optional
        A small constant for numerical stability to prevent log(0). Default is 1e-10.

    Returns
    -------
    torch.Tensor
        The average negative log-likelihood loss across all valid samples,
        weighted by the provided weights.

    References
    ----------
    .. [#] Ha, D. (2015). Mixture density networks with TensorFlow.
           Retrieved from http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow

    Example
    -------
    >>> y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float('nan'), 6.0]])
    >>> prediction = {
    ...     'mu': torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]], [[1.5, 2.5], [3.5, 4.5]]]),
    ...     'sigma': torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
    ...     'pi': torch.tensor([[[0.3, 0.7], [0.6, 0.4]], [[0.4, 0.6], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    ... }
    >>> weights = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]])  # Example weights
    >>> loss = MaskedGMMLoss_weighted(y, prediction, weights)
    >>> print(loss)
    """

    ONE_OVER_2PI_SQRT = 1.0 / np.sqrt(2.0 * np.pi)

    m = prediction['mu']
    s = prediction['sigma']
    p = prediction['pi']

    mask = ~torch.isnan(y)
    for i in range(y.shape[0]):
        seg_mask = mask[[i]].any(0).any(-1)
        if torch.sum(seg_mask) > 0:

            seg_y = y[[i]][:, seg_mask, :]
            seg_m = m[[i]][:, seg_mask, :]
            seg_s = s[[i]][:, seg_mask, :]
            seg_p = p[[i]][:, seg_mask, :]

            # likelihood calculation
            seg_error = seg_y - seg_m
            seg_result = seg_error * torch.reciprocal(seg_s)
            seg_result = -0.5 * (seg_result * seg_result)
            seg_result = seg_p * ((torch.exp(seg_result) * torch.reciprocal(seg_s)) * ONE_OVER_2PI_SQRT)

            # concatenate all the likelihoods
            try:
                total_result = torch.cat([total_result, seg_result], dim = 1)
                all_weights = torch.cat([all_weights, weights[[i]][:, seg_mask, :]], dim = 1)
            except NameError as e:
                total_result = seg_result
                all_weights = weights[[i]][:, seg_mask, :]

    # Sum the likelihoods across all Gaussian distributions for each sample
    result = torch.sum(total_result, dim=-1)
    # Compute the negative log-likelihood; add a small constant (eps) for numerical stability
    result = -torch.log(result + eps)
    # Apply the input weights to the result, scaling the negative log-likelihood accordingly
    result = all_weights[:,:,0] * result
    # Calculate the average loss by dividing the total weighted loss by the count of valid samples
    result = torch.sum(result) / torch.sum(mask)
    return(result)


def MaskedCMALLoss(y, prediction, eps = 1e-8):
    """
    Calculate the average negative log-likelihood for a model using the CMAL (Countable Mixtures of Asymmetric Laplacians) head.

    This loss function computes the average negative log-likelihood based on the CMAL framework,
    which is designed to model complex distributions. The implementation handles missing values
    in the target variable `y` by applying a mask, ensuring that only valid (non-NaN) values contribute
    to the loss calculation.

    Parameters
    ----------
    y : torch.Tensor
        The target values (observations) for which the CMAL model predicts probabilities.
        It should be a tensor of shape (n_samples, n_features).

    prediction : dict
        A dictionary containing the parameters of the CMAL model:
        - 'mu' (torch.Tensor): The predicted means, shape (n_samples, n_components, n_features).
        - 'b' (torch.Tensor): The predicted scale parameters, shape (n_samples, n_components, n_features).
        - 'tau' (torch.Tensor): The predicted transformation parameters, shape (n_samples, n_components, n_features).
        - 'pi' (torch.Tensor): The mixture weights (prior probabilities) for each component, shape (n_samples, n_components).

    eps : float, optional
        A small constant for numerical stability to prevent log(0). Default is 1e-8.

    Returns
    -------
    torch.Tensor
        The average negative log-likelihood loss across all valid samples.

    Example
    -------
    >>> y = torch.tensor([[1.0, 2.0], [3.0, float('nan'), 4.0]])
    >>> prediction = {
    ...     'mu': torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]]),
    ...     'b': torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
    ...     'tau': torch.tensor([[[0.1, 0.1], [0.1, 0.1]], [[0.2, 0.2], [0.2, 0.2]]]),
    ...     'pi': torch.tensor([[[0.3, 0.7], [0.6, 0.4]], [[0.4, 0.6], [0.5, 0.5]]])
    ... }
    >>> loss = MaskedCMALLoss(y, prediction)
    >>> print(loss)
    """

    m = prediction['mu']
    b = prediction['b']
    t = prediction['tau']
    p = prediction['pi']

    mask = ~torch.isnan(y)
    for i in range(y.shape[0]):
        seg_mask = mask[[i]].any(0).any(-1)
        if torch.sum(seg_mask) > 0:

            seg_y = y[[i]][:, seg_mask, :]
            seg_m = m[[i]][:, seg_mask, :]
            seg_b = b[[i]][:, seg_mask, :]
            seg_t = t[[i]][:, seg_mask, :]
            seg_p = p[[i]][:, seg_mask, :]

            # likelihood calculation
            seg_error = seg_y - seg_m
            seg_log_like = torch.log(seg_t) + \
               torch.log(1.0 - seg_t) - \
               torch.log(seg_b) - \
               torch.max(seg_t * seg_error, (seg_t - 1.0) * seg_error) / seg_b
            seg_log_weights = torch.log(seg_p + eps)

            # concatenate all the likelihoods
            try:
                total_log_like = torch.cat([total_log_like , seg_log_like], dim = 1)
                total_log_weights = torch.cat([total_log_weights, seg_log_weights], dim = 1)
            except NameError as e:
                total_log_like = seg_log_like
                total_log_weights = seg_log_weights

    # Aggregate
    result = torch.logsumexp(total_log_weights + total_log_like, dim=2)
    result = -torch.mean(torch.sum(result, dim=1))
    return(result)


def MaskedUMALLoss(y, taus, n_taus, prediction, eps = 1e-5):
    """
    Calculate the average negative log-likelihood for a model using the UMAL (Uncountable Mixtures of Asymmetric Laplacians) head.

    This loss function computes the average negative log-likelihood based on the UMAL framework,
    which is designed to model complex distributions effectively. The implementation handles missing
    values in the target variable `y` by applying a mask, ensuring that only valid (non-NaN) values
    contribute to the loss calculation.

    Parameters
    ----------
    y : torch.Tensor
        The target values (observations) for which the UMAL model predicts probabilities.
        It should be a tensor of shape (n_samples, n_features).

    taus : torch.Tensor
        A tensor of shape (n_samples, n_taus, n_features) representing the transformation parameters
        for the model.

    n_taus : int
        The number of transformation parameters (taus) used in the model.

    prediction : dict
        A dictionary containing the parameters of the UMAL model:
        - 'mu' (torch.Tensor): The predicted means, shape (n_samples, n_components, n_features).
        - 'b' (torch.Tensor): The predicted scale parameters, shape (n_samples, n_components, n_features).

    eps : float, optional
        A small constant for numerical stability to prevent log(0). Default is 1e-5.

    Returns
    -------
    torch.Tensor
        The average negative log-likelihood loss across all valid samples.

    Example
    -------
    >>> y = torch.tensor([[1.0, 2.0], [3.0, float('nan'), 4.0]])
    >>> taus = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.3], [0.4, 0.5]]])
    >>> n_taus = 2
    >>> prediction = {
    ...     'mu': torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]]),
    ...     'b': torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    ... }
    >>> loss = MaskedUMALLoss(y, taus, n_taus, prediction)
    >>> print(loss)
    """

    t = taus
    m = prediction['mu']
    b = prediction['b']

    mask = ~torch.isnan(y)
    for i in range(y.shape[0]):
        seg_mask = mask[[i]].any(0).any(-1)
        if torch.sum(seg_mask) > 0:

            seg_y = y[[i]][:, seg_mask, :]
            seg_m = m[[i]][:, seg_mask, :]
            seg_b = b[[i]][:, seg_mask, :]
            seg_t = t[[i]][:, seg_mask, :]

            # likelihood calculation
            seg_error = seg_y - seg_m
            seg_log_like = torch.log(seg_t) + \
               torch.log(1.0 - seg_t) - \
               torch.log(seg_b) - \
               torch.max(seg_t * seg_error, (seg_t - 1.0) * seg_error) / seg_b

            n_taus_log = torch.as_tensor(np.log(n_taus).astype('float32'))

            original_batch_size = int(seg_log_like.shape[0] / n_taus)
            seg_log_like_split = torch.cat(seg_log_like[:, :, :].split(original_batch_size, 0), 2)


            # concatenate all the likelihoods
            try:
                total_log_like = torch.cat([total_log_like , seg_log_like_split], dim = 1)
            except NameError as e:
                total_log_like = seg_log_like_split

    # Aggregate
    result = torch.logsumexp(total_log_like, dim=2) - n_taus_log
    result = -torch.mean(torch.sum(result, dim=1))
    return(result)

def get_UMAL_taus(data, n_taus, tau_min, tau_max, batch_size, extend_batch):
    """
    Generate random tau values for the UMAL (Uncountable Mixtures of Asymmetric Laplacians) model.

    This function creates a tensor of tau values sampled uniformly from a specified range
    defined by `tau_min` and `tau_max`. The generated tau values can either be created
    for a standard batch size or extended to cover multiple taus per batch.

    Parameters
    ----------
    data : torch.Tensor
        The input data tensor, shape (seq_length, n_features). The sequence length is used
        to determine how many times the tau values should be repeated.

    n_taus : int
        The number of tau values to generate for each batch.

    tau_min : float
        The minimum value for the tau sampling range.

    tau_max : float
        The maximum value for the tau sampling range.

    batch_size : int
        The number of samples in each batch.

    extend_batch : bool
        If True, generates tau values for each of the `n_taus` for the given batch size.
        If False, generates a single tau value per sample in the batch.

    Returns
    -------
    torch.Tensor
        A tensor of shape (seq_length, batch_size * n_taus, 1) if `extend_batch` is True,
        or (seq_length, batch_size, 1) if `extend_batch` is False, containing the generated tau values.

    Example
    -------
    >>> data = torch.randn(5, 3)  # Example data with seq_length=5 and n_features=3
    >>> n_taus = 2
    >>> tau_min = 0.1
    >>> tau_max = 1.0
    >>> batch_size = 4
    >>> extend_batch = True
    >>> taus = get_UMAL_taus(data, n_taus, tau_min, tau_max, batch_size, extend_batch)
    >>> print(taus.shape)  # Should print: torch.Size([5, 8, 1]) if extend_batch is True
    """
    seq_length = data.shape[0]
    if extend_batch:
        taus = ((tau_max - tau_min) * torch.rand(1, batch_size * n_taus, 1) + tau_min)
    else:
        taus = ((tau_max - tau_min) * torch.rand(1, batch_size, 1) + tau_min)
    taus = taus.repeat(seq_length, 1, 1)
    return(taus)

def fit_torch_model(model, x, y, h, c, weighting_matrix,
                    epochs, loss_fn, optimizer, gpu, head, early_stopping_patience, weights_file,
                    umal_extend_batch, umal_n_taus_train, umal_tau_min, umal_tau_max,
                    weight_loss, weight_threshold, weight_value, x_delta=None):
    """
    Train a PyTorch model using the specified parameters and data.

    This function fits a PyTorch model to the provided training data over a specified number
    of epochs. It supports the option to utilize GPU for training if available and requested.
    The function also accommodates different loss functions based on the specified head.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.

    x : torch.Tensor
        The input data tensor, shape (n_samples, n_features).

    y : torch.Tensor
        The target values tensor, shape (n_samples, n_targets).

    h : torch.Tensor
        The initial hidden state tensor for the LSTM, shape (n_layers, batch_size, hidden_size).

    c : torch.Tensor
        The initial cell state tensor for the LSTM, shape (n_layers, batch_size, hidden_size).

    weighting_matrix : torch.Tensor
        A matrix used for weighting the loss function, shape (n_samples, n_targets).

    epochs : int
        The number of epochs to train the model.

    loss_fn : callable
        The loss function to be used for training. It should accept the target and predicted values.

    optimizer : torch.optim.Optimizer
        The optimizer used for updating model weights.

    gpu : bool
        If True, the model and data will be moved to GPU for training if available.

    head : str
        The type of model head to use.

    umal_extend_batch : bool
        If True, generates multiple tau values for each sample in the batch.

    umal_n_taus_train : int
        The number of tau values to generate during training for the UMAL head.

    umal_tau_min : float
        The minimum value for tau sampling.

    umal_tau_max : float
        The maximum value for tau sampling.

    weight_loss : bool
        If True, applies weighting to the loss based on the specified conditions.

    weight_threshold : float
        The temperature threshold above which observations will be assigned a higher weight in the loss function.

    weight_value : float
        The factor by which to weight observations that exceed the threshold. For example, a value of 2 means these observations will contribute twice as much to the loss as those below the threshold.

    Returns
    -------
    torch.nn.Module
        The trained PyTorch model.

    Example
    -------
    >>> model = MyModel()  # Replace with your model class
    >>> x = torch.randn(100, 10)  # Example input data
    >>> y = torch.randn(100, 1)    # Example target data
    >>> h = torch.zeros((1, 100, 64))  # Initial hidden state
    >>> c = torch.zeros((1, 100, 64))  # Initial cell state
    >>> weighting_matrix = torch.ones(100, 1)  # Weighting matrix
    >>> epochs = 10
    >>> loss_fn = torch.nn.MSELoss()  # Example loss function
    >>> optimizer = torch.optim.Adam(model.parameters())  # Example optimizer
    >>> gpu = True  # Use GPU if available
    >>> head = 'UMAL'
    >>> umal_extend_batch = True
    >>> umal_n_taus_train = 5
    >>> umal_tau_min = 0.1
    >>> umal_tau_max = 1.0
    >>> trained_model = fit_torch_model(model, x, y, h, c, weighting_matrix,
    ...                                   epochs, loss_fn, optimizer, gpu, head,
    ...                                   umal_extend_batch, umal_n_taus_train,
    ...                                   umal_tau_min, umal_tau_max,
    ...                                   weight_loss, weight_threshold, weight_value)
    """
    # moving to gpu if available and requested
    # specifying request because gpu can be slower for small models
    if gpu == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    h = h.to(device)
    c = c.to(device)
    weighting_matrix = weighting_matrix.to(device)
    if x_delta is not None:
        x_delta = x_delta.to(device)

    if not early_stopping_patience:
        early_stopping_patience = epochs

    epochs_since_best = 0
    best_loss = 1000 # Will get overwritten

    # Initialize weights for loss calculation
    weights = torch.ones(y.shape)
    if weight_loss:
        # Assign higher weights to observations exceeding the threshold
        weights[y > weight_threshold] = weight_value

    for i in range(epochs):
        start_time = time.time()

        out, (h, c) = model(x, (h.detach(), c.detach()), weighting_matrix, x_delta) # stateful lstm
            # .detach() because prev h/c are tied to gradients/weights of
            # a different iteration

        if head == 'UMAL':
            taus = get_UMAL_taus(y, umal_n_taus_train, umal_tau_min, umal_tau_min, x.shape[0], umal_extend_batch)
            loss = loss_fn(y, taus, umal_n_taus_train, out)
        else:
            if weight_loss:
                loss = loss_fn(y, out, weights)
            else:
                loss = loss_fn(y, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        loop_time = end_time - start_time

        print('Epoch %i/' %(i+1) + str(epochs), flush = True)
        print('[==============================]',
              '{0:.2f}'.format(loop_time) + 's/step',
              '- loss: ' + '{0:.4f}'.format(loss.item()),
              flush = True)

        if loss < best_loss:
            torch.save(model.state_dict(), weights_file)
            best_loss = loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epochs_since_best > early_stopping_patience:
            print(f"Early Stopping at Epoch {i}")
            break

    # move back to cpu when complete for simplicity
    model.to('cpu')
    return(model)

def unscale_output(y_scl, y_std, y_mean):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [numpy array] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables_to_log [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :return: unscaled data
    """
    y_unscaled = y_scl.copy()

    y_unscaled = (y_scl * (y_std + 1e-10)) + y_mean

    return y_unscaled


## Generic PyTorch Training Routine
def train_loop(epoch_index,
               dataloader,
               h,
               c,
               weighting_matrix,
               head,
               model,
               loss_function,
               optimizer,
               umal_extend_batch,
               umal_n_taus_train,
               umal_tau_min,
               umal_tau_max,
               weight_loss, 
               weight_threshold, 
               weight_value,
               device = 'cpu', 
               disable_tqdm = False):
    """
    @param epoch_index: [int] Epoch number
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] Chosen optimizer
    @param device: [str] cpu or gpu
    @return: [float] epoch loss
    """
    train_loss=[]
    with tqdm(dataloader, ncols=100, desc= f"Epoch {epoch_index+1}", unit="batch", disable=disable_tqdm) as tepoch:
        for x, y in tepoch:
            trainx = x.to(device)
            trainy = y.to(device)

            # Initialize weights for loss calculation
            weights = torch.ones(trainy.shape)
            if weight_loss:
                # Assign higher weights to observations exceeding the threshold
                weights[trainy > weight_threshold] = weight_value

            optimizer.zero_grad()
            output, (h, c) = model(trainx, (h.detach(), c.detach()), weighting_matrix)
            if head == 'UMAL':
                taus = get_UMAL_taus(trainy, umal_n_taus_train, umal_tau_min, umal_tau_min, trainx.shape[0], umal_extend_batch)
                loss = loss_function(trainy, taus, umal_n_taus_train, output)
            else:
                if weight_loss:
                    loss = loss_function(trainy, output, weights)
                else: 
                    loss = loss_function(trainy, output) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            train_loss.append(loss.item())
            tepoch.set_postfix(loss=loss.item())
    mean_loss = np.mean(train_loss)
    return mean_loss

def val_loop(dataloader,
             h,
             c,
             weighting_matrix,
             head,
             model,
             loss_function,
             umal_extend_batch,
             umal_n_taus_train,
             umal_tau_min,
             umal_tau_max,
             weight_loss, 
             weight_threshold, 
             weight_value,
             device = 'cpu'):
    """
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param device: [str] cpu or gpu
    @return: [float] epoch validation loss
    """
    val_loss = []
    for iter, (x, y) in enumerate(dataloader):
        testx = x.to(device)
        testy = y.to(device)
        # Initialize weights for loss calculation
        weights = torch.ones(testy.shape)
        if weight_loss:
            # Assign higher weights to observations exceeding the threshold
            weights[testy > weight_threshold] = weight_value

        output, (h, c) = model(testx, (h.detach(), c.detach()), weighting_matrix)
        if head == 'UMAL':
            taus = get_UMAL_taus(testy, umal_n_taus_train, umal_tau_min, umal_tau_min, testx.shape[0], umal_extend_batch)
            loss = loss_function(testy, taus, umal_n_taus_train, output)
        else:
            if weight_loss:
                loss = loss_function(testy, output, weights)
            else: 
                loss = loss_function(testy, output) 
        val_loss.append(loss.item())
    mval_loss = np.mean(val_loss)
    print(f"Valid loss: {mval_loss:.2f}")
    return mval_loss

def train_torch(model,
                loss_function,
                optimizer,
                x_train,
                x_delta_train,
                y_train,
                h_train,
                c_train,
                h_val,
                c_val,
                weighting_matrix_train,
                weighting_matrix_val,
                batch_size,
                max_epochs,
                head,
                umal_extend_batch,
                umal_n_taus_train,
                umal_tau_min,
                umal_tau_max,
                weight_loss, 
                weight_threshold, 
                weight_value,
                early_stopping_patience=False,
                x_val = None,
                x_delta_val = None,
                y_val = None,
                shuffle = False,
                weights_file = None,
                log_file= None,
                device = 'cpu',
                keep_portion = None,
                disable_tqdm = False):
    """
    modified from river-dl
    @param model: [objetct] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] chosen optimizer
    @param x_train:
    @param batch_size: [int]
    @param max_epochs: [maximum number of epochs to run for]
    @param early_stopping_patience: [int] number of epochs without improvement in validation loss to run before stopping training
    @param shuffle: [bool] Shuffle training batches
    @param weights_file: [str] path save trained model weights
    @param log_file: [str] path to save training log to
    @return: [object] trained model
    """

    print(f"Training on {device}")
    print("start training...",flush=True)

    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    epochs_since_best = 0
    best_loss = 1000 # Will get overwritten

    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            period = int(keep_portion * y_train.shape[1])
        y_train[:, :-period, ...] = np.nan
        if y_val is not None:
            y_val[:, :-period, ...] = np.nan

    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        h_train = h_train.to(device)
        c_train = c_train.to(device)
        h_val = h_val.to(device)
        c_val = c_val.to(device)
        weighting_matrix_train = weighting_matrix_train.to(device)
        weighting_matrix_val = weighting_matrix_val.to(device)

    if x_delta_train is not None:
        x_delta_train = x_delta_train.to(device)
    if x_delta_val is not None: 
        x_delta_val = x_delta_val.to(device)


    # Put together dataloaders
    train_data = []
    for i in range(len(x_train)):
        train_data.append([x_train[i], y_train[i]])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    if x_val is not None:
        val_data = []
        for i in range(len(x_val)):
            val_data.append([x_val[i], y_val[i]])

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    # TODO: add in delta support for data loaders 

    val_time = []
    train_time = []

    ### Run training loop
    log_cols = ['epoch', 'loss', 'val_loss','time','val_time']
    train_log = pd.DataFrame(columns=log_cols)

    for i in range(max_epochs):

        t1 = time.time()

        model.train()
        epoch_loss = train_loop(i, train_loader, h_train, c_train, weighting_matrix_train,
                                head, model, loss_function, optimizer, umal_extend_batch,
                                umal_n_taus_train, umal_tau_min, umal_tau_max, 
                                weight_loss, weight_threshold, weight_value, device, disable_tqdm)
        train_time.append(time.time() - t1)
        train_log = pd.concat([train_log,pd.DataFrame([[i, epoch_loss, np.nan,time.time()-t1,np.nan]],columns=log_cols,index=[i])])

        #Val
        if x_val is not None:
            s1 = time.time()
            model.eval()
            epoch_val_loss = val_loop(val_loader, h_val, c_val, weighting_matrix_val,
                                      head, model, loss_function, umal_extend_batch,
                                      umal_n_taus_train, umal_tau_min, umal_tau_max,
                                      weight_loss, weight_threshold, weight_value, device)

            if epoch_val_loss < best_loss:
                torch.save(model.state_dict(), weights_file)
                best_loss = epoch_val_loss
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if epochs_since_best > early_stopping_patience:
                print(f"Early Stopping at Epoch {i}")
                break
            train_log.loc[train_log.epoch==i,"val_loss"]=epoch_val_loss
            train_log.loc[train_log.epoch==i,"val_time"]=time.time() - s1
            val_time.append(time.time()-s1)

    train_log.to_csv(log_file)
    #print(train_log)
    if x_val is None:
        torch.save(model.state_dict(), weights_file)
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    else:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Validation (Inference) Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    # move back to cpu when complete for simplicity
    model.to('cpu')
    return model
