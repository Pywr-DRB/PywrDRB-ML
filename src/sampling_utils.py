import torch
import numpy as np
#import arviz # need if using multiple distribution heads
import sys

#sys.path.insert(1, '3_model_train/src')
from .training_utils import get_UMAL_taus


def sample_GMM(out, n_samples):
    m = out['mu']
    s = out['sigma']
    p = out['pi']
    
    batch_size = m.shape[0]
    seq_length = m.shape[1]
    all_samples = torch.zeros(batch_size,
                              seq_length,
                              n_samples)
    
    m_target = torch.repeat_interleave(m, n_samples, dim=0)
    s_target = torch.repeat_interleave(s, n_samples, dim=0)
    p_target = torch.repeat_interleave(p, n_samples, dim=0)
    
    iter_range = range(m.shape[1])
    for i in iter_range:
        distribution_choice = torch.multinomial(p_target[:, i, :], num_samples=1)

        m_sample = m_target[:, i, :].gather(1, distribution_choice)
        s_sample = s_target[:, i, :].gather(1, distribution_choice)

        sample = torch.normal(m_sample, s_sample)
        all_samples[:, i, :] = sample.reshape(batch_size, n_samples)
        
    all_samples = all_samples.detach().numpy()
    return(all_samples)


def _sample_asymmetric_laplacians(m_sub: torch.Tensor, b_sub: torch.Tensor,
                                  t_sub: torch.Tensor) -> torch.Tensor:
    # Value used to avoid the case of log(0) = -inf
    num_stab = 1e-10
    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    prob = torch.FloatTensor(m_sub.shape) \
        .uniform_(0, 1) \
        .to(m_sub.device)  # sample uniformly between zero and 1
    # Avoid the case of log(0) = -inf
    prob = torch.where(prob < 0.5, prob + num_stab, prob - num_stab)
    values = torch.where(
        prob < t_sub,  # needs to be in accordance with the loss
        m_sub + ((b_sub * torch.log(prob / t_sub)) / (1 - t_sub)),
        m_sub - ((b_sub * torch.log((1 - prob) / (1 - t_sub))) / t_sub))
    return values


def sample_CMAL(out, n_samples):    
    m = out['mu']
    b = out['b']
    t = out['tau']
    p = out['pi']
    
    batch_size = m.shape[0]
    seq_length = m.shape[1]
    all_samples = torch.zeros(batch_size,
                              seq_length,
                              n_samples)
    
    m_target = torch.repeat_interleave(m, n_samples, dim=0)
    b_target = torch.repeat_interleave(b, n_samples, dim=0)
    t_target = torch.repeat_interleave(t, n_samples, dim=0)
    p_target = torch.repeat_interleave(p, n_samples, dim=0)

    samples_ls = []
    iter_range = range(m.shape[1])
    for i in iter_range:
        distribution_choice = torch.multinomial(p_target[:, i, :], num_samples=1)

        m_choice = m_target[:, i, :].gather(1, distribution_choice)
        t_choice = t_target[:, i, :].gather(1, distribution_choice)
        b_choice = b_target[:, i, :].gather(1, distribution_choice)

        samples = _sample_asymmetric_laplacians(m_choice, b_choice, t_choice)
        all_samples[:, i, :] = samples.reshape(batch_size, n_samples)
      
    all_samples = all_samples.detach().numpy()
    return(all_samples)


def sample_UMAL(out, n_samples, n_distr, batch_size):
    m = out['mu']
    b = out['b']
    
    seq_length = m.shape[1]
    all_samples = torch.zeros(batch_size,
                              seq_length,
                              n_samples)
    
    t = get_UMAL_taus(m, n_distr, 0.25, 0.75, batch_size, True)
    
    m = torch.cat(m.split(batch_size, 0), 2)
    b = torch.cat(b.split(batch_size, 0), 2)
    t = torch.cat(t.split(batch_size, 0), 2)

    m_target = torch.repeat_interleave(m, n_samples, dim=0)
    b_target = torch.repeat_interleave(b, n_samples, dim=0)
    t_target = torch.repeat_interleave(t, n_samples, dim=0)

    samples_ls = []
    iter_range = range(m.shape[1])
    for i in iter_range:
        choice = np.random.randint(0, n_distr) # weighted equally

        m_choice = m_target[:, i, choice]
        b_choice = b_target[:, i, choice]
        t_choice = t_target[:, i, choice]

        samples = _sample_asymmetric_laplacians(m_choice, b_choice, t_choice)
        all_samples[:, i, :] = samples.reshape(batch_size, n_samples)
        
    all_samples = all_samples.detach().numpy()
    return(all_samples)
  

def get_point_estimates(samples, method):
    # Assumes samples are of shape [batch, seq_len, n_samples]
    # Returns an estimate of shape [batch, seq_len, 1]
    point_estimates = np.zeros([samples.shape[0], samples.shape[1], 1])
    for seq_i in range(samples.shape[1]):
        for batch_i in range(samples.shape[0]):
            #if method == 'mode':
            #    grid, pdf = arviz.kde(samples[batch_i, seq_i])
            #    point_estimates[batch_i, seq_i] = grid[np.argmax(pdf)]
            if method == 'mean':
                point_estimates[batch_i, seq_i] = np.mean(samples[batch_i, seq_i])
    return(point_estimates)
