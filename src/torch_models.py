import torch
import torch.nn as nn
from typing import Dict

# Simple LSTM made from scratch
#   Credit to / code modified from - https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
#   Associated github repo - https://github.com/piEsposito/pytorch-lstm-by-hand
class LSTM_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout = 0, dropout = 0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        # See the file "neuralnet_math_README.md" in the root directory for
        # equations and implementation details
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states = None, adj_matrix = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim= 1)
        return hidden_seq, (h_t, c_t)


class RGrN_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout = 0, dropout = 0):
        super().__init__()

        # New stuff
        self.A = adj_matrix # torch.from_numpy(adj_matrix).float() # provided at initialization
        # parameters for mapping graph/spatial data
        self.weight_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_q = nn.Parameter(torch.Tensor(hidden_dim))

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states = None, adj_matrix = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            q_t = torch.tanh(h_t @ self.weight_q + self.bias_q)
            if adj_matrix == None:
                c_t = f_t * (c_t + self.A @ q_t) + i_t * self.recur_dropout(g_t)
            # Option to use a different adjacency matrix on forward pass
            else:
                c_t = f_t * (c_t + adj_matrix @ q_t) + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim= 1)
        return hidden_seq, (h_t, c_t)


# Credit to / code modified from https://github.com/neuralhydrology/neuralhydrology
class GMM(nn.Module):
    """Gaussian Mixture Density Network
    A mixture density network with Gaussian distribution as components. Good references are [#]_ and [#]_. The latter
    one forms the basis for our implementation. As such, we also use two layers in the head to provide it with
    additional flexibility, and exponential activation for the variance estimates and a softmax for weights.
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 3 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net,
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(GMM, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a GMM head forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the GMM components.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing mixture parameters and weights; where the key 'mu' stores the means, the key
            'sigma' the variances, and the key 'pi' the weights.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        # split output into mu, sigma and weights
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}


# Credit to / code modified from https://github.com/neuralhydrology/neuralhydrology
class UMAL(nn.Module):
    """Uncountable Mixture of Asymmetric Laplacians.
    An implicit approximation to the mixture density network with Laplace distributions which does not require to
    pre-specify the number of components. An additional hidden layer is used to provide the head more expressiveness.
    General details about UMAL can be found in [#]_. A major difference between Brando's implementation
    and NH's/ours is the binding-function for the scale-parameter (b). The scale needs to be lower-bound. The original UMAL
    implementation uses an elu-based binding. In our experiment however, this produced under-confident predictions
    (too large variances). We therefore opted for a tailor-made binding-function that limits the scale from below and
    above using a sigmoid. It is very likely that this needs to be adapted for non-normalized outputs.
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 2 times the output-size, since the scale parameters are also predicted.
    n_hidden : int
        Size of the hidden layer.
    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems,
        pp. 8838-8848, 2019.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(UMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._upper_bound_scale = 0.5  # this parameter found empirical by testing UMAL for a limited set of basins
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a UMAL head forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the UMAL components.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the means ('mu') and scale parameters ('b') to parametrize the asymmetric Laplacians.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = h.chunk(2, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps  # bind scale from two sides.
        return {'mu': m, 'b': b}


# Credit to / code modified from https://github.com/neuralhydrology/neuralhydrology
class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.
    An mixture density network with Laplace distributions as components.
    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as the GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the CMAL components.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the mixture component parameters and weights; where the key 'mu'stores the means,
            the key 'b' the scale parameters, the key 'tau' the skewness parameters, and the key 'pi' the weights).
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._softplus(b_latent) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}


# Credit to / code modified from https://github.com/neuralhydrology/neuralhydrology
class Regression(nn.Module):
    """Single-layer regression head with different output activations.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    """

    def __init__(self, n_in: int, n_out: int):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.

        Parameters
        ----------
        x : torch.Tensor
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.fc(x)}


class LSTMWithHead(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, adj_matrix, dropout, recur_dropout,
                 head, head_hidden_dim, head_n_dist, delta_temp_input_dim=None):
        super().__init__()
        self.lstm_layer = LSTM_v1(input_dim = input_dim,
                                 hidden_dim = lstm_hidden_dim,
                                 adj_matrix = adj_matrix,
                                 recur_dropout = recur_dropout,
                                 dropout = dropout)

        # Additional LSTM for predicting ΔTemp
        if delta_temp_input_dim is not None:
            self.delta_temp_lstm = LSTM_v1(input_dim=delta_temp_input_dim,
                                            hidden_dim=lstm_hidden_dim,
                                            adj_matrix=adj_matrix,
                                            recur_dropout=recur_dropout,
                                            dropout=dropout)
            self.delta_head_layer = Regression(n_in = lstm_hidden_dim,
                                               n_out = 1)

        assert(head in ['GMM', 'CMAL', 'UMAL', 'Regression'])
        if head == 'GMM':
            self.head_layer = GMM(n_in = lstm_hidden_dim,
                                  n_hidden = head_hidden_dim,
                                  n_out = 3*head_n_dist)
        if head == 'CMAL':
            self.head_layer = CMAL(n_in = lstm_hidden_dim,
                                   n_hidden = head_hidden_dim,
                                   n_out = 4*head_n_dist)
        if head == 'UMAL':
            self.head_layer = UMAL(n_in = lstm_hidden_dim,
                                   n_hidden = head_hidden_dim,
                                   n_out = 2*head_n_dist)
        if head == 'Regression':
            self.head_layer = Regression(n_in = lstm_hidden_dim,
                                         n_out = 1)

    def forward(self, x, init_states = None, adj_matrix = None, x_delta=None):
        lstm_out, (h, c) = self.lstm_layer(x, init_states, adj_matrix)

        # Predict ΔTemp
        if x_delta is not None:
            delta_temp_out, _ = self.delta_temp_lstm(x_delta, init_states, adj_matrix)
            delta_temp_pred = self.delta_head_layer(delta_temp_out)
            delta_temp_pred = torch.relu(delta_temp_pred['y_hat'])  # Ensure positive output

        out = self.head_layer(lstm_out)

        # Calculate the updated mean temp
        if x_delta is not None:
            out['mu'] = out['mu'] - delta_temp_pred

        return out, (h, c)



class RGrNWithHead(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, adj_matrix, dropout, recur_dropout,
                 head, head_hidden_dim, head_n_dist):
        super().__init__()
        self.rgcn_layer = RGrN_v1(input_dim = input_dim,
                                  hidden_dim = lstm_hidden_dim,
                                  adj_matrix = adj_matrix,
                                  recur_dropout = recur_dropout,
                                  dropout = dropout)
        assert(head in ['GMM', 'CMAL', 'UMAL', 'Regression'])
        if head == 'GMM':
            self.head_layer = GMM(n_in = lstm_hidden_dim,
                                  n_hidden = head_hidden_dim,
                                  n_out = 3*head_n_dist)
        if head == 'CMAL':
            self.head_layer = CMAL(n_in = lstm_hidden_dim,
                                   n_hidden = head_hidden_dim,
                                   n_out = 4*head_n_dist)
        if head == 'UMAL':
            self.head_layer = UMAL(n_in = lstm_hidden_dim,
                                   n_hidden = head_hidden_dim,
                                   n_out = 2*head_n_dist)
        if head == 'Regression':
            self.head_layer = Regression(n_in = lstm_hidden_dim,
                                         n_out = 1)

    def forward(self, x, init_states = None, adj_matrix = None):
        lstm_out, (h, c) = self.rgcn_layer(x, init_states, adj_matrix)
        out = self.head_layer(lstm_out)
        return out, (h, c)

