import torch
from torch import nn


class VirtualBN1D(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(VirtualBN1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        gamma = torch.normal(mean=torch.ones(1, num_features, 1), std=0.02)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma.float().to(device)   #.cuda(non_blocking=True) check the meaning
        self.beta = torch.FloatTensor(1, num_features, 1).fill_(0).to(device)

        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def prepare(self, ref_x):
        mean, mean_sq = self.get_stats(ref_x)
        # reference mode - works just like batch norm
        mean = mean.clone().detach()
        mean_sq = mean_sq.clone().detach()
        out = self._normalize(ref_x, mean, mean_sq)
        return out, mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        mean, mean_sq = self.get_stats(x)
        batch_size = x.size(0)
        new_coeff = 1. / (batch_size + 1.)
        old_coeff = 1. - new_coeff
        mean = new_coeff * mean + old_coeff * ref_mean
        mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
        out = self._normalize(x, mean, mean_sq)
        return out

    def _normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                    'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                    'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x