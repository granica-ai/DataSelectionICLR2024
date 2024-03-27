import numpy as np
from utils.misc_utils import alpha_exp_wrapper, clip_weights, get_c_opt


class Subsampler:
    """
    Weakly supervised subsampling using the surrogate model.
    """

    def __init__(self, alpha_exp, theta_surr, scale_surr=1):
        """
        Args:
            alpha_exp: float, exponent for the subselector function (\alpha in Eq 6.1).
            theta_surr: np.ndarray, surrogate model weights, shape (1, p).
            scale_surr: float, logits scaling factor for the unnormalized selection probabilities.
        """
        self.alpha_exp = alpha_exp

        # assert that either alpha is a float or int or 'Easy' or 'Hard'
        assert isinstance(alpha_exp, float) or isinstance(alpha_exp, int) or alpha_exp in ['Easy', 'Hard'], \
            "alpha_exp should be a int/float or 'Easy'/'Hard' (for topK)"

        self.scale_surr = scale_surr
        self.pi = alpha_exp_wrapper(alpha_exp)
        self.theta_surr = theta_surr

    def get_unnormalized_pi(self, X):
        """
        Get unnormalized selection probs from surrogate model.
        Unnormalized scores are [p(z) * (1 - p(z))] ** \alpha where z = scale_surr * (X @ theta_surr)
        See equation 6.1 in the paper

        Args:
            X: np.ndarray, surrogate model features, shape (N, p).
        """
        unnormalized_pi = self.pi(clip_weights(self.scale_surr * X['train'] @ self.theta_surr.T)).squeeze()
        self.N_train = len(unnormalized_pi)
        self._unnormalized_pi = unnormalized_pi

    def get_pi(self, split_pct):
        """
        Get selection probabilities for the training set given a split percentage. Takes care of normalization.
        """
        assert 0 < split_pct < 100, "split_pct should be in the range (0, 100)"

        N_selected = int(self.N_train * split_pct / 100)

        # we need to normalize the selection probabilities to ensure that the expected training size is equal to
        # split_pct% of the total training size [c(\gamma) in Eq 6.1]
        c_opt = get_c_opt(self._unnormalized_pi, N_selected)

        # normalize the unnormalized selection probabilities
        pi = np.minimum(c_opt * self._unnormalized_pi, 1)
        self.pi = pi

        return pi

    def sample_from_pi(self, sample_seed):
        """
        Sample training indices from the normalized selection probabilities.
        """
        np.random.seed(sample_seed)
        binary = np.random.binomial(1, self.pi, size=self.N_train)
        train_idx = np.argwhere(binary).squeeze()

        return train_idx

    def sampleTopk_from_unnormalized_pi(self, split_pct, easyHard):
        """
        Sample top k indices from the unnormalized selection probabilities instead of sampling with probs.
        """
        assert easyHard in ['Easy', 'Hard'], "easyHard should be either 'Easy' or 'Hard'"
        N_selected = int(self.N_train * split_pct / 100)
        if easyHard == 'Hard':
            train_idx = np.argpartition(self._unnormalized_pi, -N_selected)[-N_selected:]
        else:
            train_idx = np.argpartition(self._unnormalized_pi, N_selected)[:N_selected]

        return train_idx
