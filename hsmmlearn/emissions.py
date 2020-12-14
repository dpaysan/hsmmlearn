from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from .utils import NonParametricDistribution


class AbstractEmissions(object):
    """ Base class for emissions distributions.

    To create a HSMM with a custom emission distribution, write a derived
    class that implements some (or all) of the abstract methods. If you
    don't need all of the HSMM functionality, you can get by with implementing
    only some of the methods.

    """

    __meta__ = ABCMeta

    @abstractmethod
    def sample_for_state(self, state, size=None):
        """ Return a random emission given a state.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.sample`.

        Parameters
        ----------
        state : int
            The internal state.
        size : int
            The number of random samples to generate.

        Returns
        -------
        observations : numpy.ndarray, shape=(size, )
            Random emissions.

        """
        raise NotImplementedError

    @abstractmethod
    def likelihood(self, obs):
        """ Compute the likelihood of a sequence of observations.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` and
        :py:func:`hsmmlearn.hsmm.HSMMModel.decode`.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(n_obs, )
            Sequence of observations.

        Returns
        -------
        likelihood : float

        """
        raise NotImplementedError

    @abstractmethod
    def reestimate(self, gamma, obs):
        r""" Estimate the distribution parameters given sequences of
        smoothed probabilities and observations.

        The parameter ``gamma`` is an array of smoothed probabilities,
        with the entry ``gamma[s, i]`` giving the probability of
        finding the system in state ``s`` given *all* of the observations
        up to index ``i``:

        .. math::

            \gamma_{s, i} = P(s | o_1, \ldots, o_i ).

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit`.

        Parameters
        ----------
        gamma : numpy.ndarray, shape=(n_obs, )
            Smoothed probabilities.
        obs : numpy.ndarray, shape=(n_obs, )
            Observations.

        """
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        """ Make a copy of this object.

        This method is called by :py:func:`hsmmlearn.hsmm.HSMMModel.fit` to
        make a copy of the emissions object before modifying it.

        """
        raise NotImplementedError


class MultinomialEmissions(AbstractEmissions):
    """ An emissions class for multinomial emissions.

    This emissions class models the case where the emissions are categorical
    variables, assuming values from 0 to some value k, and the probability
    of observing an emission given a state is modeled by a multinomial
    distribution.

    """

    # TODO this is only used by sample() and can be eliminated by inferring the
    # dtype from the generated samples.
    dtype = np.int64

    def __init__(self, probabilities):
        self._update(probabilities)

    def _update(self, probabilities):
        _probabilities = np.asarray(probabilities)
        # clip small neg residual (GH #34)
        _probabilities[_probabilities < 0] = 0

        xs = np.arange(_probabilities.shape[1])
        _probability_rvs = [
            NonParametricDistribution(xs, ps) for ps in _probabilities
        ]
        self._probabilities = _probabilities
        self._probability_rvs = _probability_rvs

    def weighted_update(self, update_rate, old_multinormal_emissions):
        new_probabilities = update_rate * self._probabilities + (
                1 - update_rate) * old_multinormal_emissions._probabilities
        self._update(new_probabilities)

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        return np.vstack([rv.pmf(obs) for rv in self._probability_rvs])

    def copy(self):
        return MultinomialEmissions(self._probabilities.copy())

    def sample_for_state(self, state, size=None):
        return self._probability_rvs[state].rvs(size=size)

    def reestimate(self, gamma, observations):
        new_emissions = np.empty_like(self._probabilities)
        for em in range(self._probabilities.shape[1]):
            mask = observations == em
            new_emissions[:, em] = (
                    gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )
        self._update(new_emissions)


class GaussianEmissions(AbstractEmissions):
    """ An emissions class for Gaussian emissions.

    This emissions class models the case where emissions are real-valued
    and continuous, and the probability of observing an emission given
    the state is modeled by a Gaussian. The means and standard deviations
    for each Gaussian (one for each state) are stored as state on the
    class.

    """

    dtype = np.float64

    def __init__(self, means, scales):
        self.means = means
        self.scales = scales

    def weighted_update(self, update_rate, old_gaussian_emissions):
        self.means = update_rate * self.means + (
                1 - update_rate) * old_gaussian_emissions.means
        self.scales = update_rate * self.scales + (
                1 - update_rate) * old_gaussian_emissions.scales

    def likelihood(self, obs):
        obs = np.squeeze(obs)
        # TODO: build in some check for the shape of the likelihoods, otherwise
        # this will silently fail and give the wrong results.
        return norm.pdf(obs,
                        loc=self.means[:, np.newaxis],
                        scale=self.scales[:, np.newaxis])

    def sample_for_state(self, state, size=None):
        return norm.rvs(self.means[state], self.scales[state], size)

    def copy(self):
        return GaussianEmissions(self.means.copy(), self.scales.copy())

    def reestimate(self, gamma, observations):
        p = np.sum(gamma * observations[np.newaxis, :], axis=1)
        q = np.sum(gamma, axis=1)
        new_means = p / q

        A = observations[np.newaxis, :] - new_means[:, np.newaxis]
        p = np.sum(gamma * A ** 2, axis=1)
        variances = p / q
        new_scales = np.sqrt(variances)

        self.means = new_means
        self.scales = new_scales


class MultivariateGaussianEmissions(AbstractEmissions):
    dtype = np.float64

    """
    Arguments:

    means (list):    holds the mean-arrays for the n_states for the 
                     n_observables
                   
    cov_list (list): holds n_states n_cont_observables*n_cont_observables 
                     numpy arrays, defining the covariance matrix for each
                     state.
    """

    def __init__(self, means, cov_list):
        self._update(np.array(means), np.array(cov_list), init_cov=True)

    def _update(self, means, cov_list, init_cov=False, epsilon=1e-5):
        """

        :param means: ndarray (n_states,n_obs)
        :param cov_list: ndarray (n_states, n_obs, n_obs)
        :param epsilon: constant added to diagonal to ensure covariance matrices
                        to be psd if no variation of certain observables
        """
        self.means = means
        self.cov_list = cov_list
        n_states, n_obs = self.means.shape
        state_codes = np.arange(n_states)
        if init_cov:
            for state in state_codes:
                self.cov_list[state, :, :] = self.cov_list[state, :, :] + epsilon * np.identity(n_obs)
            self.state_distributions = [
                multivariate_normal(mean=self.means[state, :],
                                    cov=self.cov_list[state, :, :],
                                    allow_singular=False) for state in state_codes]
        else:
            self.state_distributions = [
                multivariate_normal(mean=self.means[state, :],
                                    cov=self.cov_list[
                                        state, :, :], allow_singular=False) for
                state in state_codes]

    def weighted_update(self, update_rate, old_mv_gaussian_emissions):
        self.means = self.means * update_rate + (
                1 - update_rate) * old_mv_gaussian_emissions.means
        self.cov_list = self.cov_list * update_rate + (
                1 - update_rate) * old_mv_gaussian_emissions.cov_list

    def likelihood(self, obs):
        # Todo correlated observables so far uncorrelated only
        return np.vstack(
            [state_rv.pdf(obs) for state_rv in self.state_distributions])

    # likelihood = []
    # for state_rv in self.state_distributions:
    #    likelihood.append(state_rv.pdf(obs))
    # return np.array(likelihood)

    def sample_for_state(self, state, size=None):
        return multivariate_normal.rvs(self.means[state, :],
                                       self.cov_list[state, :, :], size)

    def copy(self):
        return MultivariateGaussianEmissions(self.means.copy(),
                                             self.cov_list.copy())

    def reestimate(self, gamma, observations):
        """
        :param gamma: P(s|O_{1:i})
        :param observations: (np.array: (n_obs,n_observables)
        """
        n_states, n_obs = gamma.shape
        n_observables = observations.shape[1]

        new_mean = []
        for s in range(n_states):
            p = np.zeros(n_observables)
            q = 0
            for i in range(n_obs):
                p += gamma[s, i] * observations[i, :]
                q += gamma[s, i]
            new_mean.append(p / q)
        new_mean = np.array(new_mean)

        # p = np.dot(gamma, observations)
        # q = np.sum(gamma, axis=1)
        # new_mean = p / q

        cov_list = []
        for s in range(n_states):
            p = 0
            q = 0
            for i in range(n_obs):
                dev = observations[i, :] - new_mean[s]
                dev = dev.reshape((n_observables, 1))
                p += gamma[s, i] * np.outer(dev, dev)
                q += gamma[s, i]
            cov_list.append(p / q)

        self._update(np.array(new_mean), np.array(cov_list), init_cov=False)


class GaussianMultinomialMixtureEmissions(AbstractEmissions):
    dtype = np.float64

    """
    Arguments:
    cont_mask (np.array): n_observables array masking the continuous 
                          observables with ones and categorical with zeros
                           
    means (list): n_states*n_cont_observables array with the means for the 
                      continuous observables
                      
    cov_list (list): holds n_states n_cont_observables*n_cont_observables 
                     numpy arrays, defining the covariance for each state. 
    
    cat_probabilities (list): holds the emission probabilities of the
                              n_cat_observables symbols for the n_states
    """

    def __init__(self, cont_mask, cat_probabilities, means, cov_list):
        self._update(np.array(cont_mask), np.array(cat_probabilities),
                     np.array(means), np.array(cov_list), init_cov=True)

    def _update(self, cont_mask, cat_probabilities, means,
                cov_list, init_cov=False, epsilon=1e-5):
        """
        Arguments:

        cont_mask: ndarray (n_observables,)
        cat_probabilities: ndarray (n_states, n_cat_obs)
        means: ndarray (n_states,n_cont_obs)
        cov_list: ndarray (n_states, n_obs, n_obs)
        epsilon: constant added to diagonal to ensure covariance matrices
                        to be psd if there exist invariant observables
        """

        self.cont_mask = cont_mask
        self.means = means
        self.cov_list = cov_list
        n_states, n_obs = self.means.shape
        n_cont_obs = np.sum(cont_mask)
        n_cat_obs = len(cont_mask) - n_cont_obs
        self.state_codes = np.arange(n_states)

        if init_cov:
            self.cont_state_distributions = [
                multivariate_normal(mean=self.means[state, :],
                                    cov=np.identity(n_obs) * epsilon +
                                        self.cov_list[state, :, :],
                                    allow_singular=False) for state
                in self.state_codes]
        else:
            self.cont_state_distributions = [
                multivariate_normal(mean=self.means[state, :],
                                    cov=self.cov_list[
                                        state, :, :], allow_singular=True) for
                state in self.state_codes]

        _probabilities = np.asarray(cat_probabilities)
        # clip small neg residual (GH #34)
        _probabilities[_probabilities < 0] = 0

        xs = np.arange(_probabilities.shape[1])
        _cat_rvs = [
            NonParametricDistribution(xs, ps) for ps in _probabilities
        ]
        self.cat_probabilities = _probabilities
        self.cat_state_distributions = _cat_rvs

    def weighted_update(self, update_rate, old_mixture_emissions):
        means = self.means * update_rate + (
                1 - update_rate) * old_mixture_emissions.means
        cov_list = self.cov_list * update_rate + (
                1 - update_rate) * old_mixture_emissions.cov_list
        cat_probabilities = self.cat_probabilities * update_rate + \
                            (1 - update_rate) * \
                            old_mixture_emissions.cat_probabilities

        # Ensure changes are taken over to the iced distributions
        self._update(self.cont_mask, cat_probabilities, means, cov_list)

    def likelihood(self, obs):
        # Todo correlated observables so far uncorrelated only

        cont_likelihood = np.vstack([cont_state_rv.pdf(obs[:, self.cont_mask])
                                     for cont_state_rv in
                                     self.cont_state_distributions])
        obs_cat = np.squeeze(np.array(obs[:, ~self.cont_mask], dtype=np.int64))
        cat_likelihood = np.vstack([cat_state_rv.pmf(obs_cat)
                                    for cat_state_rv in
                                    self.cat_state_distributions])
        # print(cat_likelihood)
        return (cont_likelihood * cat_likelihood)

    def sample_for_state(self, state, size=None):
        n_obs = len(self.cont_mask)
        obs = np.zeros((size, n_obs,))
        obs_cont = multivariate_normal.rvs(self.means[state, :],
                                           self.cov_list[state, :, :],
                                           size=size)
        obs[:, self.cont_mask] = obs_cont
        obs_cat = self.cat_state_distributions[state].rvs(
            size=size)
        obs[:, ~self.cont_mask] = np.reshape(obs_cat, (size, 1))
        return (obs)

    def copy(self):
        return GaussianMultinomialMixtureEmissions(self.cont_mask.copy(),
                                                   self.cat_probabilities.copy(),
                                                   self.means.copy(),
                                                   self.cov_list.copy())

    def reestimate(self, gamma, observations):
        """
        gamma: P(s|O_{1:i})
        observations: (np.array: (n_obs,n_observables)
        """
        n_states, n_obs = gamma.shape
        n_observables = observations.shape[1]
        n_cont_observables = np.sum(self.cont_mask)
        n_cat_observables = n_observables - n_cont_observables

        new_mean = []
        for s in range(n_states):
            p = np.zeros(n_cont_observables)
            q = 0
            # n_coag = 0
            for i in range(n_obs):
                # if observations[i, 2] == 1:
                # n_coag += 1
                p += gamma[s, i] * observations[i, self.cont_mask]
                q += gamma[s, i]
            new_mean.append(p / q)
        new_mean = np.array(new_mean)

        # p = np.dot(gamma, observations)
        # q = np.sum(gamma, axis=1)
        # new_mean = p / q

        cov_list = []
        for s in range(n_states):
            p = 0
            q = 0
            for i in range(n_obs):
                dev = observations[i, self.cont_mask] - new_mean[s]
                dev = dev.reshape((n_cont_observables, 1))
                p += gamma[s, i] * np.matmul(dev, dev.transpose())
                q += gamma[s, i]
            cov_list.append(p / q)

        new_emissions = np.empty_like(self.cat_probabilities)
        for em in range(self.cat_probabilities.shape[1]):
            mask = observations[:, ~self.cont_mask] == em
            mask = np.squeeze(mask)
            new_emissions[:, em] = (
                    gamma[:, mask].sum(axis=1) / gamma.sum(axis=1)
            )

        self._update(np.array(self.cont_mask), np.array(new_emissions),
                     np.array(new_mean), np.array(cov_list))
