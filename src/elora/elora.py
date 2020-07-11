from operator import add, sub
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Elora:
    """
    Elo regressor algorithm (elora)

    Analytic implemention of margin-dependent Elo assuming normally
    distributed outcomes.

    Author: J. Scott Moreland

    """
    def __init__(self, k, scale=1, commutes=False):
        """
        Args:
            k (float): prefactor multiplying the rating exhanged between a pair
                of labels for a given comparison
            scale (float): scale factor for the distribution used to model the
                outcome of the comparison variable; must be greater than 0
            commutes (bool): true if comparisons commute under label
                interchange; false otherwise (default is false)

        Attributes:
            first_update_time (np.datetime64): time of the first comparison
            last_update_time (np.datetime64): time of the last comparison
            mean_value (float): mean expected comparison value
            labels (array of string): unique compared entity labels
            examples (ndarray): comparison training examples
            record (dict of ndarray): record of time and rating states

        """
        if k < 0:
            raise ValueError('k must be a non-negative real number')

        if scale <= 0:
            raise ValueError('scale must be a positive real number')

        self.k = k
        self.scale = scale
        self.commutes = commutes
        self.compare = add if self.commutes else sub
        self.dtype = [('time', 'datetime64[s]'), ('rating', 'float')]

        self.first_update_time = None
        self.last_update_time = None
        self.mean_value = None
        self.commutator = None
        self.labels = None
        self.examples = None
        self.record = None

    def initial_rating(self, time, label):
        """
        Customize this function for a given subclass.

        It initializes ratings as a function of time and label.

        Default initialization behavior is to return one-half the
        mean outcome value if the labels commute, otherwise 0.

        """
        return .5*self.mean_value if self.commutes else 0

    def regression_coeff(self, elapsed_time):
        """
        Customize this function for a given subclass.

        It computes the regression coefficient — prefactor multiplying the
        rating of each team evaluated at each update — as a function of
        elapsed time since the last rating update for that label.

        Default behavior is to return 1, i.e. no rating regression.

        """
        return 1

    def _examples(self, times, labels1, labels2, values, biases):
        """
        Reads training examples and initializes class variables.

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)
        values = np.array(values, dtype='float', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        self.first_update_time = times.min()
        self.last_update_time = times.max()
        self.mean_value = values.mean()
        self.commutator = 0 if self.commutes else self.mean_value
        self.labels = np.union1d(labels1, labels2)

        self.examples = np.sort(
            np.rec.fromarrays([
                times,
                labels1,
                labels2,
                values,
                biases,
            ], names=(
                'time',
                'label1',
                'label2',
                'value',
                'bias',
            )), order=['time', 'label1', 'label2'], axis=0)

    def evolve_state(self, label, state, time):
        """
        Evolves 'state' to 'time', applying rating regression if necessary

        Args:
            state (dict): state dictionary {'time': time, 'rating': rating}
            time (np.datetime64): time to evaluate state

        Returns:
            state (dict): evolved state dictionary
                {'time': time, 'rating': rating}

        """
        current_rating = state['rating']
        elapsed_time = time - state['time']

        initial_rating = self.initial_rating(time, label)
        regress = self.regression_coeff(elapsed_time)

        rating = regress * current_rating + (1 - regress) * initial_rating

        return {'time': time, 'rating': rating}

    def get_rating(self, times, labels):
        """
        Query label state(s) at the specified time accounting
        for rating regression.

        Args:
            times (array of np.datetime64): Comparison datetimes
            labels (array of string): Comparison entity labels

        Returns:
            rating (array): ratings for each time and label pair

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels = np.array(labels, dtype='str', ndmin=1)

        ratings = []

        for time, label in zip(times, labels):
            try:
                label_record = self.record[label]
                index = label_record.time.searchsorted(time)
                prior_state = label_record[index - 1]
                state = self.evolve_state(label, prior_state, time)
                rating = state['rating']
            except (KeyError, IndexError):
                rating = self.initial_rating(time, label)

            ratings.append(rating)

        return np.squeeze(ratings)

    def fit(self, times, labels1, labels2, values, biases=0):
        """
        Calibrates the model based on the training examples.

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correction factors,
                default value is 0

        """
        # read training inputs and initialize class variables
        self._examples(times, labels1, labels2, values, biases)

        # initialize empty record for each label
        self.record = {label: [] for label in self.labels}

        # initialize state for each label
        prior_state = {
            label: {
                'time': self.first_update_time,
                'rating': self.initial_rating(self.first_update_time, label)}
            for label in self.labels}

        # loop over all paired comparison training examples
        for time, label1, label2, value, bias in self.examples:

            state1 = self.evolve_state(label1, prior_state[label1], time)
            state2 = self.evolve_state(label2, prior_state[label2], time)

            value_prior = (
                self.compare(state1['rating'], state2['rating']) +
                self.commutator + bias)

            rating_change = self.k * (value - value_prior)

            sign = 1 if self.commutes else -1
            state1['rating'] += rating_change
            state2['rating'] += sign*rating_change

            # record current ratings
            for label, state in [(label1, state1), (label2, state2)]:
                self.record[label].append((state['time'], state['rating']))
                prior_state[label] = state.copy()

        # convert ratings history to a structured rec.array
        for label in self.record.keys():
            self.record[label] = np.rec.array(
                self.record[label], dtype=self.dtype)

    def cdf(self, x, times, labels1, labels2, biases=0):
        """
        Computes the comulative distribution function (CDF) for each
        comparison, i.e. prob(value < x).

        Args:
            x (array of float): threshold of comparison for each value
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            y (array of float): cumulative distribution function value
                for each input

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return norm.cdf(x, loc=loc, scale=self.scale)

    def sf(self, x, times, labels1, labels2, biases=0):
        """
        Computes the survival function (SF) for each
        comparison, i.e. prob(value > x).

        Args:
            x (array of float): threshold of comparison for each value
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            y (array of float): survival function value for each input

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return np.squeeze(norm.sf(x, loc=loc, scale=self.scale))

    def pdf(self, x, times, labels1, labels2, biases=0):
        """
        Computes the probability distribution function (PDF) for each
        comparison, i.e. P(x).

        Args:
            x (array of float): input values
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            y (array of float): probability density at each input

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return np.squeeze(norm.pdf(x, loc=loc, scale=self.scale))

    def percentile(self, p, times, labels1, labels2, biases=0):
        """
        Computes percentiles p of the probability distribution.

        Args:
            p (array of float): percentiles to evaluate (in range [0, 100])
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            x (array of float): values of the distribution corresponding to
                each percentile

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        p = np.true_divide(p, 100.0)

        if np.count_nonzero(p < 0.0) or np.count_nonzero(p > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        return np.squeeze(norm.ppf(p, loc=loc, scale=self.scale))

    def quantile(self, q, times, labels1, labels2, biases=0):
        """
        Computes quantiles q of the probability distribution.
        Same as percentiles but accepts values [0, 1].

        Args:
            q (array of float): quantiles to evaluate (in range [0, 1])
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            x (array of float): values of the distribution corresponding to
                each quantile

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return np.squeeze(
            norm.ppf(q, loc=loc[:, np.newaxis], scale=self.scale))

    def mean(self, times, labels1, labels2, biases=0):
        """
        Computes the mean of the probability distribution.

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        Returns:
            y (array of float): mean of the probability distribution

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return np.squeeze(loc)

    def residuals(self, y_true=None, standardize=False):
        """
        Computes residuals of the model predictions for each training example

        Args:
            standardize (bool): if True, the residuals are standardized to unit
                variance

        Returns:
            residuals (array of float): residuals for each example

        """
        y_pred = self.mean(
            self.examples.time,
            self.examples.label1,
            self.examples.label2,
            self.examples.bias)

        if y_true is None:
            y_true = self.examples.value

        residuals = y_true - y_pred

        if standardize is True:

            quantiles = [.159, .841]

            qlo, qhi = self.quantile(
                quantiles,
                self.examples.time,
                self.examples.label1,
                self.examples.label2,
                self.examples.bias
            ).T

            residuals /= .5*abs(qhi - qlo)

        return residuals

    def rank(self, time):
        """
        Ranks labels by comparing mean of each label to the average label.

        Args:
            time (np.datetime64): time at which the ranking should be computed.

        Returns:
            label rankings (list of tuples): returns a rank sorted list of
                (label, rank) pairs, where rank is the comparison value of
                the specified summary statistic.

        """
        ranked_list = [
            (label, np.asscalar(self.get_rating(time, label)))
            for label in self.labels]

        return sorted(ranked_list, key=lambda v: v[1], reverse=True)

    def sample(self, times, labels1, labels2, biases=0, size=1):
        """
        Draw random samples from the predicted comparison probability
        distribution.

        Args:
            times (array_like of np.datetime64): list of datetimes.
            labels1 (array_like of string): list of first entity labels.
            labels2 (array_like of string): list of second entity labels.
            biases (array_like of float, optional): single bias number or
                list of bias numbers which match the comparison inputs.
                Default is 0, in which case no bias is used.
            size (int, optional): number of samples to be drawn.
                default is 1, in which case a single value is returned.

        Returns:
            x (array of float): random samples for the comparison outcome

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)
        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        loc = self.compare(ratings1, ratings2) + self.commutator + biases

        return norm.rvs(loc=loc, scale=self.scale, size=size)

    def plot_ratings(self, label_regex):
        """
        Plot rating history for all labels matching the label regex

        """
        labels = [
            label for label in self.record.keys()
            if re.match(label_regex, label)]

        for label in labels:
            time = self.record[label].time
            rating = self.record[label].rating
            plt.plot(time, rating, 'o', label=label)

        plt.xlabel('time')
        plt.ylabel('rating')
        plt.legend()
        plt.show()
