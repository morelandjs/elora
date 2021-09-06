from operator import add, sub

import numpy as np
from scipy.stats import norm


class Elora:
    def __init__(self, times, labels1, labels2, values, biases=0):
        """
        Elo regressor algorithm for paired comparison time series prediction

        Author: J. Scott Moreland

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison outcome values
            biases (array of float or scalar, optional): comparison bias
                corrections

        Attributes:
            examples (np.recarray): time-sorted numpy record array of
                (time, label1, label2, bias, value, value_pred) samples
            first_update_time (np.datetime64): time of the first comparison
            last_update_time (np.datetime64): time of the last comparison
            labels (array of string): unique compared entity labels
            median_value (float): median expected comparison value

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
        self.labels = np.union1d(labels1, labels2)
        self.median_value = np.median(values)

        prior = self.median_value * np.ones_like(values, dtype=float)

        self.examples = np.sort(
            np.rec.fromarrays([
                times,
                labels1,
                labels2,
                biases,
                values,
                prior,
            ], names=(
                'time',
                'label1',
                'label2',
                'bias',
                'value',
                'value_pred'
            )), order=['time', 'label1', 'label2'], axis=0)

    @property
    def initial_rating(self):
        """
        Customize this function for a given subclass.

        It computes the initial rating, equal to the rating one would
        expect if all labels were interchangeable.

        Default behavior is to return one-half the median outcome value
        if the labels commute, otherwise 0.

        """
        return .5*self.median_value if self.commutes else 0

    def regression_coeff(self, elapsed_time):
        """
        Customize this function for a given subclass.

        It computes the regression coefficient—prefactor multiplying the
        rating of each team evaluated at each update—as a function of
        elapsed time since the last rating update for that label.

        Default behavior is to return 1, i.e. no rating regression.

        """
        return 1.0

    def evolve_rating(self, rating, elapsed_time):
        """
        Evolves 'state' to 'time', applying rating regression if necessary,
        and returns the evolved rating.

        Args:
            state (dict): state dictionary {'time': time, 'rating': rating}
            time (np.datetime64): time to evaluate state

        Returns:
            state (dict): evolved state dictionary
                {'time': time, 'rating': rating}

        """
        regress = self.regression_coeff(elapsed_time)

        return regress * rating + (1.0 - regress) * self.initial_rating

    def fit(self, k, commutes, scale=1, burnin=0):
        """
        Primary routine that performs model calibration. It is called
        recursively by the `fit` routine.

        Args:
            k (float): coefficient that multiplies the prediction error to
                determine the rating update.
            commutes (bool): false if the observed values change sign under
                label interchange and true otheriwse.

        """
        self.commutes = commutes
        self.scale = scale
        self.commutator = 0. if commutes else self.median_value
        self.compare = add if commutes else sub

        record = {label: [] for label in self.labels}
        prior_state_dict = {}

        for idx, example in enumerate(self.examples):
            time, label1, label2, bias, value, value_pred = example

            default = (time, self.initial_rating)
            prior_time1, prior_rating1 = prior_state_dict.get(label1, default)
            prior_time2, prior_rating2 = prior_state_dict.get(label2, default)

            rating1 = self.evolve_rating(prior_rating1, time - prior_time1)
            rating2 = self.evolve_rating(prior_rating2, time - prior_time2)

            value_pred = self.compare(rating1, rating2) + self.commutator + bias
            self.examples[idx]['value_pred'] = value_pred

            rating_change = k * (value - value_pred)
            rating1 += rating_change
            rating2 += rating_change if self.commutes else -rating_change

            record[label1].append((time, rating1))
            record[label2].append((time, rating2))

            prior_state_dict[label1] = (time, rating1)
            prior_state_dict[label2] = (time, rating2)

        for label in record.keys():
            record[label] = np.rec.array(
                record[label], dtype=[
                    ('time', 'datetime64[s]'), ('rating', 'float')])

        self.record = record

        residuals = np.rec.fromarrays([
            self.examples.time,
            self.examples.value - self.examples.value_pred
        ], names=('time', 'residual'))

        return residuals

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
        ratings = np.empty_like(times, dtype='float')

        for idx, (time, label) in enumerate(zip(times, labels)):
            try:
                label_record = self.record[label]
                index = label_record.time.searchsorted(time)
                prev_index = max(index - 1, 0)
                prior_state = label_record[prev_index]
                rating = self.evolve_rating(
                    prior_state.rating, time - prior_state.time)
            except KeyError:
                rating = self.initial_rating

            ratings[idx] = rating

        return ratings

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
            (label, self.get_rating(time, label).item())
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
