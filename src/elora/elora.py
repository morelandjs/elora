from operator import add, sub
import time

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class Elora:
    def __init__(self, times, labels1, labels2, values, biases=0):
        """
        Paired comparison regressor assuming sample values distributed
        according to a multivariate normal distribution.

        Author: J. Scott Moreland

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float or scalar, optional): comparison bias
                corrections

        Attributes:
            comparisons (np.recarray): time-sorted numpy record array of
                (time, label1, label2, bias) samples
            values (np.array): time-sorted numpy array of sampled outcomes
            first_update_time (np.datetime64): time of the first comparison
            last_update_time (np.datetime64): time of the last comparison
            median_value (float): median expected comparison value
            labels (array of string): unique compared entity labels

        """
        times = np.array(times, dtype='datetime64[s]', ndmin=1)
        labels1 = np.array(labels1, dtype='str', ndmin=1)
        labels2 = np.array(labels2, dtype='str', ndmin=1)
        values = np.array(values, dtype='float').reshape(times.size, -1)

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        comparisons = np.rec.fromarrays(
            [times, labels1, labels2, biases],
            names=('time', 'label1', 'label2', 'bias'))

        indices = np.argsort(comparisons, order=['time', 'label1', 'label2'])
        self.comparisons = comparisons[indices]
        self.values = values[indices]

        # scale values to zero mean and unit variance
        # NOTE: this must occur before the median value is computed
        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(self.values)

        self.first_update_time = times.min()
        self.last_update_time = times.max()
        self.median_value = np.median(self.values, axis=0)
        self.labels = np.union1d(labels1, labels2)

    @property
    def equilibrium_rating(self):
        """
        Customize this function for a given subclass.

        It computes the equilibrium rating, equal to the rating one would
        expect if all labels were interchangeable.

        Default behavior is to return one-half the median outcome value
        if the labels commute, otherwise 0.

        """
        rating = (
            .5*self.median_value
            if self.commutes else
            np.zeros_like(self.median_value))

        return rating

    def initial_state(self, time, label):
        """
        Customize this function for a given subclass.

        It returns the initial state for a label which has never been
        seen before.

        Default behavior is to return the equilibrium_rating at the
        specified time.

        """
        return {'time': time, 'rating': self.equilibrium_rating}

    def regression_coeff(self, elapsed_time):
        """
        Customize this function for a given subclass.

        It computes the regression coefficient—prefactor multiplying the
        rating of each team evaluated at each update—as a function of
        elapsed time since the last rating update for that label.

        Default behavior is to return 1, i.e. no rating regression.

        """
        return 1.0

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
        rating = state['rating']
        elapsed_time = time - state['time']

        regress = self.regression_coeff(elapsed_time)

        rating = regress * rating + (1.0 - regress) * self.equilibrium_rating

        return {'time': time, 'rating': rating}

    def fit(self, learning_rate, commutes, iterations=3):
        """
        Primary routine that performs model calibration. It is called
        recursively by the `fit` routine.

        Args:
            learning_rate (float): coefficient that multiplies the loss
                gradient to compute the ratings update vector.
            commutes (bool): false if the observed values change sign under
                label interchange and true otheriwse.
            iterations (int): number of times to retrain and refine the
                estimate of the covariance matrix

        """
        # scale values to zero mean and unit variance
        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(self.values)

        # set global class attributes
        self.first_update_time = times.min()
        self.last_update_time = times.max()
        self.median_value = np.median(self.values, axis=0)
        self.labels = np.union1d(labels1, labels2)

        # commutative comparisons change sign under label interchange
        self.commutes = commutes
        sign = 1 if self.commutes else -1
        self.compare = add if commutes else sub
        self.commutator = (
            np.zeros_like(self.median_value)
            if commutes else
            self.median_value)

        # initialize empty record for each label
        self.record = {label: [] for label in self.labels}

        # keep track of prior state and rating for each label
        prior_state_dict = {}

        # track prediction residuals
        residuals = np.empty_like(self.values)

        # initialize covariance terms
        cov = np.array(np.cov(self.values, rowvar=False), ndmin=2)
        prec = np.linalg.inv(cov)

        # refine covariance after each iteration
        for iteration in range(iterations):

            # loop over all paired comparison training examples
            for idx, (comp, value) in enumerate(
                    zip(self.comparisons, self.values)):

                prior_state1 = prior_state_dict.get(
                    comp.label1, self.initial_state(comp.time, comp.label1))
                prior_state2 = prior_state_dict.get(
                    comp.label2, self.initial_state(comp.time, comp.label2))

                state1 = self.evolve_state(comp.label1, prior_state1, comp.time)
                state2 = self.evolve_state(comp.label2, prior_state2, comp.time)

                loc = self.compare(state1['rating'], state2['rating'])
                value_pred = loc + self.commutator + comp.bias
                residuals[idx] = value - value_pred

                rating_change = learning_rate * np.dot(prec, value - value_pred)
                state1['rating'] += rating_change
                state2['rating'] += sign*rating_change

                # record current ratings
                for label, state in [
                        (comp.label1, state1), (comp.label2, state2)]:
                    self.record[label].append((state['time'], state['rating']))
                    prior_state_dict[label] = state.copy()

            # compute and return residual covariance matrix
            cov = np.array(np.cov(residuals, rowvar=False), ndmin=2)
            prec = np.linalg.inv(cov)
            self.scale = np.sqrt(cov.diagonal()).squeeze()

        _, nfeat = self.values.shape
        dtype = [('time', 'datetime64[s]'), ('rating', 'float', nfeat)]

        for label in self.record.keys():
            self.record[label] = np.rec.array(self.record[label], dtype=dtype)

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
                prev_index = max(index - 1, 0)
                prior_state = label_record[prev_index]
                state = self.evolve_state(label, prior_state, time)
            except KeyError:
                state = self.initial_state(time, label)

            ratings.append(state['rating'])

        return np.squeeze(ratings)

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

    def rank(self, time, column_index=0):
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
            (label, self.get_rating(time, label)[column_index])
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


if __name__ == '__main__':
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine('sqlite:///games.db')

    comp = pd.read_sql(
        "SELECT datetime, winning_abbr, losing_abbr, winner FROM games", engine)
    comp['team_home'] = np.where(
        comp.winner == 'Home', comp.winning_abbr, comp.losing_abbr)
    comp['team_away'] = np.where(
        comp.winner == 'Away', comp.winning_abbr, comp.losing_abbr)

    feat = pd.read_sql(
        "SELECT * FROM games", engine
    ).select_dtypes(include=np.number)

    away_stats = feat.filter(regex='^away.*', axis=1)
    away_stats.columns = away_stats.columns.str.replace('away_', '')
    home_stats = feat.filter(regex='^home.*', axis=1)
    home_stats.columns = home_stats.columns.str.replace('home_', '')
    stats = away_stats - home_stats

    # stats = stats[['points']]
    stats = stats[['points', 'pass_yards', 'rush_yards']]
    #stats = stats[['points', 'pass_attempts', 'pass_yards', 'rush_attempts',
    #               'rush_yards', 'third_down_conversions', 'pass_completions',
    #               'yards_from_penalties', 'penalties']]

    # initialize the estimator
    nfl_spreads = Elora(comp.datetime, comp.team_away, comp.team_home, stats)
    print(nfl_spreads.labels)

    # fit the estimator to the training data
    nfl_spreads.fit(.1, False, iterations=3)

    # specify a comparison time
    last_time = nfl_spreads.last_update_time

    # predict the mean outcome at that time
    mean = nfl_spreads.mean(last_time, 'RAV', 'PIT')
    print('RAV @PIT: {}'.format(mean))

    # rank nfl teams at end of 2018 regular season
    rankings = nfl_spreads.rank(last_time)
    for team, rank in rankings:
        print('{}: {}'.format(team, rank))
