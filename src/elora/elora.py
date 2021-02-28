from operator import add, sub

import numpy as np
from scipy.stats import multivariate_normal
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
            pairs (np.recarray): time-sorted numpy record array of
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
        biases = biases * np.ones(times.shape)

        pairs = np.rec.fromarrays(
            [times, labels1, labels2, biases],
            names=('time', 'label1', 'label2', 'bias'))

        indices = np.argsort(pairs, order=['time', 'label1', 'label2'])
        self.pairs = pairs[indices]
        self.values = values[indices]
        self.nfeat = values.shape[1]

        # used to scale values to zero mean and unit variance
        self.scaler = StandardScaler(copy=False)

        # set useful class attributes
        self.first_update_time = times.min()
        self.last_update_time = times.max()
        self.labels = np.union1d(labels1, labels2)

    @property
    def initial_rating(self):
        """
        Customize this function for a given subclass.

        It computes the initial rating, equal to the rating one would
        expect if all labels were interchangeable.

        Default behavior is to return one-half the median outcome value
        if the labels commute, otherwise 0.

        """
        return np.zeros(self.nfeat)

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
        anf returns the evolved rating.

        Args:
            state (dict): state dictionary {'time': time, 'rating': rating}
            time (np.datetime64): time to evaluate state

        Returns:
            state (dict): evolved state dictionary
                {'time': time, 'rating': rating}

        """
        regress = self.regression_coeff(elapsed_time)

        return regress * rating + (1.0 - regress) * self.initial_rating

    def format_record(self, record):
        """
        Format record as a numpy record array

        """
        if self.nfeat == 1:
            dtype = [('time', 'datetime64[s]'), ('rating', 'float')]
        else:
            dtype = [('time', 'datetime64[s]'), ('rating', 'float', self.nfeat)]

        for label in record.keys():
            record[label] = np.rec.array(record[label], dtype=dtype)

        return record

    def fit(self, learning_rate, commutes, iterations=3, max_residual=3.):
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
        self.scaler.fit_transform(self.values)
        self.compare = add if commutes else sub

        sign = 1 if commutes else -1
        residuals = np.empty_like(self.values)
        cov = np.array(np.cov(self.values, rowvar=False), ndmin=2)
        prec = np.linalg.inv(cov + 1e-4*np.eye(self.nfeat))

        for iteration in range(iterations):
            record = {label: [] for label in self.labels}
            prior_state_dict = {}

            for idx, (pair, value) in enumerate(zip(self.pairs, self.values)):
                prior_time1, prior_rating1 = prior_state_dict.get(
                    pair.label1, (pair.time, self.initial_rating))
                prior_time2, prior_rating2 = prior_state_dict.get(
                    pair.label2, (pair.time, self.initial_rating))

                rating1 = self.evolve_rating(
                    prior_rating1, pair.time - prior_time1)
                rating2 = self.evolve_rating(
                    prior_rating2, pair.time - prior_time2)

                mean = self.compare(rating1, rating2)
                value_pred = mean + pair.bias
                residual = value - value_pred
                residuals[idx] = residual

                norm = min(max_residual/np.linalg.norm(residual), 1.)
                rating_change = learning_rate * np.dot(prec, norm*residual)
                rating1 += rating_change
                rating2 += sign*rating_change

                record[pair.label1].append((pair.time, rating1))
                record[pair.label2].append((pair.time, rating2))

                prior_state_dict[pair.label1] = (pair.time, rating1)
                prior_state_dict[pair.label2] = (pair.time, rating2)

            cov = np.array(np.cov(residuals, rowvar=False), ndmin=2)
            prec = np.linalg.inv(cov + 1e-4*np.eye(self.nfeat))

        self.cov = self.scaler.var_ * cov
        self.record = self.format_record(record)
        self.scaler.inverse_transform(self.values)

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
        ratings = np.empty((times.size, self.nfeat))

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

        return ratings.squeeze()

    def pdf(self, x, time, label1, label2, bias=0):
        """
        Computes the probability distribution function (PDF) of a single
        comparison, i.e. P(x).

        Args:
            x (array of float): input values
            time (np.datetime64): comparison datetime
            label1 (str): comparison label for first entity
            label2 (str): comparison label for second entity
            value (float): comparison value observed outcome
            bias (float): comparison bias correction factor, default value is 0

        Returns:
            y (array of float): probability density at each input

        """
        rating1 = self.get_rating(time, label1)
        rating2 = self.get_rating(time, label2)

        mean = self.compare(rating1, rating2) + bias

        x = self.scaler.transform(x)

        return multivariate_normal.pdf(x, mean=mean, cov=self.cov).squeeze()

    def cdf(self, x, time, label1, label2, bias=0):
        """
        Computes the cumulative distribution function (CDF) of a single
        comparison, i.e. F(x).

        Args:
            x (array of float): input values
            time (np.datetime64): comparison datetime
            label1 (str): comparison label for first entity
            label2 (str): comparison label for second entity
            value (float): comparison value observed outcome
            bias (float): comparison bias correction factor, default value is 0

        Returns:
            y (array of float): probability density at each input

        """
        rating1 = self.get_rating(time, label1)
        rating2 = self.get_rating(time, label2)

        mean = self.compare(rating1, rating2) + bias

        x = self.scaler.transform(x)

        return multivariate_normal.cdf(x, mean=mean, cov=self.cov).squeeze()

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
        biases = np.array(biases, dtype='float').reshape(times.size, 1)

        ratings1 = self.get_rating(times, labels1)
        ratings2 = self.get_rating(times, labels2)

        mean = self.compare(ratings1, ratings2)
        biases = biases * np.ones_like(mean)

        mean = self.scaler.inverse_transform(mean + biases)

        return np.squeeze(mean)

    def residuals(self, y_true=None):
        """
        Computes residuals of the model predictions for each training example

        Args:
            y_true(array of float, optional): reference values used to compute
                the residuals. default is to use to training values.

        Returns:
            residuals (array of float): residuals for each example

        """
        y_pred = self.mean(
            self.pairs.time,
            self.pairs.label1,
            self.pairs.label2,
            self.pairs.bias)

        if y_true is None:
            y_true = self.values

        residuals = y_true - y_pred

        return residuals

    def ratings(self, time, order_by=None):
        """
        Ranks labels by comparing mean of each label to the average label.

        Args:
            time (np.datetime64): time at which the ranking should be computed
            order_by (int, optional): feature value index to rank by

        Returns:
            label rankings (list of tuples): returns a rank sorted list of
                (label, rank) pairs, where rank is the comparison value of
                the specified summary statistic.

        """
        label_ratings = [
            (label, self.get_rating(time, label))
            for label in self.labels]

        if (order_by is not None) and (self.nfeat == 1):
            return sorted(
                label_ratings, key=lambda v: v[1], reverse=True)
        elif (order_by is not None):
            return sorted(
                label_ratings, key=lambda v: v[1][order_by], reverse=True)
        else:
            return label_ratings

    def sample(self, time, label1, label2, bias=0, size=1):
        """
        Draw random samples from the predicted multivariate-normal probability
        distribution.

        Args:
            time (np.datetime64): datetime of the comparison
            label1 (string): name of the first label
            label2 (string): name of the second label
            bias (float, optional): bias coefficient of the comparison
            size (int, optional): number of samples to be drawn.
                default is 1, in which case a single value is returned.

        Returns:
            x (array of float): random samples for the comparison outcome

        """
        rating1 = self.get_rating(time, label1)
        rating2 = self.get_rating(time, label2)

        mean = self.compare(rating1, rating2) + bias

        mean = self.scaler.inverse_transform(mean)

        return multivariate_normal.rvs(mean, self.cov, size=size).squeeze()


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
    #stats = stats[['points']]
    stats = stats[['points', 'pass_yards', 'pass_attempts', 'rush_yards',
                   'rush_attempts', 'yards_from_penalties']]

    # initialize the estimator
    nfl_spreads = Elora(comp.datetime, comp.team_away, comp.team_home, stats)

    # fit the estimator to the training data
    nfl_spreads.fit(.08, False, iterations=3)

    # specify a comparison time
    last_time = nfl_spreads.last_update_time

    # predict the mean outcome at that time
    away = 'CLE'
    home = 'KAN'
    mean = nfl_spreads.mean(last_time, away, home, biases=0)
    print(f'{away} @{home}: {mean}')

    for label, rating in nfl_spreads.ratings(last_time, order_by=0):
        print(label, rating)
