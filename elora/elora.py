import numpy as np
from scipy.stats import norm


class Elora:
    """
    Elo regressor algorithm (elora)

    Analytic implemention of margin-dependent Elo assuming normally
    distributed outcomes.

    Author: J. Scott Moreland

    """
    def __init__(self, k, scale=1, regress=lambda t: 1, regress_unit='year',
                 commutes=False):
        """
        Args:
            k (float): prefactor multiplying the rating exhanged between a pair
                of labels in a given comparison
            scale (float): scale factor for the distribution used to model the
                outcome of the comparison variable; must be greater than 0.
            regress (float function of float): regression coefficient as a
                function of elapsed time, expressed in units of `regress_unit`.
            regress_unit (str): time units of `regress` function input.

        Attributes:
            initial_time (np.datetime64): time of the first comparison
            labels (array of string): unique compared entity labels
            examples (ndarray): comparison training examples
            record (dict of ndarray): record of time and rating states

        """
        if k < 0:
            raise ValueError('k must be a non-negative real number')

        if scale <= 0:
            raise ValueError('scale must be a positive real number')

        if not callable(regress):
            raise ValueError('regress must be univariate scalar function')

        self.k = k
        self.scale = scale
        self.regress = regress
        self.seconds_per_period = {
            'year': 3.154e7,
            'month': 2.628e6,
            'week': 604800.,
            'day': 86400.,
            'hour': 3600.,
            'minute': 60.,
            'second': 1.,
            'millisecond': 1e-3,
            'microsecond': 1e-6,
            'nanosecond': 1e-9,
            'picosecond': 1e-12,
            'femtosecond': 1e-15,
            'attosecond': 1e-18,
        }[regress_unit]
        self.commutes = commutes
        self.commutator = 1 if commutes else -1

        self.mean_val = None
        self.initial_time = None
        self.labels = None
        self.examples = None
        self.record = None

    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

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

        self.mean_val = values.mean()

        self.initial_time = times.min()

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
            )), order='time', axis=0)

    def evolve_state(self, state, time):
        """
        Evolves 'state' to 'time', applying rating regression if necessary

        Args:
            state (dict): state dictionary {'time': time, 'rating': rating}
            time (np.datetime64): time to evaluate state

        Returns:
            state (dict): evolved state dictionary
                {'time': time, 'rating': rating}

        """
        elapsed_seconds = (time - state['time']) / np.timedelta64(1, 's')
        elapsed_periods = elapsed_seconds / self.seconds_per_period

        regress_coeff = self.regress(elapsed_periods)

        return {'time': time, 'rating': regress_coeff * state['rating']}

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
            index = self.record[label].time.searchsorted(time)
            prior_state = self.record[label][max(index - 1, 0)]
            state = self.evolve_state(prior_state, time)
            ratings.append(state['rating'])

        return np.squeeze(ratings)

    def fit(self, times, labels1, labels2, values, biases=0):
        """
        Calibrates the model based on the training examples.

        Args:
            times (array of np.datetime64): comparison datetimes
            labels1 (array of str): comparison labels for first entity
            labels2 (array of str): comparison labels for second entity
            values (array of float): comparison value observed outcomes
            biases (array of float): comparison bias correct factors,
                default value is 0

        """
        # read training inputs and initialize class variables
        self._examples(times, labels1, labels2, values, biases)

        # initialize empty record for each label
        self.record = {label: [] for label in self.labels}

        # initialize state for each label
        prior_rating = .5*self.mean_val if self.commutes else 0
        prior_state = {
            label: {'time': self.initial_time, 'rating': prior_rating}
            for label in self.labels
        }

        # loop over all paired comparison training examples
        for time, label1, label2, value, bias in self.examples:

            state1 = self.evolve_state(prior_state[label1], time)
            state2 = self.evolve_state(prior_state[label2], time)

            value_prior = state1['rating'] + self.commutator*state2['rating']

            rating_change = self.k * (value - value_prior)

            state1['rating'] += rating_change
            state2['rating'] += self.commutator*rating_change

            # record current ratings
            for label, state in [(label1, state1), (label2, state2)]:
                self.record[label].append((state['time'], state['rating']))
                prior_state[label] = state.copy()

        # convert ratings history to a structured rec.array
        for label in self.record.keys():
            self.record[label] = np.rec.array(
                self.record[label], dtype=[
                    ('time', 'datetime64[s]'),
                    ('rating', 'float')
                ]
            )

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

        loc = ratings1 + self.commutator*ratings2 + biases

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

        loc = ratings1 + self.commutator*ratings2 + biases

        return norm.sf(x, loc=loc, scale=self.scale)

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

        loc = ratings1 + self.commutator*ratings2 + biases

        return norm.pdf(x, loc=loc, scale=self.scale)

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

        loc = ratings1 + self.commutator*ratings2 + biases

        p = np.true_divide(p, 100.0)

        if np.count_nonzero(p < 0.0) or np.count_nonzero(p > 1.0):
            raise ValueError("percentiles must be in the range [0, 100]")

        return norm.ppf(p, loc=loc, scale=self.scale)

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

        loc = ratings1 + self.commutator*ratings2 + biases

        return norm.ppf(q, loc=loc, scale=self.scale)

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

        loc = ratings1 + self.commutator*ratings2 + biases

        return loc

    def residuals(self, standardize=False):
        """
        Computes residuals of the model predictions for each training example

        Args:
            standardize (bool): if True, the residuals are standardized to unit
                variance

        Returns:
            residuals (array of float): residuals for each example

        """
        predicted = self.mean(
            self.examples.time,
            self.examples.label1,
            self.examples.label2,
            self.examples.bias
        )

        observed = self.examples.value

        residuals = observed - predicted

        if standardize is True:

            qlo, qhi = self.quantile(
                self.examples.time,
                self.examples.label1,
                self.examples.label2,
                self.examples.bias,
                q=[.159, .841],
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
            for label in self.labels
        ]

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

        loc = ratings1 + self.commutator*ratings2 + biases

        if np.isscalar(biases):
            biases = np.full_like(times, biases, dtype='float')
        else:
            biases = np.array(biases, dtype='float', ndmin=1)

        if size < 1 or not isinstance(size, int):
            raise ValueError("sample size must be a positive integer")

        return norm.rvs(loc=loc, scale=self.scale, size=size)
