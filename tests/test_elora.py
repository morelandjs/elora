# -*- coding: utf-8 -*-
import numpy as np
import pytest

from elora import Elora


def test_class_init():
    """
    Checking elora class constructor

    """
    # single comparison
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    elora = Elora(time, label1, label2, value)
    elora.fit(0, commutes=True)

    # multiple comparisons
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.normal(0, 10, size=100)

    # randomize times
    np.random.shuffle(times)
    elora = Elora(times, labels1, labels2, values)
    elora.fit(0, commutes=True)
    examples = elora.examples

    # check comparison length
    assert examples.shape == (100,)

    # check that comparisons are sorted
    assert np.array_equal(
        np.sort(examples.time), examples.time)

    # check first and last times
    assert elora.first_update_time == times.min()
    assert elora.last_update_time == times.max()


def test_equilibrium_rating():
    """
    Check that ratings regress to equilibrium rating

    """
    equilibrium_rating = float(np.random.uniform(-10, 10))

    class EloraTest(Elora):

        @property
        def initial_rating(self):
            return equilibrium_rating

        def regression_coeff(self, elapsed_time):
            return 1e-3

    samples = 10**4
    times = np.arange(samples).astype('datetime64[s]')
    labels1 = np.repeat('alpha', samples)
    labels2 = np.repeat('beta', samples)
    values = np.random.random(samples)

    for commutes in [True, False]:
        k = 1e-2
        elora = EloraTest(times, labels1, labels2, values)
        elora.fit(k, commutes=commutes)

        # check equilibrium rating for label1
        rating_alpha = elora.get_rating(times[-1], 'alpha')
        assert rating_alpha == pytest.approx(equilibrium_rating, abs=k)

        # check equilibrium rating for label2
        rating_beta = elora.get_rating(times[-1], 'beta')
        assert rating_beta == pytest.approx(equilibrium_rating, abs=k)


def test_regression_coeff():
    """
    Check rating regression functionality

    """
    sec = np.timedelta64(1, 's')
    step = np.random.randint(0, 100)

    class EloraTest(Elora):
        @property
        def initial_rating(self):
            return 0

        def regression_coeff(self, elapsed_time):
            return 0.5 if elapsed_time > sec else 1

    for commutes in [True, False]:
        times = np.linspace(0, 1000, 100).astype('datetime64[s]')
        labels1 = np.repeat('alpha', 100)
        labels2 = np.repeat('beta', 100)
        values = np.random.uniform(-10, 10, 100)

        elora = EloraTest(times, labels1, labels2, values)
        elora.fit(np.random.rand(), commutes=commutes)

        # test rating regression for label1
        rating_alpha = elora.get_rating(times[step] + sec, 'alpha')
        rating_alpha_regressed = elora.get_rating(times[step] + 2*sec, 'alpha')
        assert rating_alpha_regressed == 0.5*rating_alpha

        # test rating regression for label2
        rating_beta = elora.get_rating(times[step] + sec, 'beta')
        rating_beta_regressed = elora.get_rating(times[step] + 2*sec, 'beta')
        assert rating_beta_regressed == 0.5*rating_beta


def test_rating_conservation():
    """
    Check that rating is conserved when commutes is False

    """
    _k = np.random.uniform(low=0.01, high=1)

    times = np.linspace(0, 1000, 100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.uniform(-30, 30, size=100)

    elora = Elora(times, labels1, labels2, values)
    elora.fit(_k, commutes=False)

    # test rating conservation at random times
    for time in np.random.uniform(0, 1000, size=10).astype('datetime64[s]'):
        ratings = [elora.get_rating(time, label) for label in elora.labels]
        assert sum(ratings) == pytest.approx(0, abs=1e-4)


def test_get_rating():
    """
    Checking rating query function

    """
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    elora = Elora(time, label1, label2, value)
    elora.fit(0, commutes=True)

    # populate record data
    one_hour = np.timedelta64(1, 'h')
    elora.record['alpha'] = np.rec.array(
        [(time - one_hour, 1), (time, 2), (time + one_hour, 3)],
        dtype=[('time', 'datetime64[s]'), ('rating', 'float')]
    )

    # check rating value at time
    rating = elora.get_rating(time, 'alpha')
    assert rating == pytest.approx(1, abs=1e-4)

    # check rating value at time plus one hour
    rating = elora.get_rating(time + one_hour, 'alpha')
    assert rating == pytest.approx(2, abs=1e-4)


def test_rate():
    """
    Checking core rating function

    """
    k = np.random.rand()
    times = np.arange(2).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 2)
    labels2 = np.repeat('beta', 2)
    values = [1, -1]

    elora = Elora(times, labels1, labels2, values)
    elora.fit(k, commutes=False)

    # rating_change = k * (obs - prior) = k * (1 - 0) = k
    rec = elora.record
    assert rec['alpha'].rating[0] == k
    assert rec['beta'].rating[0] == -k
