# -*- coding: utf-8 -*-

from nose.tools import assert_almost_equal

import numpy as np

from ..elora import Elora


def test_class_init():
    """
    Checking elora class constructor

    """
    # dummy class instance
    elora = Elora(0)

    # single comparison
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    # fit to the training data
    elora.fit(time, label1, label2, value)

    # multiple comparisons
    times = np.arange(100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.normal(0, 10, size=100)

    # randomize times
    np.random.shuffle(times)
    elora.fit(times, labels1, labels2, values)
    examples = elora.examples

    # check comparison length
    assert examples.shape == (100,)

    # check that comparisons are sorted
    assert np.array_equal(
        np.sort(examples.time), examples.time)

    # check first and last times
    assert elora.first_update_time == times.min()
    assert elora.last_update_time == times.max()


def test_initial_rating():
    """
    Checking initial rating

    """
    init_rating = np.float(np.random.uniform(-10, 10, 1))

    class EloraTest(Elora):
        def initial_rating(self, time, label):
            return init_rating

    for commutes in [True, False]:
        elora = EloraTest(np.random.rand(1), commutes=commutes)

        times = np.arange(100).astype('datetime64[s]')
        labels1 = np.repeat('alpha', 100)
        labels2 = np.repeat('beta', 100)
        values = np.random.random(100)

        elora.fit(times, labels1, labels2, values)

        # check initial rating for label1
        rating_alpha = elora.get_rating(times[0], 'alpha')
        assert_almost_equal(rating_alpha, init_rating)

        # check initial rating for label2
        rating_beta = elora.get_rating(times[0], 'beta')
        assert_almost_equal(rating_beta, init_rating)


def test_regression_coeff():
    """
    Check rating regression functionality

    """
    sec = np.timedelta64(1, 's')
    step = np.random.randint(0, 100)

    class EloraTest(Elora):
        def initial_rating(self, time, label):
            return 0

        def regression_coeff(self, elapsed_time):
            return 0.5 if elapsed_time > sec else 1

    for commutes in [True, False]:
        elora = EloraTest(np.random.rand(1), commutes=commutes)

        times = np.linspace(0, 1000, 100).astype('datetime64[s]')
        labels1 = np.repeat('alpha', 100)
        labels2 = np.repeat('beta', 100)
        values = np.random.uniform(-10, 10, 100)

        elora.fit(times, labels1, labels2, values)

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
    init_rating = np.float(np.random.uniform(-10, 10, size=1))

    class EloraTest(Elora):
        def initial_rating(self, time, label):
            return init_rating

    k = np.random.uniform(0.01, 1, size=1)
    elora = EloraTest(k, commutes=False)

    times = np.linspace(0, 1000, 100).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 100)
    labels2 = np.repeat('beta', 100)
    values = np.random.uniform(-30, 30, size=100)

    elora.fit(times, labels1, labels2, values)

    # test rating conservation at random times
    for time in np.random.uniform(0, 1000, size=10).astype('datetime64[s]'):
        ratings = [elora.get_rating(time, label) for label in elora.labels]
        assert_almost_equal(sum(ratings), init_rating*elora.labels.size)


def test_get_rating():
    """
    Checking rating query function

    """
    # dummy class instance
    elora = Elora(0)

    # single entry
    time = np.datetime64('now')
    label1 = 'alpha'
    label2 = 'beta'
    value = np.random.uniform(-10, 10)

    # train the model
    elora.fit(time, label1, label2, value)

    # populate record data
    one_hour = np.timedelta64(1, 'h')
    elora.record['alpha'] = np.rec.array(
        [(time - one_hour, 1), (time, 2), (time + one_hour, 3)],
        dtype=[('time', 'datetime64[s]'), ('rating', 'float', 1)]
    )

    # check rating value at time
    rating = elora.get_rating(time, 'alpha')
    assert_almost_equal(rating, 1)

    # check rating value at time plus one hour
    rating = elora.get_rating(time + one_hour, 'alpha')
    assert_almost_equal(rating, 2)


def test_rate():
    """
    Checking core rating function

    """
    # dummy class instance
    k = np.random.rand()
    elora = Elora(k)

    # alpha wins, beta loses
    times = np.arange(2).astype('datetime64[s]')
    labels1 = np.repeat('alpha', 2)
    labels2 = np.repeat('beta', 2)
    values = [1, -1]

    # instantiate ratings
    elora.fit(times, labels1, labels2, values)

    # rating_change = k * (obs - prior) = 2 * (1 - 0)
    rec = elora.record
    assert rec['alpha'].rating[0] == k
    assert rec['beta'].rating[0] == -k
