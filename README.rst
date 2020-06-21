ELORA
=====

*Elo regressor algorithm*

.. image:: https://travis-ci.org/morelandjs/melo.svg?branch=master
    :target: https://travis-ci.org/morelandjs/melo

Documentation
-------------

Coming soon...

Quick start
-----------

Requirements: Python 2.7 or 3.3+ with numpy_ and scipy_.

Install the latest release with pip_::

   pip install elora

Example usage::

   import pkgutil
   import numpy as np
   from elora import Elora

   # the package comes pre-bundled with an example dataset
   pkgdata = pkgutil.get_data('melo', 'nfl.dat').splitlines()
   dates, teams_home, scores_home, teams_away, scores_away = zip(
       *[l.split() for l in pkgdata[1:]])

   # define a binary comparison statistic
   spreads = [int(h) - int(a) for h, a
       in zip(scores_home, scores_away)]

   # hyperparameters and options
   k = 0.245
   regress = lambda months: .413 if months > 3 else 0
   regress_unit = 'month'
   commutes = False

   # initialize the estimator
   nfl_spreads = Elora(
      k, regress=regress,
      regress_unit=regress_unit, commutes=False
   )

   # fit the estimator to the training data
   nfl_spreads.fit(dates, teams_home, teams_away, spreads, biases=2.6)

   # specify a comparison time
   time = nfl_spreads.last_update

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC')
   print('CLE VS KC: {}'.format(mean))

   # rank nfl teams at end of 2018 regular season
   rankings = nfl_spreads.rank(time, statistic='mean')
   for team, rank in rankings:
       print('{}: {}'.format(team, rank))

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
