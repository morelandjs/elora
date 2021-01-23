ELORA
=====

*Elo regressor algorithm (elora)*

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
   pkgdata = pkgutil.get_data('elora', 'nfl.dat').splitlines()
   dates, teams_home, scores_home, teams_away, scores_away = zip(
       *[l.split() for l in pkgdata[1:]])

   # define a binary comparison statistic
   spreads = [int(h) - int(a) for h, a
       in zip(scores_home, scores_away)]

   # hyperparameters and options
   k = 0.06
   scale = 13.5
   commutes = False

   # initialize the estimator
   nfl_spreads = Elora(k, scale=scale, commutes=commutes)

   # fit the estimator to the training data
   nfl_spreads.fit(dates, teams_home, teams_away, spreads, biases=2.6)

   # specify a comparison time
   time = nfl_spreads.last_update_time

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC')
   print('CLE VS KC: {}'.format(mean))

   # rank nfl teams at end of 2018 regular season
   rankings = nfl_spreads.rank(time)
   for team, rank in rankings:
       print('{}: {}'.format(team, rank))

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
