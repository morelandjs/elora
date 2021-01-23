elora
=====

*Elo regressor algorithm (elora)*

* Author: J\. Scott Moreland
* Language: Python
* Source code: `github:morelandjs/elora <https://github.com/morelandjs/elora>`_

``elora`` generalizes the `Bradley-Terry <https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model>`_ paired comparison model beyond binary outcomes to include margin-of-victory information.
The framework is general and has numerous applications in ranking, estimation, and time series prediction.

Quick start
-----------

Requirements: Python 2.7 or 3.3+ with numpy_ and scipy_.

Install the latest release with pip_: ::

   pip install elora

Example usage: ::

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

   # subclass the elora estimator to modify default behavior (optional)
   class EloraNFL(Elora):
       def regression_coeff(self, elapsed_time):
           elapsed_days = elapsed_time / np.timedelta64(1, 'D')
           return .6 if elapsed_days > 90 else 1

   # instantiate the estimator class object
   nfl_spreads = EloraNFL(0.07, scale=14, commutes=False)

   # fit the estimator to the training data
   nfl_spreads.fit(dates, teams_home, teams_away, spreads)

   # specify a comparison time
   time = nfl_spreads.last_update_time

   # predict the mean outcome at that time
   mean = nfl_spreads.mean(time, 'CLE', 'KC')
   print('CLE VS KC: {}'.format(mean))

   # rank nfl teams at end of 2018 regular season
   rankings = nfl_spreads.rank(time)
   for team, rank in rankings:
       print('{}: {}'.format(team, rank))

.. toctree::
   :caption: User guide
   :maxdepth: 2

   usage
   example

.. toctree::
   :caption: Technical info
   :maxdepth: 2

   theory
   tests

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _scipy: https://www.scipy.org
