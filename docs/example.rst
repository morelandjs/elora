.. _example:

Example
=======

The ``elora`` package comes pre-bundled with a text file containing the final
score of all regular season NFL games 2009–2018.
Let's load this dataset and use the model to predict the point spread and point
total of "future" games using the historical game data.

Training data
-------------

First, let's import ``Elora`` and load the ``nfl.dat`` package data.
Let's also load numpy_ for convenience. ::

   import pkgutil
   import numpy as np

   from elora import Elora

   # the package comes pre-bundled with an example NFL dataset
   pkgdata = pkgutil.get_data('elora', 'nfl.dat').splitlines()

The ``nfl.dat`` package data looks like this:

.. code-block:: text

   # date, home, score, away, score
   2009-09-10 PIT 13 TEN 10
   2009-09-13 ATL 19 MIA  7
   2009-09-13 BAL 38 KC  24
   2009-09-13 CAR 10 PHI 38
   2009-09-13 CIN  7 DEN 12
   2009-09-13 CLE 20 MIN 34
   2009-09-13 HOU  7 NYJ 24
   2009-09-13 IND 14 JAC 12
   2009-09-13 NO  45 DET 27
   2009-09-13 TB  21 DAL 34
   2009-09-13 ARI 16 SF  20
   2009-09-13 NYG 23 WAS 17
   ...

After we've loaded the package data, we'll need to split the game data into
separate columns. ::

   dates, teams_home, scores_home, teams_away, scores_away = zip(
       *[l.split() for l in pkgdata[1:]])

Point spread predictions
------------------------

Let's start by analyzing the home team point spreads. ::

   spreads = [int(h) - int(a) for h, a in zip(scores_home, scores_away)]

Note, if I swap the order of ``scores_home`` and ``scores_away``, my definition
of the point spread picks up a minus sign.
This means the point spread binary comparison *anti*-commutes under label
interchange.
Let's define a new constant to pass this information to the ``Elora`` class
constructor. ::

   commutes = False

Much like traditional Elo ratings, the ``elora`` model includes a hyperparameter
``k`` that controls how fast the ratings update.
Prior experience indicates that ::

   k = 0.245

is a good choice for NFL games.
Generally speaking, this hyperparameter must be tuned for each use case.

We'll also select an Elora regressor ``scale`` parameter to set the
standard deviation of our comparison predictions.
A larger scale parameter indicates greater uncertainty. ::

  scale = 13.5

These parameters will be passsed to the ``elora`` class constructor momentarily.
First, we'll want to subclass the ``elora.Elora`` regressor in order to further
customize some of its class methods.
Namely, we'll redefine the ``regression_coeff`` class method so that it
regresses our ratings to their median value by a fixed fraction each
offseason. ::

   class EloraNFL(Elora):
       def regression_coeff(self, elapsed_time):
           elapsed_days = elapsed_time / np.timedelta64(1, 'D')
           return .6 if elapsed_days > 90 else 1

Using the previous components, the model estimator is initialized as follows: ::

   # instantiate the estimator class object
   nfl_spreads = EloraNFL(k, scale=scale, commutes=False)

Note that at this point we've not yet trained the model on any data; we've
simply specified various hyperparameters and auxillary options.
The model is trained by calling its fit function on the training data: ::

   nfl_spreads.fit(dates, teams_home, teams_away, spreads)

Once the model is conditioned to the data, we can easily generate predictions by
calling its various instance methods: ::

   # time one day after the last model update
   time = nfl_spreads.last_update_time + np.timedelta64(1, 'D')

   # predict the mean outcome at 'time'
   nfl_spreads.mean(time, 'CLE', 'KC')

   # predict the interquartile range at 'time'
   nfl_spreads.quantile([.25, .5, .75], time, 'CLE', 'KC')

   # predict the win probability at 'time'
   nfl_spreads.sf(0.5, time, 'CLE', 'KC')

   # generate prediction samples at 'time'
   nfl_spreads.sample(time, 'CLE', 'KC', size=100)

Furthermore, the model can rank teams by their expected performance against a
league average opponent on a neutral field.
Let's evaluate this ranking at the end of the 2018–2019 season. ::

   # end of the 2018–2019 season
   time = nfl_spreads.last_update_time + np.timedelta64(1, 'D')

   # rank teams by expected mean spread against average team
   nfl_spreads.rank(time)

Point total predictions
-----------------------

Everything demonstrated so far can also be applied to point total comparisons
with a few small changes.
First, let's create the array of point total comparisons. ::

   totals = [int(h) + int(a) for h, a in zip(scores_home, scores_away)]

Next, we'll need to set ::

   commutes = True

since the point total comparisons are invariant under label interchange.
Finally, we'll need to provide somewhat different inputs for the ``k`` and
``scale`` hyperparameters, and the ``regression_coeff`` class method: ::

   k = .03
   scale = 13.5

   class EloraNFL(Elora):
       def regression_coeff(self, elapsed_time):
           elapsed_days = elapsed_time / np.timedelta64(1, 'D')
           return .6 if elapsed_days > 90 else 1

Putting all the pieces together, ::

   nfl_totals = EloraNFL(k, scale=scale, commutes=False)

   nfl_totals.fit(dates, teams_home, teams_away, totals)

And voila! We can easily predict the outcome of a future point total
comparison. ::

   # time one day after the last model update
   time = nfl_totals.last_update_time + np.timedelta64(1, 'D')

   # predict the mean outcome at 'time'
   nfl_totals.mean(time, 'CLE', 'KC')


.. _numpy: http://www.numpy.org
