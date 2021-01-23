Usage
=====

``elora`` is a computer model to generate rankings and predictions from paired comparison time series data.
It has obvious applications to sports, but the framework is general and can be used for numerous other purposes including consumer surveys and asset pricing.

Overview
--------

This is a brief overview of the ``elora`` Python package.
See `Theory <theory.html>`_ for an explanation of the underlying math.

1. Initialization
^^^^^^^^^^^^^^^^^

First, import the Elora class. ::

   from elora import Elora

Next, create a Elora class object and specify its constructor arguments. ::

   elora_instance = Elora(k, scale=scale, commutes=commutes)

Parameters
""""""""""

* **k** *(float)* -- At bare minimum, you'll need to specify the rating update factor k which is the first and only positional argument. The k factor controls the magnitude of each rating update, with larger k values making the model more responsive to each comparison outcome. Its value should be chosen by minimizing the model's predictive error.

* **scale** *(float, optional)* -- The scale parameter sets the standard deviation `\sigma` of the normal distribution `\mathcal{N}(\mu, \sigma^2)` used to model paired comparison outcomes. If you make the scale parameter small, the predicted outcomes become more deterministic, and if you make it large the predictions become more uncertain. The default value is 1.

* **commutes** *(bool, optional)* -- This parameter describes the expected behavior of the estimated values under label interchange. If commutes=False, it is assumed that the comparisons anti-commute under label interchange (default behavior), and if commutes=True, it is assumed they commute. For example, point totals require commutes=True and point spreads require commutes=False.

2. Training data
^^^^^^^^^^^^^^^^

Each ``elora`` training input is a tuple of the form ``(time, label1, label2)`` and each training output is a single number ``value``.
This training data is passed to the model as four array_like objects of equal length:

* **times** is an array_like object of type np.datetime64 (or compatible string). It specifies the time at which the comparison was made.
* **labels1** and **labels2** are array_like objects of type string. They specify the first and second label names of the entities involved in the comparison.
* **values** is an array_like object of type float. It specifies the numeric value of the comparison, e.g. the value of the point spread or point total.

.. warning::
   It is assumed that the elements of each array match up, i.e\. the n-th element of each array should correspond to the same comparison.
   It is not necessary that the comparisons are time ordered.

For example, the data used to train the model might look like the following: ::

   times = ['2009-09-10', '2009-09-13', '2009-09-13']
   labels1 = ['PIT', 'ATL', 'BAL']
   labels2 = ['TEN', 'MIA', 'KC']
   values = [3, 12, 14]

3. Model calibration
^^^^^^^^^^^^^^^^^^^^

The model is calibrated by calling the fit function on the training data. ::

   elora_instance.fit(times, labels1, labels2, values, biases=0)

Optionally, when training the model you can specify ``biases`` (float or array_like of floats). These are numbers which add to (or subtract from) the rating difference of each comparison, i.e.

.. math::
   \Delta R = R_\text{label1} - R_\text{label2} + \text{bias}.

These factors can be used to account for transient advantages and disadvantages such as weather and temporary injuries.
Positive bias numbers increase the expected value of the comparison, and negative values decrease it.
If ``biases`` is a single number, the bias factor is assumed to be constant for all comparisons.
Otherwise, there must be a bias factor for every training input.

.. note::
   The model automatically accounts for global spread bias such as that associated with home field advantage.
   To take advantage of this functionality, the label entries should be ordered such that the bias is alligned with the first (or second) label.

4. Making predictions
^^^^^^^^^^^^^^^^^^^^^

Once the model is fit to the training data, there are a number of different functions which can be called to generate predictions for new comparisons at arbitrary points in time.

At its most basic level, the model estimates for each comparison (matchup) the parameters `\mu` and `\sigma` of the normal distribution `\mathcal{N}(\mu, \sigma^2)` used to model that matchup outcome.
Once these parameters are known, the statistical properties of the comparison such as its mean value, PDF, and CDF are easily evaluated: ::

   elora_instance.mean(times, labels1, labels2, biases=biases)

   elora_instance.pdf(x, times, labels1, labels2, biases=biases)

   elora_instance.cdf(x, times, labels1, labels2, biases=biases)

...as well as arbitrary percentiles (or quantiles) of the distribution ::

   elora_instance.percentile([10, 50, 90], times, labels1, labels2, biases=biases)

and it can even draw samples from the estimated survival function probability distribution ::

   elora_instance.sample(times, labels1, labels2, biases=biases, size=100)

Perhaps one of the most useful applications of the model is using its mean and median predictions to create rankings. This is aided by the rank function ::

   elora_instance.rank(time)

which ranks the labels at the specified time according to their expected performance against an average opponent, i.e. an opponent with an average rating.

Reference
---------

Main class
^^^^^^^^^^
.. autoclass:: elora.Elora.__init__

Training function
"""""""""""""""""
.. autofunction:: elora.Elora.fit

Prediction functions
""""""""""""""""""""
.. autofunction:: elora.Elora.cdf

.. autofunction:: elora.Elora.sf

.. autofunction:: elora.Elora.pdf

.. autofunction:: elora.Elora.percentile

.. autofunction:: elora.Elora.quantile

.. autofunction:: elora.Elora.mean

.. autofunction:: elora.Elora.residuals

.. autofunction:: elora.Elora.rank

.. autofunction:: elora.Elora.sample
