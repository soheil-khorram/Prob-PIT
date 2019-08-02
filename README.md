.. -*- mode: rst -*-

# Prob-PIT

This repository explains a TensorFlow implementation of the probabilistic permutation invariant training (Prob-PIT) which improves and extends the conventional PIT for DNN-based speech separation systems. The details of the Prob-PIT technique are presented in the following paper:

Yousefi M., Khorram S., Hansen J., "Probabilistic Permutation Invariant Training for Speech Separation", Interspeech, 2019.

Conventional PIT
----------------

Let's first explain the standard PIT loss function. Assume we are solving a two-talker speech separation task. x_out1 and x_out2 are the outputs of the speech separation network and x_trg1 and x_trg2 are the clean speech signals. The goal of the network is to make the set {x_out1, x_out2} as close as possible to the set {x_trg1, x_trg2}. To do so, PIT minimizes the following loss function.  

.. code-block:: python

    import tensorflow as tf
    
    def calc_pow(x):
        return tf.reduce_sum(tf.pow(x, 2), 1)

    cost1 = tf.reduce_mean(calc_pow(x_out1 - x_trg1) + calc_pow(x_out2 - x_trg2), 1)
    cost2 = tf.reduce_mean(calc_pow(x_out1 - x_trg2) + calc_pow(x_out2 - x_trg1), 1)
    idx = tf.cast(cost1 > cost2, tf.float32)
    min_cost = idx * cost2 + (1 - idx) * cost1
    pit_loss = tf.reduce_sum(min_cost)

In this code snippet, cost1 and cost2 are the costs of two possible permutations. (For S sources, we will have S! permutations and therefore we need to calculate S! costs).

Prob-PIT
--------

Prob-PIT uses the soft (regularized) minimum function to calculate the optimization loss. Following equation expresses the soft minimum for two variables X1 and X2: 
.. code-block:: python

    soft_min(X1, X2) = - gamma * log( exp(-X1/gamma) + exp(-X2/gamma) )

where gamma is a smoothing factor. gamma = 0 reduces the soft-min function to the standard min function. 

Note that to ensure numerical stability of the soft-min function, we always employ the log-sum-exp stabilization trick: 

.. code-block:: python

    Xmin = min(X1, X2)
    Xmax = max(x1, x2)
    soft_min(X1, X2) = Xmin - gamma * log( 1 + exp((Xmin - Xmax)/gamma) )

Which is completely equivalent to the previous definition of the soft-min function. Considering to the above equations, tensorflow implementation of the Prob-PIT will be:

.. code-block:: python

    import tensorflow as tf

    def calc_pow(x):
        return tf.reduce_sum(x, 2), 1)

    cost1 = tf.reduce_mean(calc_pow(x_out1 - x_trg1) + calc_pow(x_out2 - x_trg2), 1)
    cost2 = tf.reduce_mean(calc_pow(x_out1 - x_trg2) + calc_pow(x_out2 - x_trg1), 1)
    idx = tf.cast(cost1 > cost2, tf.float32)
    min_cost = idx * cost2 + (1 - idx) * cost1
    max_cost = (1 - idx) * cost2 + idx * cost1
    smooth_cost = min_cost - gamma * tf.log(tf.exp((min_cost - max_cost) / gamma) + 1)
    prob_pit_loss = tf.reduce_sum(smooth_cost)

References
----------

.. [1] Midia Yousefi, Soheil Khorram, John H.L. Hansen.
       *Probabilistic Permutation Invariant Training for Speech Separation*
       In: Interspeech, 2019.

Author
------

- Soheil Khorram, 2019
