"""Weighted two-sample Kolmogorov-Smirnov statistic calculation with TensorFlow"""

from typing import Optional, Tuple

import tensorflow as tf

def get_ecdf(
        sample: tf.Tensor,
        weights: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get empirical CDF from a weighted 1D sample
    """

    #assert len(sample.shape) == 1, "Only 1D CDF is implemented"

    if weights is None:
        weights = tf.ones_like(sample)

    #assert sample.shape == weights.shape
    
    i = tf.contrib.framework.argsort(sample)
    x, w = tf.gather(sample, i), tf.gather(weights, i)
    
    w_cumsum = tf.cumsum(w)
    
    #assert w_cumsum[-1] > 0
    
    w_cumsum /= w_cumsum[-1]
    
    return x, w_cumsum


def _interleave_ecdfs(
                x1: tf.Tensor,
                y1: tf.Tensor,
                x2: tf.Tensor,
                y2: tf.Tensor
            ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Interleave two eCDFs by their argument
    """
    #assert len(x1.shape) == len(x2.shape) == 1
    #assert x1.shape == y1.shape
    #assert x2.shape == y2.shape

    x = tf.contrib.framework.sort(tf.concat([x1, x2], axis=0))
    y1 = tf.concat([[0], y1], axis=0)
    y2 = tf.concat([[0], y2], axis=0)
    return (
        x,
        tf.gather(y1, tf.searchsorted(x1, x, side='right')),
        tf.gather(y2, tf.searchsorted(x2, x, side='right'))
    )

def ks_2samp_w(
            data1: tf.Tensor,
            data2: tf.Tensor,
            w1: Optional[tf.Tensor] = None,
            w2: Optional[tf.Tensor] = None
        ) -> float:
    cdf1 = get_ecdf(data1, w1)
    cdf2 = get_ecdf(data2, w2)
    _, cdf1_i, cdf2_i = _interleave_ecdfs(*cdf1, *cdf2)
    return tf.reduce_max(tf.abs(cdf2_i - cdf1_i))