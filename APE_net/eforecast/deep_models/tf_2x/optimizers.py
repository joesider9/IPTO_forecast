import tensorflow as tf


def get_rated(rated, y):
    if rated is not None:
        norm_val = tf.constant(1, tf.float32, name='rated')
    else:
        norm_val = y
    return norm_val


def optimize_bulk(learning_rate, rated=None, probabilistic=False, quantiles=None, steps=5000):
    learning_rate_ = tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate, steps/50, t_mul=1.0,
                                                                       m_mul=1.0,
                                                                       alpha=learning_rate / 10)
    # Create losses
    if probabilistic:
        raise NotImplementedError('probabilistic')
        # losses = []
        # for i, q in enumerate(quantiles):
        #     error = y - model_output[i]
        #     loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error),
        #                           axis=-1)
        #
        #     losses.append(loss)
        # err_out = tf.add_n(losses)
        # cost_out = tf.reduce_mean(err_out)
    else:
        loss_out = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    train_out = tf.keras.optimizers.Adam(learning_rate=learning_rate_)
    if rated is not None:
        accuracy_out = tf.keras.metrics.MeanAbsoluteError()
        sse_out = tf.keras.metrics.MeanSquaredError()
    else:
        accuracy_out = tf.keras.metrics.MeanAbsolutePercentageError()
        sse_out = tf.keras.metrics.MeanSquaredError()
    return train_out, loss_out, accuracy_out, sse_out


def optimize(is_global=False, rated=None, learning_rate=1e-4, is_fuzzy=False,
             probabilistic=False, quantiles=None, n_batch=100, epochs=600):
    with tf.name_scope("optimizers") as scope:
        trainers, losses, MAEs, SSEs = optimize_bulk(learning_rate, rated=rated,
                                                     probabilistic=probabilistic,
                                                     quantiles=quantiles,
                                                     steps=n_batch * epochs)

    return trainers, losses, MAEs, SSEs, learning_rate
