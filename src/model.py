import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

from src.environment.environmentRL import EnvironmentRL
from src.parameters import Parameters as par


def basic_model(env: EnvironmentRL):

    number_outputs = 2 if par.long_positions_only else 3
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()

    price_signals = keras.layers.Input(shape=(env.window_size, 1))
    position = keras.layers.Input(shape=(1,))
    rewards_weights = keras.layers.Input(shape=(len(par.rewards),))
    gamma = keras.layers.Input(shape=(1,))

    x = keras.layers.Flatten()(price_signals)
    merged = keras.layers.Concatenate(axis=1)(
        [x, position, rewards_weights, gamma])
    merged = keras.layers.Dense(
        512 * par.model.scale_NN,
        activation=keras.activations.relu,
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(par.model.l2_penalty),
        bias_regularizer=regularizers.l2(par.model.l2_penalty)
    )(merged)

    if par.model.dropout_level > 0:
        merged = keras.layers.Dropout(par.model.dropout_level)(merged)

    y = keras.layers.Dense(
        256 * par.model.scale_NN,
        activation=keras.activations.relu,
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(par.model.l2_penalty),
        bias_regularizer=regularizers.l2(par.model.l2_penalty)
    )(merged)

    if par.model.dropout_level > 0:
        y = keras.layers.Dropout(par.model.dropout_level)(y)

    y = keras.layers.Dense(
        128 * par.model.scale_NN,
        activation=keras.activations.relu,
        kernel_initializer=init,
        kernel_regularizer=regularizers.l2(par.model.l2_penalty),
        bias_regularizer=regularizers.l2(par.model.l2_penalty)
    )(y)

    x = keras.layers.Dense(
        number_outputs,
        activation=keras.activations.linear,
        kernel_initializer=init, bias_initializer="zeros"
    )(y)

    model = keras.models.Model(
        inputs=[price_signals, position, rewards_weights, gamma], outputs=x)

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'])

    return model
