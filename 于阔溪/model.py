import tensorflow as tf

num_classes = 2


def camp_model():
    inputs = tf.keras.layers.Input(shape=(199, 40, 1))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    numOutput_P = 64 * x.get_shape()[2]
    x = tf.keras.layers.Reshape((-1, numOutput_P))(x)
    # x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.GRU(1024, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GRU(1024, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GRU(1024, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu'))(x)
    # x = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(x)
    # x = tf.keras.layers.Dense(units=512, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, x)
    return model

