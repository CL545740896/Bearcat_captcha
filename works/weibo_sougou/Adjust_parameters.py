# from __future__ import print_function
# import numpy as np
# import tensorflow as tf
# from loguru import logger
# from settings import BATCH_SIZE
# from settings import train_pack_path
# from settings import validation_pack_path
# from settings import test_pack_path
# from settings import IMAGE_HEIGHT
# from settings import IMAGE_WIDTH
# from settings import CAPTCHA_LENGTH
# from settings import IMAGE_CHANNALS
# from settings import CAPTCHA_CHARACTERS_LENGTH
# from Function_API import Image_Processing
# from Function_API import parse_function_verification
# from hyperas import optim
# from hyperopt import Trials, STATUS_OK, tpe
# from hyperas.distributions import choice, uniform
#
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(device=gpu, enable=True)
#
# logger.add('Adjust_parameters.txt', mode='a', encoding='utf-8')
# logger.debug('开始调参')
#
#
# def data():
#     train_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(train_pack_path)).map(
#         parse_function_verification).batch(BATCH_SIZE)
#
#     validation_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(validation_pack_path)).map(
#         parse_function_verification).batch(
#         BATCH_SIZE)
#     test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
#         parse_function_verification).batch(BATCH_SIZE)
#     return train_dataset, validation_dataset, test_dataset
#
#
# def create_model(train_dataset, validation_dataset, test_dataset):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
#     model.add(
#         tf.keras.layers.SeparableConv2D({{choice([16, 32, 64, 128, 256, 512])}},
#                                         kernel_size={{choice([3, 5])}}, strides={{choice([(1, 1), (2, 2)])}},
#                                         padding={{choice(['same', 'valid'])}},
#                                         activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                              tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(
#         tf.keras.layers.SeparableConv2D({{choice([16, 32, 64, 128, 256, 512])}},
#                                         kernel_size={{choice([3, 5])}}, strides={{choice([(1, 1), (2, 2)])}},
#                                         padding={{choice(['same', 'valid'])}},
#                                         activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                              tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.MaxPool2D(pool_size={{choice([(1, 1), (2, 2)])}}, strides={{choice([1, 2])}},
#                                         padding={{choice(['same', 'valid'])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(
#         tf.keras.layers.SeparableConv2D({{choice([16, 32, 64, 128, 256, 512])}},
#                                         kernel_size={{choice([3, 5])}}, strides={{choice([(1, 1), (2, 2)])}},
#                                         padding={{choice(['same', 'valid'])}},
#                                         activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                              tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(
#         tf.keras.layers.SeparableConv2D({{choice([16, 32, 64, 128, 256, 512])}},
#                                         kernel_size={{choice([3, 5])}}, strides={{choice([(1, 1), (2, 2)])}},
#                                         padding={{choice(['same', 'valid'])}},
#                                         activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                              tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.MaxPool2D(pool_size={{choice([(1, 1), (2, 2)])}}, strides={{choice([1, 2])}},
#                                         padding={{choice(['same', 'valid'])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense({{choice([16, 32, 64, 128, 256, 512])}},
#                                     activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                          tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(tf.keras.layers.Dense({{choice([16, 32, 64, 128, 256, 512])}},
#                                     activation={{choice([tf.keras.activations.relu, tf.keras.activations.selu,
#                                                          tf.keras.activations.elu])}}))
#     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(
#         tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
#     model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
#     model.compile(optimizer={{choice([
#         tf.keras.optimizers.Adam(amsgrad=True), tf.keras.optimizers.Nadam(), tf.keras.optimizers.Adagrad(),
#         tf.keras.optimizers.Adamax(), tf.keras.optimizers.RMSprop(), tf.keras.optimizers.SGD(),
#         tf.keras.optimizers.Adadelta()])}},
#         loss=tf.keras.losses.categorical_crossentropy,
#         metrics=['categorical_accuracy'])
#     result = model.fit(train_dataset, batch_size={{choice([4, 8, 16, 32, 64, 128])}}, epochs=20, verbose=1,
#                        validation_data=validation_dataset)
#     validation_acc = np.amax(result.history['val_categorical_accuracy'])
#     logger.debug(f'Best validation acc of epoch:{validation_acc}')
#     return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
#
#
# if __name__ == '__main__':
#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=1,
#                                           trials=Trials())
#     train_dataset, validation_dataset, test_dataset = data()
#     logger.debug("Evalutation of best performing model:")
#     logger.debug(best_model.evaluate(test_dataset))
#     logger.debug("Best performing model chosen hyper-parameters:")
#     logger.debug(best_run)
