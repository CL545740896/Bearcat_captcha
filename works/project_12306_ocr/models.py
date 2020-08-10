# 模型
import tensorflow as tf
from works.project_12306_ocr.settings import LR
from works.project_12306_ocr.settings import N_CLASS
from works.project_12306_ocr.settings import IMAGE_HEIGHT
from works.project_12306_ocr.settings import IMAGE_WIDTH
from works.project_12306_ocr.settings import CAPTCHA_LENGTH
from works.project_12306_ocr.settings import IMAGE_CHANNALS
from works.project_12306_ocr.settings import CAPTCHA_CHARACTERS_LENGTH


class Models(object):

    @staticmethod
    def xception_model(fine_ture_at=3):
        covn_base = tf.keras.applications.Xception(include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                                   pooling='max')
        model = tf.keras.Sequential()
        model.add(covn_base)
        model.add(tf.keras.layers.Dense(512, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
        covn_base.trainable = False
        for layer in covn_base.layers[:fine_ture_at]:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['categorical_accuracy'])
        return model

    @staticmethod
    def captcha_model():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(
            x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(32, 32, 64), stage=1)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(64, 64, 128), stage=3)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(128, 128, 256), stage=5)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        # x = Models.dense_block(x, 32, stage=7)
        # x = Models.dense_block(x, 64, stage=9)
        # x = Models.dense_block(x, 128, stage=11)
        # x = Models.dense_block(x, 256, stage=13)
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax)(
            x)
        x = tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH))(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        return model

    @staticmethod
    def dense_block(x, units=128, stage=2, block='a'):
        '''
        全连接的残差(短接)
        :param x: 输入层
        :param units: 输出维度空间
        :param stage:名字
        :param block:名字
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        x_shortcut = x
        x = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, name=conv_name_base)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base)(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        return x

    @staticmethod
    def inception_block(x, filters=64, stage=2, block='a'):
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        x_shortcut = x
        x1 = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation=tf.keras.activations.relu,
                                    name=conv_name_base + 'a')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation=tf.keras.activations.relu,
                                    name=conv_name_base + 'b')(x)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x3 = tf.keras.layers.Conv2D(filters, kernel_size=5, padding='same', activation=tf.keras.activations.relu,
                                    name=conv_name_base + 'c')(x)
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x4 = tf.keras.layers.Conv2D(filters, kernel_size=5, padding='same', activation=tf.keras.activations.relu,
                                    name=conv_name_base + 'd')(x)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x = tf.keras.layers.concatenate([x1,x2,x3,x4])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.concatenate([x,x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x


    @staticmethod
    def identity_block(x, f=3, filters=(32, 64, 128), stage=2, block='a'):
        '''
        残差块跳跃连接(短接)
        :param x:输入层
        :param filters:列表或元组，长度为3，指定卷积核数量
        :param stage:名字
        :param block:名字
        :param f:卷积核的大小
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        F1, F2, F3 = filters
        x_shortcut = x
        x = tf.keras.layers.Conv2D(filters=F1, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2a',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2b',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2c',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    @staticmethod
    def convolutional_block(x, f=3, filters=(32, 64, 128), stage=2, block='a'):
        '''
        残差块跳跃连接
        :param x:输入层
        :param filters:列表或元组，长度为3，指定卷积核数量
        :param stage:名字
        :param block:名字
        :param f:卷积核的大小
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        F1, F2, F3 = filters
        x_shortcut = x
        x = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=1, padding="same", name=conv_name_base + '2a',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2b',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2c',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding="same",
                                            name=conv_name_base + '1',
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                            activation=tf.keras.activations.relu)(x_shortcut)
        x_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x_shortcut)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    @staticmethod
    def captcha_12306():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same',
                                   activation=tf.keras.activations.relu)(x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(32, 32, 64), stage=2, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(128, 128, 256), stage=5, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(256, 256, 512), stage=9, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(N_CLASS, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        return model

    @staticmethod
    def captcha_12306_ocr():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation=tf.keras.activations.relu, padding='same')(
            x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(32, 32, 64), stage=1)
        # x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        # x = Models.identity_block(x, filters=(32, 64, 128), stage=2)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(64, 64, 128), stage=3)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(128, 128, 256), stage=4)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dense(CAPTCHA_LENGTH * N_CLASS, activation=tf.keras.activations.softmax)(x)
        x = tf.keras.layers.Reshape((CAPTCHA_LENGTH, N_CLASS))(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
        return model

    @staticmethod
    def captcha_12306_ocr_s():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation=tf.keras.activations.relu, padding='same')(
            x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.inception_block(x, filters=32, stage=1)
        # x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        # x = Models.identity_block(x, filters=(32, 64, 128), stage=2)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.inception_block(x, filters=64, stage=2)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = Models.inception_block(x, filters=128, stage=3)
        x = tf.keras.layers.MaxPool2D(strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dense(CAPTCHA_LENGTH * N_CLASS, activation=tf.keras.activations.softmax)(x)
        x = tf.keras.layers.Reshape((CAPTCHA_LENGTH, N_CLASS))(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
        return model

if __name__ == '__main__':
    model = Models.captcha_12306_ocr_s()
    model.summary()
