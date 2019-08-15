"""建立一个数字识别模型
"""

import os
import numpy as np
from tensorflow import keras
from config import Config
from data.preprocess import get_data


class DigitClassifier(object):
    def __init__(self):
        self.model = None
        self.epochs_trained = 0

        # 读取或建立模型
        if os.path.exists(Config.model_save_path):
            self.model = keras.models.load_model(Config.model_save_path)
            # 读取已训练的epochs数目
            with open(Config.model_epochs_trained_path) as f:
                line = f.readlines()[0]
                self.epochs_trained = int(line)
        else:
            self.build_model()
        self.model.summary()
        # 读取数据
        self.data = get_data()

    def build_model(self):
        """建立模型
        """
        input_data = keras.Input(shape=(28, 28, 1))  # (28, 28, 1)

        conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same'
        )(input_data)  # (28, 28, 32)
        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)  # (14, 14, 32)

        conv2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same'
        )(pool1)  # (14, 14, 64)
        pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)  # (7, 7, 64)

        fc3 = keras.layers.Conv2D(
            filters=1024,
            kernel_size=(7, 7),
            padding='VALID'
        )(pool2)  # (1, 1, 1024)
        dropout3 = keras.layers.Dropout(0.5)(fc3)  # (1, 1, 1024)

        logit = keras.layers.Conv2D(filters=10, kernel_size=(1, 1))(dropout3)  # (1, 1, 10)
        logit = keras.layers.Reshape(target_shape=(10,))(logit)
        logit = keras.layers.Softmax(axis=-1)(logit)  # (10,)

        model = keras.Model(inputs=input_data, outputs=logit)
        optimizer = keras.optimizers.Adam(Config.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, epoch_num):
        """训练模型
        """
        # tensorboard有关信息
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=Config.tensorboard_logdir_path,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        self.model.fit(
            x=self.data['train_image'],
            y=self.data['train_label'],
            epochs=epoch_num,
            batch_size=Config.batch_size,
            verbose=1,
            initial_epoch=self.epochs_trained,
            callbacks=[
                keras.callbacks.ModelCheckpoint(Config.model_save_path, save_weights_only=False),
                tb_callback]
        )

        # 保存epochs_trained
        self.epochs_trained = epoch_num
        with open(Config.model_epochs_trained_path, 'w') as f:
            f.write(str(self.epochs_trained))

    def predict(self, data):
        """输入指定大小的灰度图，返回结果
        """
        y_pred = self.model.predict(data)
        y_pred = np.argmax(y_pred, axis=-1)
        return y_pred

    def test(self):
        y_pred = self.model.predict(self.data['test_image'])
        y_pred = np.argmax(y_pred, axis=-1)
        y = self.data['test_label']

        n = y.shape[0]
        m = (y == y_pred).sum()

        print('%s %s %s' % (n, m, m/n))


def main():
    model = DigitClassifier()
    model.train(3)
    model.test()


if __name__ == '__main__':
    main()
