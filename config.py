import os


class Config(object):
    tensorboard_logdir_path = os.path.join('.', 'log')
    model_save_path = os.path.join('.', 'model', 'model.h5')
    model_epochs_trained_path = os.path.join('.', 'model', 'epoch_num')

    batch_size = 64
    learning_rate = 0.001
