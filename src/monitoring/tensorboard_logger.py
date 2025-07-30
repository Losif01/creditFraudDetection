# src/monitoring/tensorboard_logger.py
# this uses observer pattern
from tensorflow.keras.callbacks import TensorBoard
import datetime

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_graph=True)

    def get_callback(self):
        return self.callback