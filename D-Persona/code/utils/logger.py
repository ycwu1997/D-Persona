import logging
import os
import time


class Logger:
    def __init__(self, model_name, path='./log/'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        current_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.log_path = path
        os.makedirs(self.log_path, exist_ok=True)
        assert os.path.exists(self.log_path), 'Make sure the log path is exists.'
        self.log_file_name = os.path.join(self.log_path, model_name+'_'+current_time+'.log')
        self.file_handle = logging.FileHandler(self.log_file_name, mode='w')
        self.file_handle.setLevel(logging.DEBUG)

        # set logger format
        formatter = logging.Formatter("%(message)s")
        self.file_handle.setFormatter(formatter)

        self.logger.addHandler(self.file_handle)

    def write(self, message):
        self.logger.info(message)

    def write_and_print(self, message):
        self.logger.info(message)
        print(message)


if __name__ == '__main__':
    logger = Logger('./log/')
    logger.write('hello world')
