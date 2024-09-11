import torch
import torch.nn.functional as F
import logging


def calculate_psnr(pre_image, image):
    sum_PSNR = 0
    for i in range(len(image)):
        mse = torch.mean((pre_image[i] - image[i]) ** 2).to('cpu')
        psnr = 10*torch.log10(1/mse)
        sum_PSNR += psnr.detach().data.item()
    return sum_PSNR / len(image)


def pad_image(image, patch_size):
    h, w = image.shape[-2:]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    image = F.pad(image, (0, pad_w, 0, pad_h))
    return image



class Logger:
    #######################################################################################################################################
    ######################  Code based on https://github.com/sanechips-multimedia/syenet/logger.py ########################################
    #######################################################################################################################################
    def __init__(self,args):
        self.args = args
        self.log_path = args.save_path

        self.logging_level = logging.DEBUG
        self.file_level = logging.DEBUG
        self.stream_level = logging.DEBUG

        self.logger = logging.getLogger('logger.log')
        self.logger.setLevel(self.logging_level)

        self.configure()

    def configure(self):
        log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(log_format)

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(log_format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
