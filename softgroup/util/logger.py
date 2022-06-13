import logging

<<<<<<< HEAD

##########################################################
# FUNCTIONS TO CREATE NORMAL LOGGER AND TENSORBOARD LOGGER
##########################################################


# SUMMARY WRITER SAVES LOG AS EVENTFILE FOR LATER USE IN TENSORBOARD
=======
>>>>>>> d0ad4a93b778eb9170a433e205baabbc65f5d702
from tensorboardX import SummaryWriter as _SummaryWriter

from .dist import is_main_process, master_only

<<<<<<< HEAD
# CREATE LOGGER AND MAKE SURE THAT IT ONLY OUTPUTS LOG FOR MAIN PROCESS (OTHERWISE REDUNDANT OUTPUT)
=======

>>>>>>> d0ad4a93b778eb9170a433e205baabbc65f5d702
def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('softgroup')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if not is_main_process():
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


<<<<<<< HEAD
# MAKE SURE THAT SUMMARY IS ONLY WRITTEN FOR MAIN PROCESS (NOT SURE THO WHAT EXACTLY WOULD HAPPEN IF @MASTER_ONLY WOULD BE OMMITED)
=======
>>>>>>> d0ad4a93b778eb9170a433e205baabbc65f5d702
class SummaryWriter(_SummaryWriter):

    @master_only
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    @master_only
    def add_scalar(self, *args, **kwargs):
        return super().add_scalar(*args, **kwargs)

    @master_only
    def flush(self, *args, **kwargs):
        return super().flush(*args, **kwargs)
