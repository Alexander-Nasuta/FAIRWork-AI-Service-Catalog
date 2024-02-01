import atexit
import json
import os
import logging.config
import pathlib as pl
import shutil

from utils.wzl_banner import wzl_banner
from utils.fairwork_banner import banner_color as fairwork_banner

log = logging.getLogger("AI-Service")

# print banner when logger is imported
w, h = shutil.get_terminal_size((80, 20))
# print(f"terminal dimensions: {w}x{h}")


#print(small_banner if w < 140 else big_banner)
print(wzl_banner)
print(fairwork_banner)

# some paths to important directories
script_path = pl.Path(__file__).resolve()
utils_dir_path = script_path.parent
src_dir_path = utils_dir_path.parent
project_root_path = src_dir_path.parent

def setup_logging():
    """

    :return:
    """


    """
     ___      _               _                ___  _            _                
    / __| ___| |_ _  _ _ __  | |   ___  __ _  |   \(_)_ _ ___ __| |_ ___ _ _ _  _ 
    \__ \/ -_)  _| || | '_ \ | |__/ _ \/ _` | | |) | | '_/ -_) _|  _/ _ \ '_| || |
    |___/\___|\__|\_,_| .__/ |____\___/\__, | |___/|_|_| \___\__|\__\___/_|  \_, |
                      |_|              |___/                                 |__/ 

    create a log directory for the roatating file handler
    """
    log_dir = project_root_path.joinpath("logs")
    log_dir.mkdir(exist_ok=True)

    """
     _                 _                __ _      
    | |   ___  __ _ __| |  __ ___ _ _  / _(_)__ _ 
    | |__/ _ \/ _` / _` | / _/ _ \ ' \|  _| / _` |
    |____\___/\__,_\__,_| \__\___/_||_|_| |_\__, |
                                            |___/ 
                                            
    loads json file with logging configuration from resources/logging_configs/config.json
    """
    config_file = project_root_path.joinpath("resources", "logging_configs", "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    logging.config.dictConfig(config)


if __name__ == '__main__':
    # change working directory to the root of the project
    # the relative paths in 'config' are relative to the root of the project
    # 'logging.config.dictConfig(config)' parses the relative paths in 'config' relative to the location
    # of the working directory of the process the called the script.
    # if we would not change the working directory here, the relative paths in 'config'
    # would point to different locations depending on where the script was called from
    # (so log files would be written to different locations depending on where the script was called from#
    # to prevent this, we change the working directory to the root of the project
    os.chdir(project_root_path)

    setup_logging()
    log.info(f"working directory: {os.getcwd()}")
    log.info("test")
    # extra information can be added to the log message by passing a dictionary to the extra keyword argument
    # see logs/app.log.jsonl and have a look how it's different from the log above (look for extra_infor_key)
    log.info("test", extra={"extra_info_key": "extra_info_value"})
    log.debug("test")
    log.error("test")
    log.warning("test")
    log.critical("test")
    try:
        1 / 0
    except Exception as e:
        log.exception(e)

