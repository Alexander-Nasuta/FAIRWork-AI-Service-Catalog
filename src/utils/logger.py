import atexit
import json
import os
import logging.config
import pathlib as pl
import shutil

from utils.project_paths import project_root_path
from utils.wzl_banner import wzl_banner
from utils.fairwork_banner import banner_color as fairwork_banner

# print banner when logger is imported
w, h = shutil.get_terminal_size((80, 20))
# print(f"terminal dimensions: {w}x{h}")


# print(small_banner if w < 140 else big_banner)
print(wzl_banner)
print(fairwork_banner)

my_dict_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(message)s"
        },
        "detailed": {
            "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z"
        },
        "json": {
            "()": "utils.json_file_logger.MyJSONFormatter",
            "fmt_keys": {
                "level": "levelname",
                "message": "message",
                "timestamp": "timestamp",
                "logger": "name",
                "module": "module",
                "function": "funcName",
                "line": "lineno",
                "thread_name": "threadName"
            }
        }
    },
    "handlers": {
        "stderr": {
            "level": "WARNING",
            "formatter": "simple",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5
        },
        "json_file": {
            "level": "DEBUG",
            "formatter": "json",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log.jsonl",
            "maxBytes": 10485760,
            "backupCount": 5
        },
        "stdout": {
            "()": "rich.logging.RichHandler",
            "show_path": False,
            "level": "INFO",
        }
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": ["stderr", "file", "json_file", "stdout"]
        }
    }
}

no_rotating_file_dict_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(message)s"
        },
        "detailed": {
            "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z"
        },
    },
    "handlers": {
        "stderr": {
            "level": "WARNING",
            "formatter": "simple",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr"
        },
        "stdout": {
            "()": "rich.logging.RichHandler",
            "show_path": False,
            "level": "INFO",
        }
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": ["stderr", "stdout"]
        }
    }
}


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
    # config_file = project_root_path.joinpath("resources", "logging_configs", "config.json")
    # with open(config_file, "r") as f:
    #   config = json.load(f)
    logging.config.dictConfig(my_dict_config)


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
log = logging.getLogger("AI-Service")
log.info(f"working directory: {os.getcwd()}")

if __name__ == '__main__':

    log.info("test")
    # extra information can be added to the log message by passing a dictionary to the extra keyword argument
    # see logs/app.log.jsonl and have a look how it's different from the log above (look for extra_infor_key)
    log.info("test", extra={"extra_info_key": "extra_info_value"})
    log.debug("test")
    log.error("test")
    log.warning("test")
    try:
        1 / 0
    except ZeroDivisionError as e:
        log.exception(e)
