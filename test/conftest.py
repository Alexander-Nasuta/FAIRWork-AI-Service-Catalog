import pytest
import os

from utils.project_paths import project_root_path


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # change working directory to the root of the project
    # the relative paths in 'config' are relative to the root of the project
    # 'logging.config.dictConfig(config)' parses the relative paths in 'config' relative to the location
    # of the working directory of the process the called the script.
    # if we would not change the working directory here, the relative paths in 'config'
    # would point to different locations depending on where the script was called from
    # (so log files would be written to different locations depending on where the script was called from#
    # to prevent this, we change the working directory to the root of the project
    #os.chdir(project_root_path)

    yield  # This is where the testing happens
