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

@pytest.fixture(scope="function")
def valid_single_instance():
    import numpy as np
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    yield custom_jsp_instance

@pytest.fixture(scope="function")
def invalid_triple_instance():
    import numpy as np
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    yield custom_jsp_instance

@pytest.fixture(scope="function")
def valid_triple_instance():
    import numpy as np
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    yield custom_jsp_instance

@pytest.fixture(scope="function")
def invalid_triple_instance():
    import numpy as np
    custom_jsp_instance = np.array([
        [
            [0, 1, 2, 3],  # job 0
            [0, 2, 1, 3]  # job 1
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])
    yield custom_jsp_instance
