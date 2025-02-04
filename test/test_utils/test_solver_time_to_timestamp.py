from utils.crf_timestamp_solver_time_conversion import solver_time_to_timestamp


def test_solver_time_to_timestamp_starting_monday_6am():
    monday_6am_timestamp = 1693807200  # 2023-09-04 06:00 (Monday)

    expected_results = [
        # (solver_time, start_timestamp, human readable time)
        (0, 1693807200, '04.09.2023 6:00 Monday'),
        (480, 1693836000, '04.09.2023 14:00 Monday'),
        (960, 1693864800, '04.09.2023 22:00 Monday'),

        (2880, 1693980000, '06.09.2023 6:00 Wednesday'),
        (3360, 1694008800, '06.09.2023 14:00 Wednesday'),
        (3840, 1694037600, '06.09.2023 22:00 Wednesday'),

        (5760, 1694152800, '08.09.2023 6:00 Friday'),
        (6240, 1694181600, '08.09.2023 14:00 Friday'),
        (6720, 1694210400, '08.09.2023 22:00 Friday'),

        (7199, 1694239140, '09.09.2023 5:59 Saturday'),

        (7200, 1694412000, '11.09.2023 6:00 Monday'),
        (7680, 1694440800, '11.09.2023 14:00 Monday'),
    ]

    for solver_time, expected_timestamp, human_readable_time in expected_results:
        assert expected_timestamp == solver_time_to_timestamp(
            solver_time=solver_time,
            start_timestamp=monday_6am_timestamp
        )


def test_solver_time_to_timestamp_starting_friday_6am():
    friday_6am_timestamp = 1694152800  # 08.09.2023 06:00 (Friday)

    expected_results = [
        (0, 1694152800, '08.09.2023 06:00 Friday'),
        (480, 1694181600, '08.09.2023 14:00 Friday'),
        (960, 1694210400, '08.09.2023 22:00 Friday'),

        (1440, 1694412000, '11.09.2023 6:00 Monday'),
        (1920, 1694440800, '11.09.2023 14:00 Monday'),

        (7200, 1694757600, '15.09.2023 06:00 Friday'),

        (8640, 1695016800, '17.09.2023 06:00 Monday'),
        (8700, 1695020400, '17.09.2023 07:00 Monday'),
    ]

    for solver_time, expected_timestamp, human_readable_time in expected_results:
        assert expected_timestamp == solver_time_to_timestamp(
            solver_time=solver_time,
            start_timestamp=friday_6am_timestamp
        )


def test_solver_time_to_timestamp_starting_wednesday_2pm():
    wednesday_2pm_timestamp = 1694008800  # 2023-09-06 06:00 (Wednesday)

    expected_results = [
        (0, 1694008800, '06.09.2023 14:00 Wednesday'),
        (480, 1694037600, '06.09.2023 22:00 Wednesday'),

        (2400, 1694152800, '08.09.2023 6:00 Friday'),
        (2880, 1694181600, '08.09.2023 14:00 Friday'),
        (3360, 1694210400, '08.09.2023 22:00 Friday'),

        (3840, 1694412000, '11.09.2023 6:00 Monday'),
        (4320, 1694440800, '11.09.2023 14:00 Monday'),
    ]

    for solver_time, expected_timestamp, human_readable_time in expected_results:
        assert expected_timestamp == solver_time_to_timestamp(
            solver_time=solver_time,
            start_timestamp=wednesday_2pm_timestamp
        )


