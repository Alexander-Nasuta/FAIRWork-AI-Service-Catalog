from utils.logger import log
from datetime import datetime, timedelta
import pytz

# timezone = pytz.timezone('Europe/Bucharest')  # specifies the correct timezone
timezone = pytz.FixedOffset(0)


def is_between_saturday_and_monday(timestamp: int | float) -> bool:
    """
    Check if the given timestamp is between Saturday 6:00 AM and Monday 6:00 AM.
    """
    current_time = datetime.fromtimestamp(timestamp, timezone)
    weekday = current_time.weekday()
    hour = current_time.hour

    # Check if it's Saturday after 6:00 AM or Sunday
    if (weekday == 5 and hour >= 6) or (weekday == 6):
        return True

    # Check if it's Monday before 6:00 AM
    if weekday == 0 and hour < 6:
        return True

    return False


def next_monday_6am(timestamp_to_convert: int | float) -> int:
    """
    Calculate the timestamp of the next Monday at 6:00 AM for a given timezone.
    """
    current_time = datetime.fromtimestamp(timestamp_to_convert, timezone)

    # Calculate the number of days until the next Monday
    days_until_monday = (0 - current_time.weekday() + 7) % 7
    if days_until_monday == 0:
        days_until_monday = 7

    # Calculate the next Monday at 6:00 AM
    next_monday = current_time + timedelta(days=days_until_monday)
    next_monday_6am = next_monday.replace(hour=6, minute=0, second=0, microsecond=0)

    # Convert to timestamp
    return int(next_monday_6am.timestamp())


def next_saturday_6am(timestamp_to_convert: int | float) -> int:
    """
    Calculate the timestamp of the next Saturday at 6:00 AM for a given timezone.
    """
    current_time = datetime.fromtimestamp(timestamp_to_convert, timezone)

    # Calculate the number of days until the next Saturday
    days_until_saturday = (5 - current_time.weekday() + 7) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7

    # Calculate the next Saturday at 6:00 AM
    next_saturday = current_time + timedelta(days=days_until_saturday)
    next_saturday_6am = next_saturday.replace(hour=6, minute=0, second=0, microsecond=0)

    # Convert to timestamp
    return int(next_saturday_6am.timestamp())


def log_time(timestamp: int | float, extra_text: str | None = '') -> None:
    """
    Log the timestamp in human readable format.
    """
    human_readable_time = datetime.fromtimestamp(timestamp, timezone)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    log.debug(
        f"{str(extra_text) + ' ' if extra_text else ''}Timestamp: {timestamp}, Human readable: {human_readable_time} ({days[human_readable_time.weekday()]})")


def timestamp_to_solver_time(timestamp_to_convert: int | float, start_timestamp: int | float) -> int:
    """
    Convert timestamps to solver time.
    The solver time is the time elapsed since the start timestamp in minutes.
    It excludes the time between Saturday 6am and Monday 6am.
    """
    # log_time(timestamp_to_convert, "'Timestamp to convert'")
    # log_time(start_timestamp, "'Start timestamp'")

    if timestamp_to_convert < start_timestamp:
        log.error("The timestamp to convert is before the start timestamp.",
                  extra={"timestamp_to_convert": timestamp_to_convert, "start_timestamp": start_timestamp})
        raise ValueError("The timestamp to convert is before the start timestamp.")
    if timestamp_to_convert == start_timestamp:
        return 0

    # Check if the start timestamp is between Saturday and Monday
    # If it's between Saturday 6am and Monday 6am, set the start timestamp to the next Monday 6am
    if is_between_saturday_and_monday(start_timestamp):
        log.debug("The start timestamp is between Saturday and Monday", extra={"start_timestamp": timestamp_to_convert})
        # raise ValueError("The start timestamp is between Saturday and Monday.")
        start_timestamp = next_monday_6am(start_timestamp)
    else:
        log.debug("The start timestamp is valid", extra={"start_timestamp": timestamp_to_convert})

    # check if timestamp_to_convert is between Saturday (6am) and Monday (6am)
    if is_between_saturday_and_monday(timestamp_to_convert):
        log.debug("timestamp_to_convert is between Saturday and Monday. Setting it to next monday",
                 extra={"timestamp_to_convert": timestamp_to_convert})
        timestamp_to_convert = next_monday_6am(timestamp_to_convert)
    else:
        log.debug("timestamp_to_convert is valid", extra={"timestamp_to_convert": timestamp_to_convert})

    # check if the end timestamp is before end of this week (Staurday 6am)
    if timestamp_to_convert < next_saturday_6am(start_timestamp):
        # the end timestamp is before the end of the week
        elapsed_time_in_seconds = timestamp_to_convert - start_timestamp
        elapsed_time_in_minutes = int(elapsed_time_in_seconds / 60)
        return elapsed_time_in_minutes

    else:
        # the end timestamp is after the end of the week
        elapsed_time_this_week_in_seconds = next_saturday_6am(start_timestamp) - start_timestamp
        elapsed_time_this_week_in_minutes = int(elapsed_time_this_week_in_seconds / 60)

        # recursively call the function to calculate the time for the next week
        # this will terminate when the end timestamp is before the end of the week in the case above
        log.debug(
            f"The end timestamp is after the end of the week. Elapsed time this week: {elapsed_time_this_week_in_minutes}",
            extra={"timestamp_to_convert": timestamp_to_convert, "start_timestamp": start_timestamp})

        return elapsed_time_this_week_in_minutes + timestamp_to_solver_time(timestamp_to_convert,
                                                                            next_monday_6am(start_timestamp))


def solver_time_to_timestamp(solver_time: int | float, start_timestamp: int | float) -> int:
    """
    Convert solver time to timestamp.
    """
    log.debug(f"converting solver_time={solver_time} to timestamp with start_timestamp: {start_timestamp}")
    # if solver time is negative, raise an error
    if solver_time < 0:
        log.error("The solver time is negative.",
                  extra={"solver_time": solver_time, "start_timestamp": start_timestamp})
        raise ValueError("The solver time is negative.")

    # check if the start timestamp is between Saturday and Monday
    # if it's between Saturday 6am and Monday 6am, set the start timestamp to the next Monday 6am
    if is_between_saturday_and_monday(start_timestamp):
        log.debug("The start timestamp is between Saturday and Monday", extra={"start_timestamp": start_timestamp})
        start_timestamp = next_monday_6am(start_timestamp)

    # if solver time is 0, return the start timestamp
    if solver_time == 0:
        return start_timestamp

    end_timestamp_of_this_week = next_saturday_6am(start_timestamp)

    time_till_end_of_this_week_in_seconds = end_timestamp_of_this_week - start_timestamp
    time_till_end_of_this_week_in_minutes = int(time_till_end_of_this_week_in_seconds / 60)

    if solver_time < time_till_end_of_this_week_in_minutes:
        log.debug(f"the solver-time is within this week")
        solver_time_in_seconds = solver_time * 60
        return start_timestamp + solver_time_in_seconds
    else:
        # log_time(next_monday_6am(start_timestamp), extra_text="asdf ")
        log.debug(f"time_this_week: {time_till_end_of_this_week_in_minutes}, remaining_time: {solver_time-time_till_end_of_this_week_in_seconds}")
        return solver_time_to_timestamp(
            solver_time=solver_time - time_till_end_of_this_week_in_minutes,
            start_timestamp=next_monday_6am(start_timestamp)
        )


if __name__ == '__main__':
    monday_6am_timestamp = 1693807200  # 2023-09-04 06:00 (Monday)
    wednesday_2pm_timestamp = 1694008800  # 2023-09-06 06:00 (Wednesday)
    friday_6am_timestamp = 1694152800  # 2023-09-08 06:00 (Friday)

    # sunday_timestamp = 1694325600  # 2023-09-09 06:00 (Sunday)
    # log_time(start_timestamp, "Start timestamp")
    # log_time(sunday_timestamp)

    # next_monday = next_monday_6am(monday_6am_timestamp)
    # next_saturday = next_saturday_6am(monday_6am_timestamp)

    # log_time(next_monday, "Next Monday")
    # log_time(next_saturday, "Next Saturday")

    # my_timestamp = 1694440800
    # log_time(my_timestamp, "'My timestamp':")
    # elapsed_time = timestamp_to_solver_time(timestamp_to_convert=my_timestamp, start_timestamp=wednesday_2pm_timestamp)
    # log.info(f"Elapsed time in minutes: {elapsed_time}")

    res_timestamp = solver_time_to_timestamp(7199, start_timestamp=monday_6am_timestamp)
    log_time(res_timestamp, "Result timestamp:")
