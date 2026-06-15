import datetime


def get_datetime() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d %Y, %I:%M %p").lstrip("0").replace(" 0", " ")
