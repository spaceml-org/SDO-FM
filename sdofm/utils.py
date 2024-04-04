

def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return "{0}d:{1:02}:{2:02}:{3:02}".format(int(d), int(h), int(m), int(s))