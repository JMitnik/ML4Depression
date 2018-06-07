def get_weekend_days(dt_series):
    weekend_days = dt_series.apply(lambda x: (
        x.isoweekday() == 6 or x.isoweekday() == 7)).astype(int)

    return weekend_days
