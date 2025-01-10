merged = pd.concat([timeseries_lab, timeseries_resp], sort=False)
merged = pd.concat([merged, timeseries_periodic], sort=False)
merged = pd.concat([merged, timeseries_aperiodic], sort=True)
