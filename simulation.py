import numpy as np
import pandas as pd


def simulate_nowait(n):
    """
    Simulate arrival and departure process with no waiting.

    Checkout begins immediately as if there are unlimited registers available.
    Inter-arrival durations are iid exponential draws with rate parameter fixed at 1.
    Checkout durations are iid beta(2,5) draws scaled to have the same mean duration as arrivals.
    """

    arrival_time = pd.DataFrame(np.random.exponential(size=n).cumsum())
    departure_time = arrival_time.copy()

    departure_time[0] = departure_time[0] + np.random.beta(2, 5, size=n)*3.5

    arrival_time['change_in_queue'], departure_time['change_in_queue'] = 1, -1

    timeline = pd.concat([arrival_time, departure_time], ignore_index=True)
    timeline.set_index(0, inplace=True)
    timeline.sort_index(0, inplace=True)
    timeline['queue_size'] = timeline.change_in_queue.cumsum()

    return timeline


def simulate_queue(n):
    """
    Simulate single-server arrival and departure process.

    Inter-arrival durations are iid exponential draws with rate parameter fixed at 1.
    Checkout durations are iid beta(2,5) draws scaled to have the same mean duration as arrivals.
    """

    arrival = pd.DataFrame(np.random.exponential(size=n).cumsum(), columns=['arrival_time'])
    arrival['process_duration'] = np.random.beta(2, 5, size=n)*3.5

    # Initial condition: empty queue
    arrival = arrival.set_value(0, 'departure_time',
                                arrival.loc[0, 'arrival_time'] + arrival.loc[0, 'process_duration'])

    for i in range(1, n):
        """
        Starting with the 2nd job in the queue:
        If the nth job arrives before the (n-1)th job is processed, the nth job starts processing only
        after the (n-1)th job is finished.
        But if the nth job arives after the (n-1)th job is processed, then there is no wait.
        """
        if arrival.loc[i-1, 'departure_time'] > arrival.loc[i, 'arrival_time']:
            arrival.set_value(i, 'departure_time',
                              arrival.loc[i-1, 'departure_time'] + arrival.loc[i, 'process_duration'])
        else:
            arrival.set_value(i, 'departure_time',
                              arrival.loc[i, 'arrival_time'] + arrival.loc[i, 'process_duration'])

    arrival['change_in_queue'] = 1
    arrival.rename(columns={'arrival_time': 'timeline'}, inplace=True)

    departure = pd.DataFrame()
    departure['timeline'] = arrival.departure_time
    departure['change_in_queue'] = -1

    master_timeline = pd.concat([arrival, departure], join='inner', ignore_index=True)
    master_timeline.set_index('timeline', inplace=True)
    master_timeline.sort_index(0, inplace=True)
    master_timeline['queue_size'] = master_timeline.change_in_queue.cumsum()

    return master_timeline
