from typing import Union

import numpy as np
import pandas as pd
import torch

from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import check_y_survival

def check_ipcw_calc(y_train_set, y_val_set):
    """ Check that IPCW is calculated. Will throw error if not. 
    
    Args:
        y_train_set: train set ordered array or tuple of (event, time)
        y_val_set: val set, s.o."""

    y_train = (y_train_set[0].to("cpu").numpy(), y_train_set[1].to("cpu").numpy())
    y_train = e_t_to_tuple(y_train[0], y_train[1])

    y_val = (y_val_set[0].to("cpu").numpy(), y_val_set[1].to("cpu").numpy())
    y_val = e_t_to_tuple(y_val[0], y_val[1])

    cde = CensoringDistributionEstimator()
    event, time = check_y_survival(y_train)
    event_val, time_val = check_y_survival(y_val)
    cde.fit(y_train)
    print("max time train", max(time_val))
    kaplan_probas = cde.predict_proba(time_val[event_val])
    cde.predict_ipcw(y_val)

    return True

def e_t_to_tuple(
    e,
    t,
    time_type=float,
    event_col_name="event",
    time_col_name="time",
    order=["event", "time"],
):
    """Convert event and time to structured array.

    Args:
        e (np.array): Array of events
        t (np.array): Array of times
        time_type (type): Type of time. Can be int or float
        event_col_name (str): Name of event column. Defaults to "event"
        time_col_name (str): Name of time column. Defaults to "time"
        order (list): Wanted order of event and time in tuple. Defaults to ["event", "time"]

    Returns:
        np.array: Structured array of events and times in the specified order
    """

    assert len(e) == len(t), "e and t must have the same length"
    assert order == ["event", "time"] or order == [
        "time",
        "event",
    ], "order must be ['event', 'time'] or ['time', 'event']"
    if order == ["time", "event"]:
        t_ = t.copy()
        t = e.copy()
        e = t_.copy()
    if time_type == int:
        return np.array(
            [(e[i], int(t[i])) for i in range(len(e))],
            dtype=[(event_col_name, bool), (time_col_name, int)],
        )
    return np.array(
        [(e[i], t[i]) for i in range(len(e))],
        dtype=[(event_col_name, bool), (time_col_name, float)],
    )

def et_tuple_to_df(et_tuple, event_col_name="event", time_col_name="time"):
    """Convert structured array to dataframe.
    Uses column names "event" and "time" by default.
    """

    return pd.DataFrame(et_tuple, columns=[event_col_name, time_col_name])


def get_event_indicator_matrix(events: Union[torch.Tensor, np.ndarray], durations: Union[torch.Tensor, np.ndarray]):
    """Matrix with 1s at all visits >= duration to event. Times are columns, patients are rows.
    Column index corresponds to time in the unit given by the model's durations/times array.
    
    Example: times = [0, 1, 2, 3, 4, 5]
                labels are then found through: calc_event_indicator_matrix()[:, times] 
    """

    m = torch if isinstance(events, torch.Tensor) else np

    visits = m.arange(0, durations.max()+1)
    y_ = m.zeros((len(events), len(visits)))

    for i in range(len(events)):
        for v in visits:
            if v >= durations[i]:
                if events[i] == 1.0:
                    y_[i, v] = 1.0
    
    return y_
