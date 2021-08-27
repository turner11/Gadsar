from datetime import datetime
from datetime import time
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import re

reverse_renames = {
    'mi': "מספר אישי",
    'first_name': "שם פרטי",
    'last_name': "שם משפחה",
    'pluga': "פלוגה",
    'team': "מחלקה",
    'phone': "טלפון",
    'hand': "ימני/שמאלי",
    'type': "נשק",
    # 'group': 'קבוצה',
    'group': 'צוות',
    'hour': 'שעה',
}

renames = {v: k for k, v in reverse_renames.items()}
renames.update({k.replace(' ', '_'): v for k, v in renames.items()})
now = datetime.now()


def get_arrived(hour):
    return isinstance(hour, pd._libs.tslibs.timestamps.Timestamp)


def get_time(raw_hour):
    try:
        # if isinstance(raw_hour, datetime):
        if isinstance(raw_hour, time):
            time_stamp = datetime(now.year, now.month, now.day, raw_hour.hour, raw_hour.minute, raw_hour.second)
        elif isinstance(raw_hour, str):
            clean_time = re.sub("[^0-9:]", "", raw_hour)

            time_stamp = datetime.strptime(clean_time, '%I:%M').time()
        elif isinstance(raw_hour, float) and np.isnan(raw_hour):
            time_stamp = None
        else:
            time_stamp = raw_hour

        if time_stamp is not None:
            out_date = datetime(now.year, now.month, now.day, time_stamp.hour, time_stamp.minute, time_stamp.second)
        else:
            out_date = None
    except Exception as ex:
        out_date = None
    return out_date


class DataBundle:
    """"""

    @property
    def df_arrived(self):
        return self.df[self.df.arrived].reset_index(drop=True).copy()

    def __init__(self, excel_path: Union[str, Path]):
        """"""
        super().__init__()
        self.excel_path = Path(excel_path)
        assert self.excel_path.exists()
        self.df = self.get_base_data(excel_path)

    @staticmethod
    def get_base_data(excel_path: Union[str, Path, pd.DataFrame]):
        if isinstance(excel_path, pd.DataFrame):
            df = excel_path.copy()
        else:
            df = pd.read_excel(str(excel_path))
        df = df[[c for c in df.columns if 'unnamed' not in c.lower()]]

        df: pd.DataFrame = df.rename(columns=renames)
        df.pluga = df.pluga.apply(lambda p: str(p).replace('"', ''))
        df.infer_objects()

        # df['arrived'] = df.hour.apply(lambda v: True if v and not np.isnan(v) else False).astype(bool)
        df['hour'] = df.hour.apply(get_time)
        df['arrived'] = df.hour.apply(get_arrived).astype(bool)
        if 'team' not in df.columns:
            df = df.assign(team='?')
        return df.sort_values('hour').reset_index(drop=True).copy()
