from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


renames = {'תאריך': 'date',
           'שם ומספר אישי': 'id',}
drops_prefixes = {'מס', 'האם ברצונך לדווח'}


class ResultsDataBundle(object):

    @property
    def dates(self):
        return self.df.date.unique()

    @property
    def filter_dates(self):
        return list(self._filter_dates)

    @filter_dates.setter
    def filter_dates(self, dates):
        self._filter_dates[:] = list(dates)

    @property
    def df(self):
        df = self._df
        ret = df
        if self._filter_dates:
            ret = df[df.date.isin(self._filter_dates)]
        return ret
    @property
    def stats_cols(self):
        cols = self.get_stats_cols(self.df)
        return cols

    def __init__(self, excel_arg: Union[str, Path, pd.DataFrame]):
        """"""
        super().__init__()
        self._filter_dates = []
        excel_path = ''
        if isinstance(excel_arg, pd.DataFrame):
            df = excel_arg
        else:
            excel_path = Path(excel_arg)
            assert self.excel_path.exists()
            df = self.load_raw_data(excel_path)

        self.excel_path = excel_path
        self._df = df

    @staticmethod
    def get_stats_cols(df):
        excludes = ['average']
        dd = df.copy().fillna(0.).infer_objects()
        cols = sorted([str(c) for c, t in dd.dtypes.items() if t == float and c not in excludes])
        return cols

    @staticmethod
    def load_raw_data(excel_arg):
        # טכניקות ותרגולות בש"
        # פגיעה באויב - זה פנים מבנה
        if isinstance(excel_arg, Path):
            excel_arg = str(excel_arg)

        df = pd.read_excel(excel_arg)#, sheet_name='edited')
        df = df.replace("לא נענה", np.NaN)
        df = df.rename(columns={c: c.strip().replace("'", "").replace('"', '') for c in df.columns})

        df = df.rename(columns=renames)
        df['date'] = pd.to_datetime(df.date)
        for c in df.columns:
            if any(c.startswith(p) for p in drops_prefixes):
                del df[c]
            else:
                try:
                    df[c] = df[c].apply(lambda v: float(v))
                except:
                    pass
        df = ResultsDataBundle._prettify(df)

        drop_cols, col_renames = ResultsDataBundle._get_drop_cols(df)
        for col in drop_cols:
            df = df.drop(col, axis='columns')
        df = df.rename(columns=col_renames)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        counts = defaultdict(lambda: 0)
        for idx in range(len(df.columns)):
            curr_col = df.columns[idx]
            counts[curr_col] += 1
            curr_count = counts[curr_col]
            if curr_count > 1:
                df.columns.values[idx] = f'{curr_col}_{curr_count - 1}'

        df = df.infer_objects()
        df = df.set_index('id')
        data_cols = ResultsDataBundle. get_stats_cols(df)
        for col in data_cols:
            try:
                df[col] = df[col].apply(lambda v: v/1.0 if v <=10 else v /10.0)
            except:
                logger.exception(f'Failed to handle data at {col}')
                del df[col]

        df['average'] = df[data_cols].replace(0, np.nan).mean(numeric_only=True, skipna=True, axis=1)
        # df['average'] = df[data_cols].apply(lambda r: r.dropna().mean(skipna=True), axis='columns')
        return df.copy()

    @staticmethod
    def _prettify(df):
        def is_number(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        s_id = df.id.copy()

        df = df.rename(columns={'id': 'id_old'})
        df['id'] = s_id.apply(lambda s: next((int(v) for v in s.split() if is_number(v)), -1))
        df['name'] = df.apply(lambda row: row.id_old.replace(str(row.id), '').strip(), axis=1)

        df = df.drop('id_old', axis='columns')

        columns = ['id', 'name'] + [c for c in df.columns if c not in ['id', 'name']]
        return df[columns]

    @staticmethod
    def _get_drop_cols(df):
        drop_cols = ['שעה', 'מבצע הסקר', ]

        for col in df.columns:
            if len(df[col].unique()) == 1:
                drop_cols.append(col)

        _renames = {c: c.split('-')[0] for c in df.columns}
        return set(drop_cols), _renames
