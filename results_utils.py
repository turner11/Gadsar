from functools import reduce
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from collections import defaultdict
import altair as alt
import logging

logger = logging.getLogger(__name__)

renames = {'תאריך': 'date',
           'שם ומספר אישי': 'id', }
drops_prefixes = {'מס', 'האם ברצונך לדווח', 'MA'}


# noinspection PyBroadException
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

    # noinspection SpellCheckingInspection, PyBroadException
    @staticmethod
    def get_stats_cols(df):
        excludes = ['average', 'group', 'pluga', 'mahlaka']
        dd = df.copy().fillna(0.).infer_objects()
        cols = sorted([str(c) for c, t in dd.dtypes.items() if t == float and c not in excludes])
        return cols

    @staticmethod
    def load_raw_data(excel_arg):
        if isinstance(excel_arg, Path):
            excel_arg = str(excel_arg)

        df = pd.read_excel(excel_arg)
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
        data_cols = ResultsDataBundle.get_stats_cols(df)
        for col in data_cols:
            try:
                df[col] = df[col].apply(lambda v: v / 1.0 if v <= 10 else v / 10.0)
            except:
                logger.exception(f'Failed to handle data at {col}')
                del df[col]

        df['average'] = df[data_cols].replace(0, np.nan).mean(numeric_only=True, skipna=True, axis=1)
        for col in ['pluga', 'mahlaka', 'group']:
            if col not in df.columns:
                df = df.assign(**{col:''})
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

    # noinspection PyBroadException
    def get_items(self, search_key):
        df = self.df
        records = df[[False] * len(df)]
        if search_key:
            try:
                records = df.loc[[int(search_key)], :]
            except Exception:
                name_match = df.name.str.contains(f'.*{search_key}.*', regex=True)
                records = df.loc[name_match]

            if len(records) == 1:
                records = records.iloc[0]
                records = records.dropna()
            elif len(records) > 1:
                records = records.dropna(how='all', axis='columns')

        return records

    def get_charts(self, affective_cols, highlighted_records=tuple()):
        df = self.df
        stats_cols = [c for c in affective_cols if c in df.columns]

        charts = {}
        for col in stats_cols:
            chart = self.__get_chart_for_col(df, col, highlighted_records)
            charts[col] = chart
        return charts

    @staticmethod
    def __get_chart_for_col(df, col, highlighted_records=tuple()):
        df = df.copy()
        selected = df.index.isin(highlighted_records)

        df_highlight = df.iloc[selected, :].copy()
        dfs = [(df, 75, None), (df_highlight, 150, alt.value('purple'))]
        charts = []
        selection = alt.selection_multi(fields=['pluga'], bind='legend')
        for tpl in dfs:
            curr_df, size, color = tpl
            if not len(curr_df):
                continue

            x = alt.Chart(curr_df).mark_circle(size=size).encode(
                x=alt.X(f'{col}:Q',
                        scale=alt.Scale(zero=False)),
                y='average',
                # color=color,
                color=color if color is not None else 'pluga',
                tooltip=[c for c, t in df.dtypes.items() if t == object or c == 'average'],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
            ).interactive()
            charts.append(x)

        result = reduce(lambda c1, c2: c1 + c2, charts)
        result = result.add_selection(selection)
        return result

    def get_total_series(self):
        df = self.df
        stats_cols = self.stats_cols
        return self.__get_total_series(df, stats_cols)

    @staticmethod
    def __get_total_series(df, stats_cols):
        df_avg = df[stats_cols].replace(0, np.nan).mean(numeric_only=True, skipna=True, axis=0)
        s_total = df_avg.mean()
        return df_avg, s_total

    def get_comparison_chart(self):
        df = self.df
        stats_cols = self.stats_cols

        required_fields = ['pluga', 'mahlaka']
        out_doors_cols = [c for c in stats_cols if 'טכניקות ותרגולות' in c]
        in_doors_cols = [c for c in stats_cols if 'פגיעה באויב' in c]

        has_groups = all(c in df.columns for c in required_fields)
        has_data = all(len(arr) for arr in [out_doors_cols, in_doors_cols])

        if not has_groups or not has_data:
            return None

        df = df[(~df.pluga.isna()) & (~df.mahlaka.isna())]

        if not len(df) or len(set(df.pluga)) <= 1:
            return None

        df_data = df[required_fields + out_doors_cols + in_doors_cols].copy()
        df_data['indoor'] = \
            df_data[in_doors_cols].mean(skipna=True).mean()

        in_doors = pd.concat([df_data[required_fields + [c]].rename(columns={c: 'indoor'}) for c in in_doors_cols])
        out_doors = pd.concat([df_data[required_fields + [c]].rename(columns={c: 'outdoor'}) for c in out_doors_cols])

        df_plot = in_doors.reset_index().merge(out_doors.reset_index()).set_index('id')

        dfps = df_plot.groupby('pluga', as_index=True)[['indoor', 'outdoor']].agg(np.mean)
        df_plot['mahlaka'] = df_plot.pluga.apply(str) + '/' + df_plot.mahlaka.apply(str)
        dfms = df_plot.groupby(['mahlaka', 'pluga'], as_index=True)[['indoor', 'outdoor']].agg(np.mean)

        selection = alt.selection_multi(fields=['pluga'], bind='legend')
        charts = []
        for pluga, dfp in dfms.groupby('pluga'):
            chart_p = alt.Chart(dfp.reset_index()).mark_square(size=50).encode(
                x=alt.X('indoor:Q',
                        scale=alt.Scale(zero=False)),
                y='outdoor',
                color='pluga',
                tooltip=['mahlaka', 'indoor', 'outdoor'],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
            )

            df_center = dfps[dfps.index == pluga].reset_index()
            centroid = alt.Chart(df_center).mark_circle(size=350).encode(
                x=alt.X('indoor:Q',
                        scale=alt.Scale(zero=False)),
                y='outdoor',
                color='pluga',
                tooltip=['pluga', 'indoor', 'outdoor'],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
            )

            chart_p = chart_p + centroid
            charts.append(chart_p)

        chart = None
        if charts:
            chart = reduce(lambda c1, c2: c1 + c2, charts)
            chart = chart.add_selection(selection).interactive()

        return chart
