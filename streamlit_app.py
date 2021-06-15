import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np
import re

from datetime import datetime
from datetime import time
import altair as alt
# from time import time

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import Spectral6

now = datetime.now()

reverse_renames = {
    'mi': "מספר אישי",
    'first_name': "שם פרטי",
    'last_name': "שם משפחה",
    'pluga': "פלוגה",
    # 'team': "צוות",
    'phone': "טלפון",
    'hand': "ימני/שמאלי",
    'type': "נשק",
    # 'group': 'קבוצה',
    'group': 'צוות',
    'hour': 'שעה',
}

renames= {v:k for k,v in reverse_renames.items()}

def get_arrived(hour):
    return isinstance(hour, pd._libs.tslibs.timestamps.Timestamp)
    try:
        out  = pd.to_datetime(hour)
        # return pd.is
    except Exception as ex:
        return len(str(hour).strip()) > 0


def get_time(raw_hour):



    try:
        # if isinstance(raw_hour, datetime):
        if isinstance(raw_hour, time):
            time_stamp = datetime(now.year, now.month, now.day ,raw_hour.hour, raw_hour.minute, raw_hour.second)
        elif isinstance(raw_hour, str):
            clean_time = re.sub("[^0-9:]", "", raw_hour)

            time_stamp = datetime.strptime(clean_time, '%I:%M').time()
        elif isinstance(raw_hour, float) and np.isnan(raw_hour):
            time_stamp = None
        else:
            time_stamp = raw_hour

        if time_stamp is not None:
            out_date = datetime(now.year, now.month, now.day ,time_stamp.hour, time_stamp.minute, time_stamp.second)
        else:
            out_date = None
    except Exception as ex:
        out_date = None
    return out_date


def get_sidebar_inputs():
    excel_path = st.sidebar.text_input('Excel path', Path('./data.xlsx').resolve())
    excel_path = Path(excel_path)
    assert excel_path.exists()

    df = pd.read_excel(str(excel_path))
    df = df[[c for c in df.columns if 'unnamed' not in c.lower()]]

    df: pd.DataFrame = df.rename(columns=renames)
    df.pluga = df.pluga.apply(lambda p: str(p).replace('"', ''))
    df.infer_objects()

    # df['arrived'] = df.hour.apply(lambda v: True if v and not np.isnan(v) else False).astype(bool)
    df['hour'] = df.hour.apply(get_time)
    df['arrived'] = df.hour.apply(get_arrived).astype(bool)

    plugot = list(df.pluga.drop_duplicates())
    plugot = st.sidebar.multiselect('סינון פלוגות', plugot, plugot)
    if plugot:
        df = df[df.pluga.isin(plugot)]
    return df


# noinspection PyStatementEffect
def main():
    inputs = get_sidebar_inputs()
    df: pd.DataFrame = inputs
    df_arrived = df[df.arrived]

    st.subheader(f'Total arrived: {len(df_arrived)}')

    with st.beta_expander('מידע גולמי', expanded=False):
        df


    with st.beta_expander('כמה הגיעו מכל פלוגה', expanded=False):
        df_arrived_count = df_arrived.groupby('pluga').agg({'mi':'count'}).rename(columns={'mi':'count'}).sort_values('count')
        df_arrived_count
        st.bar_chart(df_arrived_count)

    with st.beta_expander('כמה יש בכל קבוצה', expanded=True):
        df_teams = df_arrived.groupby('group').agg({'mi':'count'}).rename(columns={'mi':'counts'}).sort_values('counts', ascending=False)
        df_teams
        st.bar_chart(df_teams)

    with st.beta_expander('זמני הגעה', expanded=True):
        # dfc = df_arrived.copy().head()
        # col = 'hour'
        #
        # dfc[col] = pd.to_datetime(dfc[col])
        # dfc[col] = [time.time() for time in dfc[col]]
        # dfc

        import matplotlib.pyplot as plt
        pd.plotting.register_matplotlib_converters()


        cols = ['pluga', 'hour', ] + [c for c in df_arrived.columns if 'name' in c]
        dfc = df_arrived[cols]
        dfc = dfc.sort_values('hour').reset_index()
        # color_values = ['g','b','c','y', 'm','k', 'w']
        # colors = dict(zip([p for p in dfc.pluga.drop_duplicates()], color_values))
        # dfc['colors'] = dfc.pluga.apply(lambda p: colors[p])
        dfc['total_arrived'] = pd.Series(range(len(dfc)), name='total_arrived')
        dfc['time_stamp'] = dfc.hour.apply(lambda h: str(h.time()))


        chart = st.empty()
        x = alt.Chart(dfc).mark_circle().encode(
            x='hour',
            y='total_arrived',
            color='pluga',
            tooltip=list(sorted(dfc.columns)),
        ).interactive()
        chart.altair_chart(x, use_container_width=True)

if __name__ == '__main__':
    main()
