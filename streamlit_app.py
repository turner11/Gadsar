from functools import reduce

import pandas as pd
import streamlit as st
from pathlib import Path
from utils import DataBundle
import altair as alt
import gsheetsdb

# Create a connection object.
conn = gsheetsdb.connect()


@st.cache(ttl=10)
def run_query(query) -> pd.DataFrame:
    cursor = conn.execute(query, headers=1)
    df = pd.DataFrame(cursor.fetchall())
    return df.copy()


def get_df():
    sheet_url = st.secrets.get("public_gsheets_url", '')
    query_params = st.experimental_get_query_params() or dict()
    sheet_url = query_params.get('url', sheet_url)

    path = sheet_url or Path('./data.xlsx').resolve()
    excel_path = st.sidebar.text_input('Excel path / URL', str(path))

    if excel_path.lower().startswith('http'):
        path_arg = run_query(f'SELECT * FROM "{sheet_url}"')
    else:
        path_arg = path
    df = DataBundle.get_base_data(path_arg)
    return df


def get_sidebar_inputs():
    df = get_df()

    plugot = list(df.pluga.drop_duplicates())
    plugot = st.sidebar.multiselect('סינון פלוגות', plugot, plugot)
    if plugot:
        df = df[df.pluga.isin(plugot)]
    return df


# noinspection PyStatementEffect
def main():
    pd.plotting.register_matplotlib_converters()
    inputs = get_sidebar_inputs()
    df: pd.DataFrame = inputs
    df_arrived = df[df.arrived]

    st.subheader(f'Total arrived: {len(df_arrived)}')

    with st.beta_expander('מידע גולמי', expanded=False):
        df

    with st.beta_expander('כמה הגיעו מכל פלוגה', expanded=False):
        df_arrived_count = df_arrived.groupby('pluga').agg({'mi': 'count'}).rename(columns={'mi': 'count'}).sort_values(
            'count')
        df_arrived_count
        st.bar_chart(df_arrived_count)

    with st.beta_expander('כמה יש בכל קבוצה', expanded=True):
        df_teams = df_arrived.groupby('group').agg({'mi': 'count'}).rename(columns={'mi': 'counts'}).sort_values(
            'counts', ascending=False)
        df_teams
        st.bar_chart(df_teams)

    with st.beta_expander('זמני הגעה', expanded=True):
        cols = ['pluga', 'hour', 'team'] + [c for c in df_arrived.columns if 'name' in c]
        dfc = df_arrived[cols]
        dfc = dfc.sort_values('hour').reset_index()
        dfc['total_arrived'] = pd.Series(range(len(dfc)), name='total_arrived') + 1
        dfc['time_stamp'] = dfc.hour.apply(lambda h: str(h.time()))

        plugas = list(dfc.pluga.unique())
        group_by = 'pluga' if len(plugas) != 1 else 'team'
        dfc[group_by] = dfc[group_by].apply(lambda v: str(v))  # Make sure its categorical in legend
        selection = alt.selection_multi(fields=[group_by], bind='legend')
        charts = []
        for group_name, dfp in dfc.groupby(group_by):
            # size=50
            chart_p = alt.Chart(dfp.reset_index()).mark_circle().encode(
                x='hour',
                y='total_arrived',
                color=group_by,
                tooltip=[c for c in cols if c in dfp.columns],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
            )
            charts.append(chart_p)
        if charts:
            chart = reduce(lambda c1, c2: c1 + c2, charts)
            chart = chart.add_selection(selection).interactive()
            st_chart = st.empty()
            st_chart.altair_chart(chart, use_container_width=True)



if __name__ == '__main__':
    main()
