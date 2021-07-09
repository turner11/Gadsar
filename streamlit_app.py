import pandas as pd
import streamlit as st
from pathlib import Path
from utils import DataBundle
import altair as alt

def get_sidebar_inputs():
    excel_path = st.sidebar.text_input('Excel path', Path('./data.xlsx').resolve())

    df = DataBundle.get_base_data(excel_path)
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
        df_arrived_count = df_arrived.groupby('pluga').agg({'mi':'count'}).rename(columns={'mi':'count'}).sort_values('count')
        df_arrived_count
        st.bar_chart(df_arrived_count)

    with st.beta_expander('כמה יש בכל קבוצה', expanded=True):
        df_teams = df_arrived.groupby('group').agg({'mi':'count'}).rename(columns={'mi':'counts'}).sort_values('counts', ascending=False)
        df_teams
        st.bar_chart(df_teams)

    with st.beta_expander('זמני הגעה', expanded=True):
        cols = ['pluga', 'hour', ] + [c for c in df_arrived.columns if 'name' in c]
        dfc = df_arrived[cols]
        dfc = dfc.sort_values('hour').reset_index()
        # color_values = ['g','b','c','y', 'm','k', 'w']
        # colors = dict(zip([p for p in dfc.pluga.drop_duplicates()], color_values))
        # dfc['colors'] = dfc.pluga.apply(lambda p: colors[p])
        dfc['total_arrived'] = pd.Series(range(len(dfc)), name='total_arrived') +1
        dfc['time_stamp'] = dfc.hour.apply(lambda h: str(h.time()))


        chart = st.empty()
        x = alt.Chart(dfc).mark_circle().encode(
            x='hour',
            y='total_arrived',
            color='pluga',
            tooltip=list(sorted(dfc.columns)),
        ).interactive()
        chart.altair_chart(x, use_container_width=True)


    # with st.beta_expander('זמני הגעה לפי מחלקות', expanded=True):
    if True:
        from bokeh_script import get_bokeh_figure
        lay = get_bokeh_figure(r'C:\Users\avitu\Documents\GitHub\IMI\data.xlsx')
        st.bokeh_chart(lay)
        1


if __name__ == '__main__':
    main()
