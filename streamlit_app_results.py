import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import altair as alt
from results_utils import ResultsDataBundle

DEBUG = False


@st.cache
def get_excel(excel_path):
    return ResultsDataBundle.load_raw_data(excel_path)


def get_sidebar_inputs():
    if DEBUG:
        excel_path = st.sidebar.text_input('Excel path', Path('./imi_results.xlsx').resolve())
        excel_arg = excel_path
    else:
        uploaded_file = st.sidebar.file_uploader("Choose the IMI results file", type="xlsx")
        excel_arg = uploaded_file
    if not excel_arg:
        return


    df = get_excel(excel_arg)
    group_filter = ''
    if 'group' in df.columns:
        groups = ['']+ [g for g in df['group'].dropna().drop_duplicates() if g.strip()]
        groups = sorted(groups)
        group_filter = st.sidebar.selectbox('פילטר', groups, groups.index(''))
    if group_filter:
        df = df[df['group']==group_filter]

    bundle = ResultsDataBundle(df)
    filter_dates = list(bundle.dates)

    if not group_filter:
        # HACK for gadsar...
        idx = 2 if len(filter_dates) >2 else 0
        date = st.sidebar.selectbox('dates', filter_dates, idx)
        if date:
            bundle.filter_dates = [date]
    return bundle


def main():
    pd.plotting.register_matplotlib_converters()
    bundle = get_sidebar_inputs()
    if not bundle:
        st.warning("No Data")
        return

    with st.beta_expander('מידע גולמי', expanded=False):
        df = bundle.df
        st.dataframe(df)

    with st.beta_expander('חיפוש לפי חייל', expanded=False):
        records = show_searchable_item(df)

    highlighted_records = (records.name,) if isinstance(records, pd.Series) else tuple(records.index)

    show_stats(df, bundle.stats_cols, highlighted_records=highlighted_records)

    with st.beta_expander('סיכום', expanded=False):
        show_total(df, bundle.stats_cols)


def show_stats(df, stats_cols, highlighted_records=tuple()):
    stats_cols = sorted(set(stats_cols).intersection(set(df.columns)))
    if not stats_cols:
        st.warning('לא נמצאו נתונים לסטטיסטיקה')
        return

    default_stat = stats_cols[0]
    affective_cols = st.multiselect("אימון:", stats_cols, default_stat)
    for col in affective_cols:
        get_stats_for_col(df, col, highlighted_records)


def show_total(df, stats_cols):
    avg = df[stats_cols].replace(0, np.nan).mean(numeric_only=True, skipna=True, axis=0)
    st.subheader('ממוצע לפי נושא:')
    st.dataframe(avg)
    st.subheader('סה"כ:')
    total_avg = avg.mean()
    s = pd.Series({'מתאמנים':len(df), 'ממוצע':total_avg})
    st.dataframe(s)




def get_stats_for_col(df, col, highlighted_records=tuple()):

    df = df.copy()
    selected = df.index.isin(highlighted_records)

    df_highlight = df.iloc[selected, :].copy()
    dfs = [(df, 75, 'average'), (df_highlight, 50, alt.value('red'))]
    charts = []
    for tpl in dfs:
        curr_df, size, color = tpl

        x = alt.Chart(curr_df).mark_circle(size=size).encode(
            x=col,
            y='average',
            # color=color,
            color=color,
            tooltip=[c for c, t in df.dtypes.items() if t == object or c == 'average'],  # 'description',
        ).interactive()
        charts.append(x)

    result = charts[0] + charts[1]

    chart = st.empty()
    chart = chart.altair_chart(result, use_container_width=True)
    return chart


# noinspection PyBroadException
def show_searchable_item(df):
    name = st.text_input('חייל לחיפוש:', help="Could be name or ID")
    if name:
        try:
            records = df.loc[[int(name)], :]
        except Exception:
            name_match = df.name.str.contains(f'.*{name}.*', regex=True)
            records = df.loc[name_match]

        if len(records) == 0:
            st.warning(f'No match found for "{name}"')
        elif len(records) == 1:
            records = records.iloc[0]
            records = records.dropna()
        else:
            records = records.dropna(how='all', axis='columns')

        st.dataframe(records)
    else:
        records = df[[False] * len(df)]
    return records


if __name__ == '__main__':
    main()

