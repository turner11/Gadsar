import os
import pandas as pd
import streamlit as st
from pathlib import Path

from results_utils import ResultsDataBundle


@st.cache
def get_excel(excel_path):
    return ResultsDataBundle.load_raw_data(excel_path)


def get_sidebar_inputs():
    debug = os.environ.get('debug')
    if debug:
        excel_path = st.sidebar.text_input('Excel path', Path('./imi_results.xlsx').resolve())
        excel_arg = excel_path
    else:
        uploaded_file = st.sidebar.file_uploader("Choose the IMI results file", type="xlsx")
        excel_arg = uploaded_file
    if not excel_arg:
        return

    df = get_excel(excel_arg)
    group_filter = ''
    # if 'group' in df.columns:
    #     groups = [''] + [g for g in df['group'].dropna().drop_duplicates() if g.strip()]
    #     groups = sorted(groups)
    #     group_filter = st.sidebar.selectbox('פילטר', groups, groups.index(''))
    # if group_filter:
    #     df = df[df['group'] == group_filter]

    bundle = ResultsDataBundle(df)
    filter_dates = list(bundle.dates)

    if not group_filter:
        # HACK for gadsar...
        idx = 2 if len(filter_dates) > 2 else 0
        date = st.sidebar.selectbox('dates', filter_dates, idx)
        if date:
            bundle.filter_dates = [date]
    return bundle


def show_single_file_info(info, bundle):
    info = info.dropna()
    df, stats_cols = bundle.df, bundle.stats_cols
    stats_cols = set(stats_cols + ['average']).intersection(set(info.keys()))
    meta_cols = [c for c in info.keys() if c not in stats_cols]
    stats = info[stats_cols]
    stats.name = ''
    meta = info[meta_cols]

    col_title, col_desc = st.beta_columns([1, 1])

    for k, val in meta.items():
        v = str(val).strip()
        if not v:
            continue
        col_title.text(k)
        col_desc.text(v)

    # st.dataframe(stats)
    percentiles = {}
    for col, value in stats.items():
        if col in df.columns:
            general_values = df[col].dropna()
        else:
            general_values = []
        if not len(general_values):
            percentile = None
        else:
            percentile = sum(general_values <= value) / len(general_values) * 100.0

        percentiles[col] = percentile

    s_percentiles = pd.Series(percentiles, name='אחוזון')
    stats.name = 'אימון'
    dt = pd.DataFrame([stats, s_percentiles]).T
    st.dataframe(dt)


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
        name = st.text_input('חייל לחיפוש:', help="Could be search_key or ID")
        records = bundle.get_items(name)

        if len(name) and not len(records):
            st.warning(f'No match found for "{name}"')
        elif isinstance(records, pd.DataFrame):
            st.dataframe(records)
        elif isinstance(records, pd.Series):
            show_single_file_info(records, bundle)

    with st.beta_expander('פירוט אימונים', expanded=False):
        highlighted_records = (records.name,) if isinstance(records, pd.Series) else tuple(records.index)
        stats_cols = sorted(set(bundle.stats_cols).intersection(set(df.columns)))
        default_stat = stats_cols[0]
        affective_cols = st.multiselect("אימון:", stats_cols, default_stat)

        charts = bundle.get_charts(affective_cols, highlighted_records=highlighted_records)
        if len(affective_cols) and not len(charts):
            st.warning('לא נמצאו נתונים לסטטיסטיקה')
        else:

            for key, alt_chart in charts.items():
                chart = st.empty()
                chart.altair_chart(alt_chart, use_container_width=True)

    comparison_chart = bundle.get_comparison_chart()
    if comparison_chart is not None:
        with st.beta_expander('השוואה בין מסגרות', expanded=True):
            chart = st.empty()
            chart.altair_chart(comparison_chart, use_container_width=True)

    with st.beta_expander('סיכום', expanded=False):
        df_avg, s_total = bundle.get_total_series()
        st.subheader('ממוצע לפי נושא:')
        st.dataframe(df_avg)
        st.subheader('סה"כ:')

        s = pd.Series({'מתאמנים': len(df), 'ממוצע': s_total})
        st.dataframe(s)


if __name__ == '__main__':
    main()
