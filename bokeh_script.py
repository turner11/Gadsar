from bokeh.models import ColumnDataSource, Panel, Tabs, DataTable, TableColumn
from bokeh.layouts import layout, row, column
from bokeh.palettes import Category10 # Viridis256, Category20_20, Spectral6, Spectral4, Spectral11,
from bokeh.plotting import figure, output_file, show, save
import pandas as pd


from utils import DataBundle


def main():
    excel_path = r'C:\Users\avitu\Documents\GitHub\IMI\data.xlsx'
    lay = get_bokeh_figure(excel_path)
    output_file("arrival.html")
    # save(out_element)
    show(lay)


def get_bokeh_figure(excel_path):
    bundle = DataBundle(excel_path)
    df = bundle.df_arrived
    # df = df.dropna(subset=['team', 'pluga'])
    df.loc[:, ['team', 'pluga']].fillna('?', inplace=True)
    df.loc[:, 'full_team_name'] = df.apply(lambda row: f'{row.pluga}/{row.team}', axis=1)
    by_teams = {pluga: get_cumulative_arrival_figure(df_group, 'full_team_name') for pluga, df_group in df.groupby('pluga')}
    group_bys = ['pluga']
    glyphs = {'גדודי': get_cumulative_arrival_figure(df, group_by) for group_by in group_bys}
    glyphs.update(by_teams)
    tab_list = []
    for key, gl in glyphs.items():
        lay = layout([[gl]], sizing_mode='stretch_both')
        tab = Panel(child=lay, title=f"{key}")
        tab_list.append(tab)
    tabs = Tabs(tabs=tab_list, sizing_mode='stretch_both')
    lay = row(column(children=[tabs], sizing_mode='stretch_both'), sizing_mode='stretch_both')
    return lay


def get_cumulative_arrival_figure(df, group_by):
    # from bokeh.transform import factor_cmap
    df = df.sort_values('hour').reset_index(drop=True).copy()
    df.loc[:, 'total_arrived'] = pd.Series(range(len(df)), name='total_arrived') + 1
    df['time'] = df.hour.apply(lambda x: x.strftime("%H:%M:%S"))
    groups = tuple(sorted(set(df[group_by])))
    TOOLS = "hover,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset, save,crosshair, tap,"
    # TOOLS =+ "box_select,poly_select,lasso_select,"

    TOOLTIPS = [
        ("שם", "@first_name @last_name"),
        ("", "@pluga"),
        ("", "@team"),
        ("", "@time"),
        ("", "# @total_arrived"),
    ]

    p = figure(tools=TOOLS, x_axis_type='datetime', tooltips=TOOLTIPS,
               active_drag="box_zoom", active_scroll='wheel_zoom')
    # fill_color = factor_cmap(group_by, palette=Viridis256, factors=groups)
    pallete = Category10[len(groups)]
    for group, color in zip(groups, pallete): #Spectral6 Viridis256
        data = df[df[group_by] == group]
        source = ColumnDataSource(data)
        # p.line(df['date'], df['close'], line_width=2, color=color, alpha=0.8, legend_label=name)
        p.scatter(source=source, x='hour', y='total_arrived',
                  # fill_color=fill_color,
                  fill_color=color,
                  fill_alpha=0.6,
                  line_color=None,
                  legend_label=group)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    summary = \
        df.groupby(group_by).agg({group_by:'count'}).rename(columns={group_by:'count'})\
            .sort_values('count', ascending=False).reset_index()
    source = ColumnDataSource(summary)

    columns = [
        TableColumn(field="count", title="סה''כ",),
        TableColumn(field=group_by, title="שם"),
    ]
    data_table = DataTable(source=source, columns=columns, width=400, height=280)
    # lay = row(column(children=[tabs], sizing_mode='stretch_both'), sizing_mode='stretch_both')
    lay = row(column(children=[ p, data_table], sizing_mode='stretch_both'), sizing_mode='stretch_both')
    return lay


if __name__ == '__main__':
    main()
