#import dash_core_components as dcc
#import dash_html_components as html
from contextlib import redirect_stderr

from dash import dcc
from dash import html
from dash import dash_table

import dash_loading_spinners as dls

#from dash_holoniq_wordcloud import DashWordcloud

# Constantes
MAXCLUST = 4  # máxima cantidad de clusters

# Variables
cluster_por_nivel = []

start_layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '99%',
            'height': '120px',
            'lineHeight': '120px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True,
        contents=[],
        filename=[]
    ),
    dcc.Markdown(id= 'error_list', children=[]), #"* Error 1 \n* Error 2"
    html.Button('Continue', id='btn-continue', style={'font-family' : 'helvetica', 'textAlign' : 'center'}, disabled=False)
])


main_layout = lambda explorer: html.Div([
    #dcc.Store(id='store-data', data=[], storage_type='memory'),
    #dcc.Store(id='store-ind', data=[], storage_type='memory'),
    dcc.Store(id='store-explorer', data=explorer),
    dcc.ConfirmDialog(
        id='error_message',
        message="There are no solutions that fulfill all restrictions",
        ),
    html.H2("Clustering-based Pareto-Front exploration", style={'text-align': 'center'}),
    dcc.Slider(
        id="slider_nivel",
        min=0,
        max=10,
        value=0,
        step=1,
        marks={0: '0'},
        included=False,
        disabled=True
    ),
    html.Div([
        html.Div([
            html.Button('Restore', id='btn-restore', style={'font-family' : 'helvetica', 'width': "33%", 'height': "20%"}, disabled=True),
            html.Button('Undo', id='btn-undo', style={'font-family' : 'helvetica', 'width': "33%", 'height': "20%"}, disabled=True),
            html.Button('Zoom in', id='btn-zoomin', style={'font-family' : 'helvetica', 'width': "33%", 'height': "20%"}, disabled=False),
            dcc.RadioItems(
                id="cluster_seleccionado",
                value=1,
                # list comprehension to save time avoiding loops
                options=[{'label': 'Cluster{}'.format(c), 'value': c} for c in range(1, MAXCLUST + 1)],
                labelStyle={'display': 'inline-block'},
                inline=True
                ),
            html.Br(),
            html.Div(children='Desired number of clusters', style={'text-align': 'left'}),
            dcc.Slider(
                id="slider_num_clusters",
                min=2,
                max=MAXCLUST,
                value=MAXCLUST,
                step=1,
                # dict from list comprehension to save time avoiding loops
                marks=dict(zip(range(2, MAXCLUST+1), [str(x) for x in range(2, MAXCLUST+1)])),
                included=False,
                )
            ], style={'columnCount': 2}),
        html.Div([
            html.Div(children='Filters'),
            dcc.Dropdown(
                id='filtros-dropdown',
                options=[],
                value=None,
                multi=True,
                placeholder="Filter solutions that include..."
                ),
            html.Hr()
            ])
        ]),

    html.Div([
        dls.Hash(
            dcc.Graph(id='profit_cost_graph', responsive=True, style={'height': '30vw'}),
            color="#435278",
            speed_multiplier=2,
            size=100),
        ], style={'columnCount': 1}),
    html.Div([
        dls.Hash(
            dcc.Graph(id='dendrogram_graph', responsive=True, style={'height': '18vw'}),
            color="#435278",
            speed_multiplier=2,
            size=100),
        ], style={'columnCount': 1}),
    html.Div([html.H3("Data Table", style={'text-align': 'center'}),
             dash_table.DataTable(data=[], columns=[], id='data-table2', page_size=10,
                                  style_table={'overflowX': 'auto'},
                                  )]),
    html.Hr(),
    html.H3("Cluster metadata", style={'text-align': 'center'}),
    html.P("Box plot graph", style={'text-align': 'left'}),
    html.Div([
    html.P("y-axis:"),
        dcc.RadioItems(
            id='y-axis-boxplot',
            options=['profit', 'cost'],
            value='profit',
            inline=True
        ),
        dls.Hash(
            dcc.Graph(id='box_plot_graph', figure={}),
            color="#435278",
            speed_multiplier=2,
            size=100,
        )
    ], style={'columnCount': 1}),
    html.Hr(),
    html.Div([html.H3("Statistics", style={'text-align': 'center'}),
             dash_table.DataTable(data=[], columns=[], id='data-table-statistics', page_size=10,
                                  style_table={'overflowX': 'auto'},
                                  )], style={'columnCount': 1}),
    html.Hr(),
    html.Div([
        dls.Hash(
            dcc.Graph(id='treemaps-graph', responsive=True, style={'height': '75vw'}),
            color="#435278",
            speed_multiplier=2,
            size=100,
        )
    ], style={'columnCount': 1}),
    ])


layout = html.Div([
    dcc.Store(id='store-data', data=[], storage_type='memory'),
    #dcc.Store(id='store-ind', data=[], storage_type='memory'),
    dcc.Store(id='store-reqs', data=[], storage_type='memory'),
    dcc.Store(id='store-stks', data=[], storage_type='memory'),
    dcc.Store(id='store-keys', data=[], storage_type='memory'),
    html.Div(id='page_content', children=start_layout, style={'font-family' : 'helvetica'})
])