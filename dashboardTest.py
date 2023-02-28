import collections
import operator
from functools import reduce
from itertools import count

import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import math
import time

from dash import no_update
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
from scipy.spatial.distance import squareform, pdist

import json
import base64
import io
import ast

import dash  # (version 1.12.0) pip install dash
from dash.dependencies import Input, Output, State

import core

# ------------------------------------------------------------------------------
# The layout was placed on a separate file for organizational reasons.
# The variables related to this layout were also put aside
from layout import *
from util import distance

modo_desarrollador = False  # True


def generar_filtros(data, keys):
    """
    Generates a boolean array that filter the solutions
    :param keys: list of strings that represent requerimients or stakeholders
    :return: boolean array
    """
    #global data

    filtro = {r: True for r in data.index}
    # For each solution
    for r in data.index:
        # For each key
        for key in keys:
            # A key is composed by a category and an index. For example: "reqs,4" refers to index 4 in the 'reqs' array of a solution.
            (categoria, index) = key.split(',')
            # The filter's corresponding position to each solution mantains its True value if that solution includes all the req/stk of the keys
            filtro[r] = np.logical_and(filtro[r], data[categoria][r][int(index)] == '1')

    return filtro


def plot_treemaps(dataset, dim, categorias, nivel):
    """
    Generates a treemap for each cluster
    :param indices: list of boolean that indicates which rows are going to be clustered.
    :param dim: data column to get info from (string)
    :param categorias: dict of categories to pair the data info
    :param nivel: level of clustering zoom.
    :return: array of treemaps
    """
    dict_treemaps = {}
    conjuntos_palabras = {}
    cluster_set = set(dataset["clusters"])
    for x in cluster_set:
        conjuntos_palabras[str(x)] = {}  # preparo una lista de diccionarios vacios, uno por cada cluster
        dict_treemaps[str(x)] = {}

    # esto habría que hacerlo 1 sola vez al principio cuando se carga el dataset
    # TODO: moverlo al lugar que corresponde
    keys_counts = reduce(operator.add, [collections.Counter(obj['keys']) for (pos, obj) in categorias.items()])

    # calculamos las frecuencias de cada categoría en cada solución
    keys_counts_per_solution = {
        r: reduce(operator.add, [collections.Counter(categorias[str(i)]['keys']) for i in range(len(categorias))
                                 if dataset[dim][r][i] == '1'], collections.Counter()) for r in dataset["id"]}

    # calculamos cuán "cubierta" está cada palabra por las distintas soluciones:
    for r, counter in keys_counts_per_solution.items():
        for key in counter.keys():
            counter[key] = counter[key] / keys_counts[key]

    fuzzy_quantifiers = ["few", "several", "many", ""]
    mf_fuzzy_quantifiers = {q: (0. + 0.25 * j, 0.25 + 0.25 * j, 0.5 + 0.25 * j) for j, q in enumerate(fuzzy_quantifiers)}

    # Triangular membership function with abc = (a, b, c)
    trimf = lambda x, abc : max(0., min((x - abc[0])/(abc[1]-abc[0]), (x - abc[2])/(abc[1]-abc[2])))

    build_mf = lambda counter, abc: {key:trimf(count, abc) for key, count in dict(counter).items()}
    quantified_keys_membership_per_solution = {(q, r):collections.Counter(build_mf(counter, mf_fuzzy_quantifiers[q])) for r, counter in keys_counts_per_solution.items() for q in fuzzy_quantifiers}

    for c in cluster_set:  # se crea un treemap por cada cluster

        df2 = dataset[dataset["clusters"] == c] # obtenemos sólo las filas del cluster

        cluster_size = len(df2)

        quantified_keys_by_cluster = {(q, key): sum([quantified_keys_membership_per_solution[(q, r.id)][key] for r in df2.itertuples()]) for q in fuzzy_quantifiers for key in keys_counts.keys()}

        fuzziness_keys_by_cluster = {(q, key): 2 * sum(
            [abs(quantified_keys_membership_per_solution[(q, r.id)][key] - (1 if quantified_keys_membership_per_solution[(q, r.id)][key] >= 0.5 else 0))
             for r in df2.itertuples()]
        )/cluster_size for q in fuzzy_quantifiers for key in keys_counts.keys()}

        ids = ["{} {}".format(q, r) for (q, r), count in quantified_keys_by_cluster.items() if count]

        labels = ids
        parents = ["Cluster {}".format(c) for _ in ids]
        values = [count for _, count in quantified_keys_by_cluster.items() if count] #fuzzy cardinal
        colors = [fuzziness_keys_by_cluster[(q, key)] for (q, key), count in quantified_keys_by_cluster.items() if count] # index of fuzziness

        treemap = go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                #colorscale=random.choice(['ice', 'solar', 'Aggrnyl', 'Hot']),
                colorscale='ice',
                showscale=True,
                cmid=0.5,
                cmin=0.,
                cmax=1.
            ),
            hovertemplate='<b>%{label} </b> <br> Support: %{value}<br> Fuzziness: %{color:.2f}',
        )

        dict_treemaps[str(c)] = treemap
    return dict_treemaps


def validate_files(list_upload):
    data_isok = False
    reqs_isok = False
    stks_isok = False
    error_list = []

    #print("Validating")
    if "pareto_front.json" in list_upload.keys():
        content_type, content_string = list_upload["pareto_front.json"].split(',')
        decoded = base64.b64decode(content_string)

        dataframe = pd.read_json(io.BytesIO(decoded), orient='index', dtype={'reqs': str, 'stks': str})
        if set(['id', 'profit', 'cost', 'reqs', 'stks']).issubset(dataframe.columns):
            if (len(dataframe) >= 3):
                data_isok = True
            else:
                error_list.append("The file 'pareto_front.json' must have at least 3 elements")
        else:
            error_list.append(
                "The file 'pareto_front.json' does not meet the standard (it must have 'id','profit','cost','reqs' and 'stks' fields)")
    else:
        error_list.append("The file 'pareto_front.json' is missing")

    if "requirements.json" in list_upload.keys():
        content_type, content_string = list_upload["requirements.json"].split(',')
        decoded = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))

        if data_isok:
            if (len(decoded) == len(dataframe['reqs'][0])):
                if (("id" in decoded['0'].keys()) and ("keys" in decoded['0'].keys())):
                    reqs_isok = True
                else:
                    error_list.append(
                        "The file 'requirements.json' does not meet the standard (it must have 'id' and 'keys' fields)")
            else:
                error_list.append(
                    "The file 'requirements.json' do not fit the lenght of 'pareto_front.json' 'reqs' field")
    else:
        error_list.append("The file 'requirements.json' is missing")

    if "stakeholders.json" in list_upload.keys():
        content_type, content_string = list_upload["stakeholders.json"].split(',')
        decoded = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))

        if data_isok:
            if (len(decoded) == len(dataframe['stks'][0])):
                if (("id" in decoded['0'].keys()) and ("keys" in decoded['0'].keys())):
                    stks_isok = True
                else:
                    error_list.append(
                        "The file 'stakeholders.json' does not meet the standard (it must have 'id' and 'keys' fields)")
            else:
                error_list.append(
                    "The file 'stakeholders.json' do not fit the lenght of 'pareto_front.json' 'stks' field")
    else:
        error_list.append("The file 'stakeholders.json' is missing")

    return error_list


# ------------------------------------------------------------------------------

app = dash.Dash(__name__)
# App layout
app.title = "ParetoFrontExploration"
app.layout = layout


# ------------------------------------------------------------------------------

@app.callback(
    [Output(component_id='btn-continue', component_property='disabled'),
     Output(component_id='error_list', component_property='children')],
    [Input(component_id='upload-data', component_property='contents')],
    [State(component_id='upload-data', component_property='filename')]
)
def upload_data(list_contents, list_filenames):
    """
    Callback function for uploading the dataframe.
    :content: .json file that contains the dataframe.
    :return: a boolean that disables the "Continue" button and a graph of the dataframe.
    """

    list_upload = dict(zip(list_filenames, list_contents))
    error_list = validate_files(list_upload)
    #print(error_list)

    continue_disabled = not (error_list == [])

    reporte_errores = "".join(["\n* " + error for error in error_list])

    return continue_disabled, reporte_errores



@app.callback(
    [Output(component_id='page_content', component_property='children'),
     Output(component_id='store-data', component_property='data'),
     Output(component_id='store-reqs', component_property='data'),
     Output(component_id='store-stks', component_property='data'),
     Output(component_id='store-keys', component_property='data'),
     Output(component_id="loading-output-1", component_property= 'children'),
     #Output(component_id='store-explorer', component_property='data')
    ],
    [Input(component_id='btn-continue', component_property='n_clicks')],
    [State(component_id='upload-data', component_property='contents'),
    State(component_id='upload-data', component_property='filename')]
)
def start(n_clicks, list_contents, list_filenames):
    """
    Callback function that updates the layout.
    :n_clicks: not used
    :content: .json file that contains the dataframe.
    :return: the main layout.
    """

    layout = start_layout
    dataset = []
    requirements = []
    stakeholders = []
    palabras_clave = []

    if n_clicks > 0:

        list_upload = dict(zip(list_filenames, list_contents))

        content_type, content_string = list_upload["pareto_front.json"].split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_json(io.BytesIO(decoded), orient='index', dtype={'reqs': str, 'stks': str})

        content_type, content_string = list_upload["requirements.json"].split(',')
        requirements = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))

        content_type, content_string = list_upload["stakeholders.json"].split(',')
        stakeholders = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))

        for x in range(len(requirements)):
            palabras_clave.append({
                'label': requirements[str(x)]['id'],
                'value': "reqs,{}".format(x)
            })
        for x in range(len(stakeholders)):
            palabras_clave.append({
                'label': stakeholders[str(x)]['id'],
                'value': "stks,{}".format(x)
            })

        max_cost = max(data["cost"])
        max_profit = max(data["profit"])
        distance_matrix = squareform(pdist(data, lambda x, y: distance(x, y, max_profit, max_cost)))
        linkage_matrix = linkage(distance_matrix, method="complete")
        explorer = core.ParetoFrontExplorer(state=core.ExplorerState(indexes=data.index, linkage_matrix=linkage_matrix))
        dataset = data.to_dict('records')
        #print("Antes de instanciar el test_layout")
        explorer.cluster(4)
        layout = main_layout(explorer=explorer.save())
        #print("Luego de instanciar el test_layout")


    return layout, dataset, requirements, stakeholders, palabras_clave, {'display': 'none'}

@app.callback(
    [Output(component_id='filtros-dropdown', component_property='options')],
    Input(component_id='store-keys', component_property='data')
)
def populate_filtros_options(stored):

    #print(stored)
    return stored,


@app.callback(
    [Output(component_id='store-explorer', component_property='data')],
    [Input(component_id='btn-zoomin', component_property='n_clicks'),
     Input(component_id='btn-undo', component_property='n_clicks'),
     Input(component_id='btn-restore', component_property='n_clicks'),
     Input(component_id='filtros-dropdown', component_property='value'),
     Input(component_id='slider_num_clusters', component_property='value')
     ],
    [State(component_id='store-explorer', component_property='data'),
     State(component_id='store-data', component_property='data'),
     State(component_id='cluster_seleccionado', component_property='value'),
     ]
)
def update_explorer(zoomin_nclicks, undo_nclicks, restore_nclicks, keys, n_clusters, explorer_as_dict, stored_data, selected_cluster):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except:
        print("Algo salió mal con la carga del explorer")

    # get triggered event from callback_context
    ctx = dash.callback_context

    # Filtrado
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'slider_num_clusters':
            explorer.cluster(n_clusters)

        if trigger_id == 'btn-zoomin':
            explorer.zoomin_then_cluster(selected_cluster, n_clusters)

        if trigger_id == 'btn-undo':
            explorer.undo()

        if trigger_id == 'btn-restore':
            explorer.restore()
            explorer.cluster(n_clusters)

        if trigger_id == 'filtros-dropdown':
            # Estoy prácticamente seguro de que esta lógica se puede encapsular dentro del explorer, deuda técnica
            if keys is not None:
                dict_filtros = generar_filtros(dataset, keys)
                explorer.apply_filter(dict_filtros)

    return explorer.save(),

@app.callback(
    [Output(component_id='btn-zoomin', component_property='disabled'),
     Output(component_id='btn-undo', component_property='disabled'),
     Output(component_id='btn-restore', component_property='disabled'),
     Output(component_id='cluster_seleccionado', component_property='options'),
     Output(component_id='cluster_seleccionado', component_property='value'),
     Output(component_id='slider_nivel', component_property='marks'),
     Output(component_id='slider_nivel', component_property='value'),
     ],
    Input(component_id='store-explorer', component_property='data')
)
def manage_ui_controls(explorer_as_dict):
    explorer = core.ParetoFrontExplorer()
    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer\n{}".format(ex))

    cluster_seleccionado_options = no_update
    zoomin_disabled = False
    undo_disabled = True
    restore_disabled = True
    cluster_seleccionado_value = None

    if explorer.actual_state.clusters is not None:
        clusters_set = {explorer.actual_state.clusters[idx] for idx in explorer.actual_state.indexes}
        cluster_seleccionado_options = [{'label': 'Cluster{}'.format(c), 'value': c, 'disabled': not explorer.can_zoomin(c)}
                                        for c in clusters_set]

        zoomin_disabled = len(clusters_set) <= 1 or all([option['disabled'] for option in cluster_seleccionado_options])
        cluster_seleccionado_value = None if zoomin_disabled else [option['value'] for option in cluster_seleccionado_options][0]
        undo_disabled = not explorer.can_undo()
        restore_disabled = not explorer.can_restore()

    slider_nivel_marks = {i:str(i) for i in range(explorer.actual_state.level+1)}

    return zoomin_disabled, undo_disabled, restore_disabled, cluster_seleccionado_options, cluster_seleccionado_value, slider_nivel_marks, explorer.actual_state.level


@app.callback(
    [Output(component_id='data-table2', component_property='data'),
    Output(component_id='data-table2', component_property='columns')],
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data')],
    [State(component_id='store-reqs', component_property='data'),
     State(component_id='store-stks', component_property='data')]
)

def test_table(stored_data, explorer_as_dict, requirements, stakeholders):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del test_table\n{}".format(ex))

    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is not None:

        dataset["clusters"] = [explorer.actual_state.clusters[idx] for idx in dataset.index]


    datatable = dataset.sort_values(by=['id'], inplace=False).to_dict('records')

    for data_line in datatable:
        req_list = []
        stk_list = []
        # Requirements
        for req_index in range(len(data_line["reqs"])):
            if data_line["reqs"][req_index] == "1":
                req_list.append(requirements[str(req_index)]["id"])
        # StakeHolders
        for stk_index in range(len(data_line["stks"])):
            if data_line["stks"][stk_index] == "1":
                stk_list.append(stakeholders[str(stk_index)]["id"])
        data_line["reqs"] = ", ".join(req_list)
        data_line["stks"] = ", ".join(stk_list)

    datacolumns = []
    for i in dataset.columns:
        col_options = {"name": i, "id": i}
        if i == "id":
            col_options["type"] = "numeric"
        datacolumns.append(col_options)

    # Ojo que tal vez hay que hacer un no_update si el último comando no actualizó indexes
    return datatable, datacolumns

@app.callback(
    Output(component_id='profit_cost_graph', component_property='figure'),
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data')]
)
def plot_profit_cost_graph(stored_data, explorer_as_dict):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer\n{}".format(ex))

    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is not None:
        dataset["clusters"] = [str(explorer.actual_state.clusters[idx]) for idx in dataset.index]

    pcg = px.scatter(dataset, x="profit", y="cost", color="clusters", hover_data=['id'], color_discrete_sequence=['#d73027', '#fc8d59', '#fee090', '#4575b4', '#91bfdb'])
    pcg.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return pcg

@app.callback(
    Output(component_id='dendrogram_graph', component_property='figure'),
    [Input(component_id='store-explorer', component_property='data'),
     Input(component_id='show-ids-checkbox', component_property='value')
     ]
)
def plot_dendrogram(explorer_as_dict, show_ids):
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer en plot_dendrogram\n{}".format(ex))

    color_treshold = None
    if explorer.actual_state.clusters is not None:
        n_clusters = len(set(explorer.actual_state.clusters.values()))
        pos = max(-n_clusters+1, -len(explorer.actual_state.linkage_matrix))
        color_treshold = explorer.actual_state.linkage_matrix[pos, 2] - 0.001

    dn = ff.create_dendrogram(explorer.actual_state.linkage_matrix, linkagefun=lambda x: explorer.actual_state.linkage_matrix, color_threshold=color_treshold, colorscale=['#d73027', '#d81b60', '#ffc107', '#4575b4', '#fc8d59', '#1e88e5', '#91bfdb', '#fee090'])

    dn.update_xaxes(showticklabels=True if show_ids else False)
    #dn.update_yaxes(showticklabels=False)
    dn.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return dn

@app.callback(
    Output(component_id='box_plot_graph', component_property='figure'),
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data'),
     Input(component_id='y-axis-boxplot', component_property='value')]
)
def plot_boxplot_graph(stored_data, explorer_as_dict, y_input):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer en el plot_boxplot_graph\n{}".format(ex))

    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is not None:
        dataset["clusters"] = [explorer.actual_state.clusters[idx] for idx in dataset.index]

    fig = px.box(dataset, x="clusters", y=y_input)

    return fig  # returned objects are assigned to the component property of the Output

@app.callback(
    [Output(component_id='data-table-statistics', component_property='data'),
     Output(component_id='data-table-statistics', component_property='columns')
    ],
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data')]
)
def write_statistics(stored_data, explorer_as_dict):

    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer\n{}".format(ex))

    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is None:
        raise dash.exceptions.PreventUpdate("Can't calc statistics if clusters is not set")
    else:
        dataset["clusters"] = [explorer.actual_state.clusters[idx] for idx in dataset.index]
        stats = dataset[["id", "clusters", "cost", "profit"]].groupby("clusters").agg({'id': 'count',
                                                                                       'profit': ['min', 'median', 'max', 'std'],
                                                                                       'cost':['min', 'median', 'max', 'std']})
        stats.columns = ['_'.join(col) for col in stats.columns.values]

        stats = stats.reset_index(names="cluster") # Para que se vea el número de cluster

        datacolumns = []
        for i in stats.columns:
            col_options = {"name": i, "id": i}
            datacolumns.append(col_options)

    return stats.to_dict('records'), datacolumns  # returned objects are assigned to the component property of the Output

@app.callback(
    Output(component_id='treemaps-graph', component_property='figure'),
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data'),
     Input(component_id='store-reqs', component_property='data'),
     Input(component_id='store-stks', component_property='data'),]
)
def plot_treemaps_graphs(stored_data, explorer_as_dict, requirements, stakeholders):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()

    try:
        explorer.load(explorer_as_dict)
    except Exception as ex:
        print("Algo salió mal durante la carga del explorer\n{}".format(ex))

    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is None:
        raise dash.exceptions.PreventUpdate("Can't calc statistics if clusters is not set")

    dataset["clusters"] = [explorer.actual_state.clusters[idx] for idx in dataset.index]

    tmreq = plot_treemaps(dataset, "reqs", requirements, explorer.actual_state.level)
    tmstk = plot_treemaps(dataset, "stks", stakeholders, explorer.actual_state.level)

    fig = make_subplots(
        2, NCLUSTERS := len(tmreq.keys()),
        specs=[[{'type':'domain'} for _ in range(NCLUSTERS)] for _ in range(2)],
        horizontal_spacing=0.01,
        vertical_spacing=0.05
    )

    for i, tm in enumerate(tmreq.values()):
        fig.add_trace(tm, 1, i+1)

    for i, tm in enumerate(tmstk.values()):
        fig.add_trace(tm, 2, i+1)

    fig.update_layout(margin=dict(t=15, l=5, r=5, b=15))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=modo_desarrollador, dev_tools_props_check=modo_desarrollador)
