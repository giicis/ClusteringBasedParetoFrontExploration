import collections
import operator
from functools import reduce

import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import math

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
from scipy.spatial.distance import squareform, pdist
# from wordcloud import WordCloud, STOPWORDS
# from dash_holoniq_wordcloud import DashWordcloud


import json
import base64
import io
import ast

import dash  # (version 1.12.0) pip install dash
from dash.dependencies import Input, Output, State

import core
# from datetime import datetime

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

    filtro = [True] * len(data)
    # For each solution
    for r in range(len(data)):
        # For each key
        for key in keys:
            # A key is composed by a category and an index. For example: "reqs,4" refers to index 4 in the 'reqs' array of a solution.
            (categoria, index) = key.split(',')
            # The filter's corresponding position to each solution mantains its True value if that solution includes all the req/stk of the keys
            filtro[r] = np.logical_and(filtro[r], data[categoria][r][int(index)] == '1')

    return filtro


distance_matrix = None


def get_distance_matrix():
    """
    Distance matrix calculated from the data
    :return: distance matrix
    """
    global distance_matrix
    if distance_matrix is None:
        # Calculate the distance between each pair of quasi-optimal solutions.
        distance_matrix = squareform(pdist(data.values, lambda x, y: distance(x, y, max_profit, max_cost)))
        print('se generó la distance_matrix')
    return distance_matrix


# Statistics--------------------------------------------------------------------------
def cluster_statistics(indices, nivel, n_generado_de_clusters):
    """
    Statistics of each cluster
    :param indices: list of boolean that indicates which rows are going to be clustered.
    :param nivel: level of clustering zoom.
    :n_generado_de_clusters: number of generated clusters (int)
    :return: dictionary of statistics
    """
    global data

    df = data.loc[indices]
    estadisticos = {}
    for c in range(1, n_generado_de_clusters + 1):
        cluster_elements = df.loc[df["Level {}".format(nivel)] == str(c)]

        stat = {
            'n_elementos': len(cluster_elements),
            'profit_min': cluster_elements.profit.min(),
            'profit_median': cluster_elements.profit.median(),
            'profit_max': cluster_elements.profit.max(),
            'profit_SD': cluster_elements.profit.std(),
            'cost_min': cluster_elements.cost.min(),
            'cost_median': cluster_elements.cost.median(),
            'cost_max': cluster_elements.cost.max(),
            'cost_SD': cluster_elements.cost.std(),
        }

        estadisticos[c] = stat
    return estadisticos


## Plotting----------------------------------------
def fill(array):
    """
    Fills an array with empty dicts
    :param array: array to be filled
    :return: the filled array
    """
    faltantes = MAXCLUST - len(array)
    for x in range(faltantes):
        array.append({})
    return array


def plot_treemaps(indices, dim, categorias, nivel):
    """
    Generates a treemap for each cluster
    :param indices: list of boolean that indicates which rows are going to be clustered.
    :param dim: data column to get info from (string)
    :param categorias: dict of categories to pair the data info
    :param nivel: level of clustering zoom.
    :return: array of treemaps
    """
    global data
    df = data.loc[indices]
    array_treemaps = []
    conjuntos_palabras = {}
    for x in list(set(df["Level {}".format(nivel)])):
        conjuntos_palabras[str(x)] = {}  # preparo una lista de diccionarios vacios, uno por cada cluster
    for i in range(MAXCLUST):
        array_treemaps.append({})
    # array_treemaps = [{},{},{},{}]  # array para retornar los treemaps generados

    # esto habría que hacerlo 1 sola vez al principio cuando se carga el dataset
    # TODO: moverlo al lugar que corresponde
    keys_counts = reduce(operator.add, [collections.Counter(obj['keys']) for (pos, obj) in categorias.items()])
    #print(categorias)
    #print(keys_counts)

    # calculamos las frecuencias de cada categoría en cada solución
    keys_counts_per_solution = {
        r: reduce(operator.add, [collections.Counter(categorias[str(i)]['keys']) for i in range(len(categorias))
                                 if df[dim][r][i] == '1'], collections.Counter()) for r in df["id"]}

    # calculamos cuán "cubierta" está cada palabra por las distintas soluciones:
    #print("Antes de la normalizacion")
    for r, counter in keys_counts_per_solution.items():
        for key in counter.keys():
            counter[key] = counter[key] / keys_counts[key]

    fuzzy_quantifiers = ["few", "several", "many", ""]
    mf_fuzzy_quantifiers = {q: (0. + 0.25 * j, 0.25 + 0.25 * j, 0.5 + 0.25 * j) for j, q in enumerate(fuzzy_quantifiers)}

    # Triangular membership function with abc = (a, b, c)
    trimf = lambda x, abc : max(0., min((x - abc[0])/(abc[1]-abc[0]), (x - abc[2])/(abc[1]-abc[2])))

    print("Calculando las funciones de pertenencia")

    build_mf = lambda counter, abc: {key:trimf(count, abc) for key, count in dict(counter).items()}
    quantified_keys_membership_per_solution = {(q, r):collections.Counter(build_mf(counter, mf_fuzzy_quantifiers[q])) for r, counter in keys_counts_per_solution.items() for q in fuzzy_quantifiers}
    level = "Level {}".format(nivel)  # lo asignamos antes por tema performance
    print("Terminó el cálculo de las funciones de pertenencia")
    clusters = list(set(df[level]))
    clusters.sort()
    for c in clusters:  # se crea un treemap por cada cluster
        print("Comenzamos con el cluster {}".format(c))

        df2 = df[df[level] == c] # obtenemos sólo las filas del cluster
        print("Calculando cantidad de filas")
        cluster_size = len(df2)
        print("Calculando cardinales")
        quantified_keys_by_cluster = {(q, key): sum([quantified_keys_membership_per_solution[(q, r.id)][key] for r in df2.itertuples()]) for q in fuzzy_quantifiers for key in keys_counts.keys()}
        print("Calculando borrosidades")
        fuzziness_keys_by_cluster = {(q, key): 2 * sum(
            [abs(quantified_keys_membership_per_solution[(q, r.id)][key] - (1 if quantified_keys_membership_per_solution[(q, r.id)][key] >= 0.5 else 0))
             for r in df2.itertuples()]
        )/cluster_size for q in fuzzy_quantifiers for key in keys_counts.keys()}
        print("Fin de los cálculos")
        ids = ["{} {}".format(q, r) for (q, r), count in quantified_keys_by_cluster.items() if count]

        labels = ids
        parents = ["Cluster {}".format(c) for _ in ids]
        values = [count for _, count in quantified_keys_by_cluster.items() if count] #fuzzy cardinal
        colors = [fuzziness_keys_by_cluster[(q, key)] for (q, key), count in quantified_keys_by_cluster.items() if count] # index of fuzziness

        fig = go.Figure(go.Treemap(
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
        ))

        fig.update_layout(margin=dict(t=15, l=5, r=5, b=15))
        array_treemaps[int(c) - 1] = fig
        print("Terminamos con el cluster {}".format(c))
    return array_treemaps


def algoritmo_clustering(indices, n_deseado_de_clusters, nivel):
    """
    Hierarchical clustering on the dataframe rows indicated by indeces parameter.
    :param indices: list of boolean that indicates which rows are going to be clustered.
    :n_deseado_de_clusters: maximum number of clusters to be generated.
    :param nivel: level of clustering zoom.
    :return: linkage matrix, number of generated clusters.
    """
    global data

    # time_1 = datetime.now()

    # filter rows and columns according to indices
    dist_matrix_filtrada = pd.DataFrame.to_numpy((pd.DataFrame(get_distance_matrix())).loc[indices, indices])

    # time_2 = datetime.now()
    # time_diff = time_2 - time_1
    # print(time_diff)

    # get linkage matrix for hierarchical clustering
    linkage_matrix = linkage(dist_matrix_filtrada, method="complete")

    # time_3 = datetime.now()
    # time_diff = time_3 - time_2
    # print(time_diff)

    # get clusters
    clusters = fcluster(linkage_matrix, n_deseado_de_clusters, criterion='maxclust')

    # time_4 = datetime.now()
    # time_diff = time_4 - time_3
    # print(time_diff)

    n_generado_de_clusters = len(set(clusters))

    # if index = True, next(clusters) else None
    clusters = iter(clusters)
    data["Level {}".format(nivel)] = [str(next(clusters)) if valor else None for valor in
                                      indices]  # esta guardando los clusters como float

    return linkage_matrix, n_generado_de_clusters


def plot_all(nivel, n_deseado_de_clusters, array_filtros):
    """
    Runs the clustering algorithm and generates a dendogram, cluster statistics and treemaps, and a dispersion graph.
    :param nivel: level of clustering zoom.
    :n_deseado_de_clusters: maximum number of clusters to be generated.
    :return: control variables, graphs to be displayed.
    """
    global data
    global requirements
    global stakeholders

    error = False
    try:
        # Applies the filters on each stage of clustering (this avoids errors when zooming out after applying filters)
        for n in range(nivel + 1):
            indices = np.logical_and(obtener_indices(n), array_filtros)
            if not (any(indices)):
                # An exception is raised if the resulting boolean array is all False
                raise Exception("There are no solutions that fulfill all restrictions")
            (linkage_matrix, n_generado_de_clusters) = algoritmo_clustering(indices, n_deseado_de_clusters, n)
    except:
        error = True
        return (error, True, [], {}, {}, {}, {}, {}, "", "", "", "", {}, {}, {}, {}, {})

    dn = ff.create_dendrogram(linkage_matrix)

    dn.update_xaxes(showticklabels=False)
    dn.update_yaxes(showticklabels=False)
    dn.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    wcreq = plot_treemaps(indices, "reqs", requirements, nivel)
    wcstk = plot_treemaps(indices, "stks", stakeholders, nivel)

    estadisticos = cluster_statistics(indices, nivel, n_generado_de_clusters)
    stat = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: ""}
    for clus in range(1, n_generado_de_clusters + 1):
        stat[
            clus] = "Cluster {}\nElement count:\t{}\nProfit min:\t{}\nProfit median:\t{}\nProfit max:\t{}\nProfit SD:\t{}\nCost min:\t{}\nCost median:\t{}\nCost max:\t{}\nCost SD:\t{}".format(
            clus, estadisticos[clus]['n_elementos'], estadisticos[clus]['profit_min'],
            estadisticos[clus]['profit_median'], estadisticos[clus]['profit_max'], estadisticos[clus]['profit_SD'],
            estadisticos[clus]['cost_min'], estadisticos[clus]['cost_median'], estadisticos[clus]['cost_max'],
            estadisticos[clus]['cost_SD'])

    pcg = px.scatter(data.loc[indices], x="profit", y="cost", color=("Level {}".format(nivel)), hover_data=['id'])
    pcg.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))

    # DataTable
    datatable = data.loc[indices].__dict__('records')
    ##Replace datatable requirementes and stakeholders
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
    # DataColumns
    datacolumns = []
    for i in data.columns:
        col_options = {"name": i, "id": i}
        if i == "id":
            col_options["type"] = "numeric"
        datacolumns.append(col_options)
    # EndDataTable

    # Disable zoomin if there are no clusters to choose
    deshabilitar_zoomin = (n_generado_de_clusters <= 1)
    # You can't zoom in a cluster with less than 3 solutions
    options = [{'label': 'Cluster{}'.format(c), 'value': c, 'disabled': (estadisticos[c]['n_elementos'] <= 2)} for c in
               range(1, n_generado_de_clusters + 1)]
    return (
    error, deshabilitar_zoomin, options, dn, wcreq[0], wcreq[1], wcreq[2], wcreq[3], stat[1], stat[2], stat[3], stat[4],
    wcstk[0], wcstk[1], wcstk[2], wcstk[3], pcg, datatable, datacolumns)


def obtener_indices(nivel):
    """
    Generates the boolean array used to filter the pareto-front by its level of clustering zoom
    :param nivel: level of clustering zoom.
    :return: boolean array
    """
    global data
    global cluster_por_nivel

    if nivel >= 1:
        # obtiene los indices de las filas cuya última columna de cluster coincide con
        # el ultimo cluster elegido. Es un array de booleanos.
        indices = data["Level {}".format(nivel - 1)] == cluster_por_nivel[nivel - 1]
    else:
        indices = [True] * len(data)
    return indices


def validate_files(list_upload):
    data_isok = False
    reqs_isok = False
    stks_isok = False
    error_list = []

    print("Validating")
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
    #global data
    #global requirements
    #global stakeholders
    #global palabras_clave

    #continue_disabled = True
    list_upload = dict(zip(list_filenames, list_contents))
    error_list = validate_files(list_upload)
    print(error_list)
    #dataset = []
    continue_disabled = not (error_list == [])
    #if (error_list == []):
        #continue_disabled = False

        #content_type, content_string = list_upload["pareto_front.json"].split(',')
        #decoded = base64.b64decode(content_string)
        #data = pd.read_json(io.BytesIO(decoded), orient='index', dtype={'reqs': str, 'stks': str})

        #content_type, content_string = list_upload["requirements.json"].split(',')
        #requirements = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))

        #content_type, content_string = list_upload["stakeholders.json"].split(',')
        #stakeholders = ast.literal_eval((base64.b64decode(content_string)).decode("UTF-8"))


        #dataset = data.to_dict('records')

        #print(dataset[0], requirements, stakeholders)

        #print(type(dataset), type(requirements), type(stakeholders))

        #palabras_clave = []
        #for x in range(len(requirements)):
        #    palabras_clave.append({
        #        'label': requirements[str(x)]['id'],
        #        'value': "reqs,{}".format(x)
        #    })
        #for x in range(len(stakeholders)):
        #    palabras_clave.append({
        #        'label': stakeholders[str(x)]['id'],
        #        'value': "stks,{}".format(x)
        #    })

    reporte_errores = "".join(["\n* " + error for error in error_list])
    #for error in error_list:
    #    reporte_errores += "\n* " + error
    return continue_disabled, reporte_errores



@app.callback(
    [Output(component_id='page_content', component_property='children'),
     Output(component_id='store-data', component_property='data'),
     Output(component_id='store-reqs', component_property='data'),
     Output(component_id='store-stks', component_property='data'),
     Output(component_id='store-keys', component_property='data'),
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
        print("Antes de instanciar el test_layout")
        layout = test_layout(explorer=explorer)
        print("Luego de instanciar el test_layout")


    return layout, dataset, requirements, stakeholders, palabras_clave

@app.callback(
    [Output(component_id='filtros-dropdown', component_property='options')],
    Input(component_id='store-keys', component_property='data')
)
def populate_filtros_options(stored):

    #print(stored)
    return stored,

@app.callback(
    [Output(component_id='data-table2', component_property='data'),
    Output(component_id='data-table2', component_property='columns')],
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data')]
)

def test_table(stored_data, explorer_as_dict):
    #print("Test_Table")
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()
    #print("Explorer: {}".format(explorer_as_dict))
    explorer.load(explorer_as_dict)
    #print("Cargó el explorer")
    #print("Estos son los índices que se cargaron:\n{}".format(explorer.actual_state.indexes))
    dataset = dataset.loc[explorer.actual_state.indexes]

    # si clusters está disponible en el estado agregarlo como última columna

    if explorer.actual_state.clusters is not None:
        print("Entró a agregar la columna de clusters en la tabla")
        dataset["clusters"] = explorer.actual_state.clusters
        print("Salió de agregar la columna de clusters en la tabla")

    datacolumns = []
    for i in dataset.columns:
        col_options = {"name": i, "id": i}
        if i == "id":
            col_options["type"] = "numeric"
        datacolumns.append(col_options)
    #print(datacolumns)

    # Ojo que tal vez hay que hacer un no_update si el último comando no actualizó indexes
    return dataset.to_dict('records'), datacolumns

@app.callback(
    [Output(component_id='box_plot_graph', component_property='figure')],
    [Input(component_id='store-data', component_property='data'),
     Input(component_id='store-explorer', component_property='data'),
     Input(component_id='y-axis-boxplot', component_property='value')]
)

def boxplot(stored_data, explorer_as_dict, y_input):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()
    explorer.load(explorer_as_dict)
    dataset = dataset.loc[explorer.actual_state.indexes]
    print("Boxplots")
    print(explorer.actual_state.clusters)
    if explorer.actual_state.clusters is not None:
        print("Entró a agregar la columna de clusters en la tabla")
        dataset["clusters"] = explorer.actual_state.clusters
        print("Salió de agregar la columna de clusters en la tabla")

    print(dataset.columns)
    bxp = px.box(dataset, x="clusters", y=y_input)
    return bxp


@app.callback(
    [Output(component_id='store-explorer', component_property='data')],
    Input(component_id='filtros-dropdown', component_property='value'),
    [State(component_id='store-explorer', component_property='data'),
     State(component_id='store-data', component_property='data')]
)
def update_explorer(keys, explorer_as_dict, stored_data):
    dataset = pd.DataFrame(stored_data)
    explorer = core.ParetoFrontExplorer()
    # print("Explorer: {}".format(explorer_as_dict))
    # print("Cargó el explorer")
    # print("Estos son los índices que se cargaron:\n{}".format(explorer.actual_state.indexes))

    print("Por levantar el explorer")
    explorer.load(explorer_as_dict)
    print(explorer_as_dict)
    print(explorer.save())
    print("Explorer loaded")

    # get triggered event from callback_context
    ctx = dash.callback_context

    # Filtrado
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'filtros-dropdown':
            if keys is not None:
                if explorer.actual_state.filters is not None:
                    print("Undoing command")
                    print(explorer.save())
                    explorer.undo()
                    print("Undone command")
                    print(explorer.actual_state.__dict__())
                dataset = dataset.loc[explorer.actual_state.indexes]
                print("Por generar los filtros")
                array_filtros = generar_filtros(dataset, keys)
                print("Filtros generados")
                explorer.apply_filter(array_filtros)
                print("Explorer terminó de aplicar filtros")
    #print(stored)
    print("Explorer que devolvemos")
    print(explorer.save())
    return explorer.save(),

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=modo_desarrollador, dev_tools_props_check=modo_desarrollador)
