import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table, State
from sklearn.cluster import MeanShift, estimate_bandwidth

# --- Fonctions de traitement des données ---

def insert_useful_column(df):
    pannes_a_exclure = ["AUTO FLT A/THR OFF", "AUTO FLT AP OFF", "BRAKES HOT", "SURV ROW/ROP LOST", "NAV ALTI DISCREPANCY"]
    df.insert(df.columns.get_loc('FAULT') + 1, 'USEFUL', ~df['FAULT'].isin(pannes_a_exclure))
    return df

def filter_rows_by_mean_gap(df, facteur):
    filtered_rows = []
    for index, row in df.iterrows():
        event_dates = sorted(
            [int(col.split(" ")[1]) for col in df.filter(like="Day ") if row[col] != 0],
            reverse=True
        )
        if not event_dates:
            continue

        X = np.array(event_dates).reshape(-1, 1)
        bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)
        mean_shift = MeanShift(bandwidth=bandwidth).fit(X)
        clusters = mean_shift.labels_

        last_event_day = min(event_dates)
        last_cluster = clusters[np.where(np.array(event_dates) == last_event_day)[0][0]]
        cluster_days = [event_dates[i] for i in range(len(event_dates)) if clusters[i] == last_cluster]

        if len(cluster_days) >= 3:
            cluster_days.sort()
            mean_interval = np.mean(np.diff(cluster_days))
        else:
            mean_interval = 0

        if facteur * mean_interval > last_event_day:
            filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)

def calculate_recent_occurrences(df):
    result = []
    for _, row in df.iterrows():
        event_dates = sorted(
            [int(col.split(" ")[1]) for col in df.filter(like="Day ") if row[col] != 0],
            reverse=True
        )
        if not event_dates:
            recent_occurrences = 0
        else:
            X = np.array(event_dates).reshape(-1, 1)
            bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)
            clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)
            last_cluster = clusters_raw[-1] if len(clusters_raw) > 0 else 0
            recent_occurrences = sum(clusters_raw == last_cluster)
        result.append(recent_occurrences)

    df["Occurrences récentes"] = result
    max_occurrences = max(result) if result else 1
    df["Color Scale"] = df["Occurrences récentes"] / max_occurrences
    return df[["IMMAT", "ATA", "FAULT", "Occurrences récentes", "Color Scale"]].sort_values(
        by="Occurrences récentes", ascending=False
    )

def plot_timeline_with_clusters_meanshift_plotly(df, immat=None, fault=None):
    df_selected = df[(df["IMMAT"] == immat) & (df["FAULT"] == fault)]
    if df_selected.empty:
        return go.Figure()

    event_dates = sorted(
        [int(col.split(" ")[1]) for col in df.filter(like="Day ") if df_selected.iloc[0][col] != 0],
        reverse=True
    )
    if not event_dates:
        return go.Figure()

    X = np.array(event_dates).reshape(-1, 1)
    bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)
    clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)

    cluster_mapping = {}
    current_cluster_num = 1
    clusters = []
    for cluster in clusters_raw:
        if cluster not in cluster_mapping:
            cluster_mapping[cluster] = current_cluster_num
            current_cluster_num += 1
        clusters.append(cluster_mapping[cluster])

    cumulative_faults = np.arange(1, len(event_dates) + 1)

    points_by_cluster = {}
    for day, cum_fault, cluster in zip(event_dates, cumulative_faults, clusters):
        if cluster not in points_by_cluster:
            points_by_cluster[cluster] = {"x": [], "y": [], "text": []}
        points_by_cluster[cluster]["x"].append(-day)
        points_by_cluster[cluster]["y"].append(cum_fault)
        points_by_cluster[cluster]["text"].append(f"Day {day}")

    unique_clusters = sorted(points_by_cluster.keys())
    colors = px.colors.qualitative.Set1
    cluster_colors = {c: colors[(c - 1) % len(colors)] for c in unique_clusters}

    fig = go.Figure()
    for cluster, data in points_by_cluster.items():
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers+text",
            marker=dict(color=cluster_colors[cluster], size=10),
            text=data["text"],
            textposition="top center",
            name=f"Cluster {cluster}"
        ))

    fig.update_layout(
        title=f"Timeline avec clustering pour l'avion {immat} et la fault {fault}",
        xaxis_title="Jours avant aujourd'hui",
        yaxis_title="Nombre de pannes",
        template="plotly_white"
    )
    return fig

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    xls_new = pd.ExcelFile(io.BytesIO(decoded))

    dfs_cleaned = {}
    for sheet in xls_new.sheet_names:
        df_sheet = xls_new.parse(sheet)
        df_sheet.columns = ["ATA", "IMMAT", "FAULT"] + [f"Day {i}" for i in range(1, df_sheet.shape[1] - 2)]
        df_sheet = df_sheet.iloc[1:].reset_index(drop=True)
        df_sheet.iloc[:, 3:] = df_sheet.iloc[:, 3:].fillna(0)
        dfs_cleaned[sheet] = df_sheet

    df_60J, df_360J = dfs_cleaned["60 J"], dfs_cleaned["360 J"]
    df_60J[["ATA", "IMMAT"]] = df_60J[["IMMAT", "ATA"]]
    df_60J = df_60J.rename(columns={"ATA": "IMMAT", "IMMAT": "ATA"})
    df_360J = df_360J.rename(columns={"ATA": "IMMAT", "IMMAT": "ATA"})

    df_60J = insert_useful_column(df_60J)
    df_360J = insert_useful_column(df_360J)

    df_60J["IMMAT"] = df_60J["IMMAT"].str.strip()
    df_60J["FAULT"] = df_60J["FAULT"].str.strip()
    df_360J["IMMAT"] = df_360J["IMMAT"].str.strip()
    df_360J["FAULT"] = df_360J["FAULT"].str.strip()

    df = df_360J
    df_filtre = df[df["USEFUL"] & (df.filter(like="Day ").astype(bool).sum(axis=1) >= 3)]
    df_final = filter_rows_by_mean_gap(df_filtre, 1.2)
    filtered_df = calculate_recent_occurrences(df_final)

    return df_final.to_json(date_format='iso', orient='split'), filtered_df.to_json(date_format='iso', orient='split')

def generate_row_styles(data):
    styles = []
    for i, row in enumerate(data):
        intensity = int(255 * (1 - row["Color Scale"]))
        color = f"rgb(255, {intensity}, {intensity})"
        styles.append({
            "if": {"row_index": i},
            "backgroundColor": color,
            "color": "black"
        })
    return styles

# --- Initialisation de l'application Dash ---

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Détection de récurrences"),
    # Zone d'upload centrée
    html.Div(
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Excel File')]),
            style={
                'width': '300px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
            multiple=False
        ),
        style={'width': '100%', 'display': 'flex', 'justifyContent': 'center', 'margin': '20px 0'}
    ),
    # Conteneur principal en flex : tableau à gauche, graphique à droite
    html.Div([
        html.Div(
            dcc.Loading(
                id='loading-upload',
                type='default',
                children=html.Div(id='table-container', children=[
                    dash_table.DataTable(id='table', columns=[], data=[])
                ])
            ),
            style={
                'width': '45%',
                'padding': '10px',
                'overflowY': 'auto',
                'height': '600px'
            }
        ),
        html.Div(
            dcc.Graph(id='timeline-plot'),
            style={
                'width': '55%',
                'padding': '10px'
            }
        )
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    dcc.Store(id='processed-data')
])

# --- Callback pour traiter l'upload et afficher le tableau ---

@app.callback(
    [Output('processed-data', 'data'),
     Output('table-container', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return dash.no_update, "Veuillez uploader un fichier Excel."

    df_final_json, filtered_df_json = parse_contents(contents, filename)
    filtered_df = pd.read_json(io.StringIO(filtered_df_json), orient='split')
    records = filtered_df.to_dict('records')

    table = dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in filtered_df.columns if col != "Color Scale"],
        data=records,
        filter_action='native',
        sort_action='native',
        row_selectable='single',
        style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'scroll'},
        style_data_conditional=generate_row_styles(records)
    )

    content = html.Div([table])
    store_data = {'df_final': df_final_json, 'filtered_df': filtered_df_json}
    return store_data, content

# --- Callback combiné pour afficher le graphique ou recentrer selon le clic sur la légende ---

@app.callback(
    Output('timeline-plot', 'figure'),
    [Input('table', 'selected_rows'),
     Input('timeline-plot', 'restyleData')],
    [State('processed-data', 'data'),
     State('timeline-plot', 'figure')]
)
def update_figure(selected_rows, restyleData, processed_data, current_fig):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'table':
        if not selected_rows or processed_data is None:
            return go.Figure()
        df_final = pd.read_json(io.StringIO(processed_data['df_final']), orient='split')
        filtered_df = pd.read_json(io.StringIO(processed_data['filtered_df']), orient='split')
        selected_row = filtered_df.iloc[selected_rows[0]]
        immat, fault = selected_row["IMMAT"], selected_row["FAULT"]
        new_fig = plot_timeline_with_clusters_meanshift_plotly(df_final, immat, fault)
        return new_fig

    elif trigger_id == 'timeline-plot' and restyleData is not None:
        update_info = restyleData[0]
        trace_indices = restyleData[1]
        if 'visible' in update_info:
            new_visible = update_info['visible'][0]
            if new_visible != 'legendonly':
                idx = trace_indices[0]
                if current_fig and 'data' in current_fig and len(current_fig['data']) > idx:
                    x_data = current_fig['data'][idx].get('x', [])
                    y_data = current_fig['data'][idx].get('y', [])
                    if x_data and y_data:
                        xmin = min(x_data)
                        xmax = max(x_data)
                        ymin = min(y_data)
                        ymax = max(y_data)
                        x_margin = (xmax - xmin) * 0.1 if xmax > xmin else 1
                        y_margin = (ymax - ymin) * 0.1 if ymax > ymin else 1
                        new_xrange = [xmin - x_margin, xmax + x_margin]
                        new_yrange = [ymin - y_margin, ymax + y_margin]
                        current_fig['layout']['xaxis']['range'] = new_xrange
                        current_fig['layout']['yaxis']['range'] = new_yrange
        return current_fig

    else:
        return current_fig

# --- Lancement de l'application en mode serveur ---

if __name__ == '__main__':
    # L'application écoute sur 0.0.0.0 et utilise le port défini dans la variable d'environnement PORT (Render.com le définit automatiquement)
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
