import base64
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table, State
from sklearn.cluster import MeanShift, estimate_bandwidth

# --- Fonctions utilitaires ---

def compute_bandwidth(X):
    """Calcule la largeur de bande pour MeanShift en tenant compte de l'écart type."""
    return max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)

# --- Fonctions de traitement des données ---

def insert_useful_column(df):
    pannes_a_exclure = [
        "AUTO FLT A/THR OFF", "AUTO FLT AP OFF",
        "BRAKES HOT", "SURV ROW/ROP LOST", "NAV ALTI DISCREPANCY"
    ]
    fault_idx = df.columns.get_loc('FAULT')
    df.insert(fault_idx + 1, 'USEFUL', ~df['FAULT'].isin(pannes_a_exclure))
    return df

def filter_rows_by_mean_gap(df, facteur):
    day_cols = [col for col in df.columns if col.startswith("Day ")]
    filtered_rows = []
    for _, row in df.iterrows():
        event_dates = sorted(
            [int(col.split()[1]) for col in day_cols if row[col] != 0],
            reverse=True
        )
        if not event_dates:
            continue

        X = np.array(event_dates).reshape(-1, 1)
        bandwidth = compute_bandwidth(X)
        clusters = MeanShift(bandwidth=bandwidth).fit(X).labels_
        last_event_day = min(event_dates)
        idx_last = np.where(np.array(event_dates) == last_event_day)[0][0]
        last_cluster = clusters[idx_last]
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
    day_cols = [col for col in df.columns if col.startswith("Day ")]
    occurrences = []
    for _, row in df.iterrows():
        event_dates = sorted(
            [int(col.split()[1]) for col in day_cols if row[col] != 0],
            reverse=True
        )
        if not event_dates:
            occurrences.append(0)
        else:
            X = np.array(event_dates).reshape(-1, 1)
            bandwidth = compute_bandwidth(X)
            clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)
            last_cluster = clusters_raw[-1] if len(clusters_raw) > 0 else 0
            occurrences.append(int(np.sum(clusters_raw == last_cluster)))
    df["Occurrences récentes"] = occurrences
    max_occ = max(occurrences) if occurrences else 1
    df["Color Scale"] = df["Occurrences récentes"] / max_occ
    return df[["IMMAT", "ATA", "FAULT", "Occurrences récentes", "Color Scale"]].sort_values(
        by="Occurrences récentes", ascending=False
    )

def plot_timeline_with_clusters_meanshift_plotly(df, immat=None, fault=None):
    df_selected = df[(df["IMMAT"] == immat) & (df["FAULT"] == fault)]
    if df_selected.empty:
        return go.Figure()
    
    day_cols = [col for col in df.columns if col.startswith("Day ")]
    event_dates = sorted(
        [int(col.split()[1]) for col in day_cols if df_selected.iloc[0][col] != 0],
        reverse=True
    )
    if not event_dates:
        return go.Figure()

    X = np.array(event_dates).reshape(-1, 1)
    bandwidth = compute_bandwidth(X)
    clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)

    # Remappage des clusters pour un affichage séquentiel
    cluster_mapping = {}
    clusters = []
    for cluster in clusters_raw:
        if cluster not in cluster_mapping:
            cluster_mapping[cluster] = len(cluster_mapping) + 1
        clusters.append(cluster_mapping[cluster])

    cumulative_faults = np.arange(1, len(event_dates) + 1)
    points_by_cluster = {}
    for day, cum_fault, cluster in zip(event_dates, cumulative_faults, clusters):
        points_by_cluster.setdefault(cluster, {"x": [], "y": [], "text": []})
        points_by_cluster[cluster]["x"].append(-day)
        points_by_cluster[cluster]["y"].append(cum_fault)
        points_by_cluster[cluster]["text"].append(f"Day {day}")

    colors = px.colors.qualitative.Set1
    cluster_colors = {c: colors[(c - 1) % len(colors)] for c in sorted(points_by_cluster.keys())}

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
    """
    Traite uniquement la feuille "360 J" et ignore la feuille "60 J".
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    xls = pd.ExcelFile(io.BytesIO(decoded))
    # Utilisation exclusive de la feuille "360 J"
    df = xls.parse("360 J")
    nb_cols = df.shape[1]
    # Les 3 premières colonnes sont fixes, les suivantes correspondent aux jours
    df.columns = ["ATA", "IMMAT", "FAULT"] + [f"Day {i}" for i in range(1, nb_cols - 2)]
    df = df.iloc[1:].reset_index(drop=True)
    df.iloc[:, 3:] = df.iloc[:, 3:].fillna(0)
    df = df.rename(columns={"ATA": "IMMAT", "IMMAT": "ATA"})
    df = insert_useful_column(df)
    df["IMMAT"] = df["IMMAT"].str.strip()
    df["FAULT"] = df["FAULT"].str.strip()

    df_filtre = df[df["USEFUL"] & (df.filter(like="Day ").astype(bool).sum(axis=1) >= 3)]
    df_final = filter_rows_by_mean_gap(df_filtre, 1.2)
    filtered_df = calculate_recent_occurrences(df_final)

    return df_final.to_json(date_format='iso', orient='split'), filtered_df.to_json(date_format='iso', orient='split')

def generate_row_styles(data):
    return [
        {
            "if": {"row_index": i},
            "backgroundColor": f"rgb(255, {int(255 * (1 - row['Color Scale']))}, {int(255 * (1 - row['Color Scale']))})",
            "color": "black"
        }
        for i, row in enumerate(data)
    ]

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
            style={'width': '45%', 'padding': '10px', 'overflowY': 'auto', 'height': '600px'}
        ),
        html.Div(
            dcc.Graph(id='timeline-plot'),
            style={'width': '55%', 'padding': '10px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    dcc.Store(id='processed-data')
])

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
    return {'df_final': df_final_json, 'filtered_df': filtered_df_json}, html.Div([table])

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
        return plot_timeline_with_clusters_meanshift_plotly(df_final, immat, fault)
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
                        xmin, xmax = min(x_data), max(x_data)
                        ymin, ymax = min(y_data), max(y_data)
                        x_margin = (xmax - xmin) * 0.1 if xmax > xmin else 1
                        y_margin = (ymax - ymin) * 0.1 if ymax > ymin else 1
                        current_fig['layout']['xaxis']['range'] = [xmin - x_margin, xmax + x_margin]
                        current_fig['layout']['yaxis']['range'] = [ymin - y_margin, ymax + y_margin]
        return current_fig
    return current_fig

if __name__ == '__main__':
    # L'application écoute sur 0.0.0.0 et utilise le port défini dans la variable d'environnement PORT
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
