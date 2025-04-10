import base64
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from dateutil.parser import parse

# Pour l'application web (Dash et Plotly)
import dash
from dash import dcc, html, dash_table, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURATION DE L'APPLICATION
# =============================================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# =============================================================================
# FONCTIONS UTILITAIRES ET DE TRAITEMENT DES DONNÉES
# =============================================================================
def generate_date_columns(col_headers, day_numbers):
    new_date_cols = []
    for header, day in zip(col_headers, day_numbers):
        if isinstance(header, (int, float)):
            year_str = str(int(header))
        else:
            year_str = str(header).split('.')[0]
        try:
            year = int(year_str)
        except Exception:
            year = 1900
        try:
            day_int = int(day)
        except Exception:
            day_int = 1
        date_val = datetime(year, 1, 1) + timedelta(days=day_int - 1)
        new_date_cols.append(date_val)
    return new_date_cols

def clean_common_columns(df, col_names):
    for col in col_names:
        df[col] = df[col].astype(str).str.strip()
    return df

# =============================================================================
# TRAITEMENT DES FICHIERS EXCEL (Airbus & Boeing)
# =============================================================================
def process_sheet_new_format(file_path, sheet_name, swap_columns=False):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    day_numbers = df_raw.iloc[0, 3:].tolist()
    date_col_headers = df_raw.columns[3:]
    new_date_cols = generate_date_columns(date_col_headers, day_numbers)
    
    df_data = df_raw.iloc[1:, :].reset_index(drop=True)
    if swap_columns:
        cols = df_data.columns.tolist()
        cols[0], cols[1] = cols[1], cols[0]
        df_data = df_data[cols]
    
    new_columns = ["IMMAT", "ATA", "FAULT"] + new_date_cols
    df_data.columns = new_columns
    df_data.iloc[:, 3:] = df_data.iloc[:, 3:].fillna(0)
    df_data = clean_common_columns(df_data, ["IMMAT", "FAULT"])
    df_data["ATA"] = pd.to_numeric(df_data["ATA"], errors="coerce").fillna(0).astype(int)

    pannes_a_exclure = [
        "AUTO FLT A/THR OFF", "AUTO FLT AP OFF", "BRAKES HOT",
        "SURV ROW/ROP LOST", "NAV ALTI DISCREPANCY",
        "NAV GPS 1 FAULT", "NAV GPS 2 FAULT", "NAV GPS1 FAULT", "NAV GPS2 FAULT"
    ]
    df_data.insert(3, 'USEFUL', ~df_data['FAULT'].isin(pannes_a_exclure))
    
    return df_data

def process_airbus_excel_new_format(file_path):
    df_60J = process_sheet_new_format(file_path, sheet_name="60 J", swap_columns=True)
    df_360J = process_sheet_new_format(file_path, sheet_name="360 J", swap_columns=False)
    return df_60J, df_360J

def process_sheet_new_format_boeing(file_path, sheet_name):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    df_raw = df_raw.iloc[:, 1:]
    day_numbers = df_raw.iloc[0, 2:].tolist()
    date_col_headers = df_raw.columns[2:]
    new_date_cols = generate_date_columns(date_col_headers, day_numbers)

    df_data = df_raw.iloc[1:, :].reset_index(drop=True)
    new_columns = ["IMMAT", "FAULT"] + new_date_cols
    df_data.columns = new_columns
    df_data.iloc[:, 2:] = df_data.iloc[:, 2:].fillna(0)
    df_data = clean_common_columns(df_data, ["IMMAT", "FAULT"])
    
    pannes_a_exclure = [
        "AUTO FLT A/THR OFF", "AUTO FLT AP OFF", "BRAKES HOT",
        "SURV ROW/ROP LOST", "NAV ALTI DISCREPANCY",
        "NAV GPS 1 FAULT", "NAV GPS 2 FAULT", "NAV GPS1 FAULT", "NAV GPS2 FAULT"
    ]
    df_data.insert(2, 'USEFUL', ~df_data['FAULT'].isin(pannes_a_exclure))
    
    return df_data

def process_boeing_excel_new_format(file_path):
    df_60J = process_sheet_new_format_boeing(file_path, sheet_name="60 J")
    df_360J = process_sheet_new_format_boeing(file_path, sheet_name="360 J")
    return df_60J, df_360J

# =============================================================================
# FONCTIONS DE FUSION DES DATAFRAMES
# =============================================================================
def merge_60j_into_360j_generic(df_60j, df_360j, key_columns, date_start_index):
    date_cols_60j = df_60j.columns[date_start_index:]
    date_cols_360j = df_360j.columns[date_start_index:]
    missing_date_cols = [col for col in date_cols_60j if col not in date_cols_360j]
    for col in missing_date_cols:
        df_360j[col] = 0
    fixed_cols = df_360j.columns[:date_start_index]
    dynamic_cols = df_360j.columns[date_start_index:]
    sorted_date_cols = sorted(dynamic_cols, key=lambda x: pd.to_datetime(x, errors='coerce'))
    df_360j = df_360j[list(fixed_cols) + sorted_date_cols]
    for _, row_60j in df_60j.iterrows():
        mask = np.logical_and.reduce([df_360j[k] == row_60j[k] for k in key_columns])
        matching_index = df_360j[mask].index
        if matching_index.empty:
            new_row = {}
            for col in fixed_cols:
                new_row[col] = row_60j[col] if col in row_60j else 0
            for col in sorted_date_cols:
                new_row[col] = 0
            for col in date_cols_60j:
                new_row[col] = row_60j[col]
            df_360j = pd.concat([df_360j, pd.DataFrame([new_row])], ignore_index=True)
        else:
            for col in date_cols_60j:
                df_360j.loc[matching_index, col] = row_60j[col]
    df_360j = df_360j[list(fixed_cols) + sorted_date_cols]
    return df_360j

def merge_60j_into_360j(df_60J_airbus, df_360J_airbus):
    return merge_60j_into_360j_generic(df_60J_airbus, df_360J_airbus, key_columns=["IMMAT", "ATA", "FAULT"], date_start_index=4)

def merge_60j_into_360j_boeing(df_60J_boeing, df_360J_boeing):
    return merge_60j_into_360j_generic(df_60J_boeing, df_360J_boeing, key_columns=["IMMAT", "FAULT"], date_start_index=3)

# =============================================================================
# FILTRAGE ET CLUSTERING DES LIGNES
# =============================================================================
def filter_rows_by_mean_gap_dates(df, facteur):
    potential_dates = pd.to_datetime(df.columns, errors='coerce')
    date_cols = df.columns[potential_dates.notna()]
    today = pd.to_datetime("today").normalize()
    filtered_rows = []
    for _, row in df.iterrows():
        event_offsets = []
        for col in date_cols:
            if row[col] != 0:
                col_date = pd.to_datetime(col)
                event_offsets.append((today - col_date).days)
        if not event_offsets:
            continue
        X = np.array(event_offsets).reshape(-1, 1)
        bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)
        ms = MeanShift(bandwidth=bandwidth).fit(X)
        clusters = ms.labels_
        last_event_day = min(event_offsets)
        last_cluster = clusters[np.where(X.flatten() == last_event_day)[0][0]]
        cluster_days = [event_offsets[i] for i in range(len(event_offsets)) if clusters[i] == last_cluster]
        mean_interval = np.mean(np.diff(sorted(cluster_days))) if len(cluster_days) >= 3 else 0
        if facteur * mean_interval > last_event_day:
            filtered_rows.append(row.copy())
    return pd.DataFrame(filtered_rows)

# =============================================================================
# CALCUL DU NOMBRE D'OCCURRENCES RÉCENTES
# =============================================================================
def compute_recent_occurrences(row):
    event_dates = []
    for col in row.index:
        dt = pd.to_datetime(col, format="%Y-%m-%d %H:%M:%S", errors='coerce')
        if pd.notnull(dt) and row[col] != 0:
            event_dates.append(dt)
    if not event_dates:
        return 0
    event_dates.sort()
    X = np.array([d.toordinal() for d in event_dates]).reshape(-1, 1)
    bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)), np.std(X)/4, 1.0)
    clusters = MeanShift(bandwidth=bandwidth).fit_predict(X)
    cluster_mapping = {}
    clusters_new = []
    current_label = 1
    for c in clusters:
        if c not in cluster_mapping:
            cluster_mapping[c] = current_label
            current_label += 1
        clusters_new.append(cluster_mapping[c])
    cluster_to_dates = {}
    for d, cl in zip(event_dates, clusters_new):
        cluster_to_dates.setdefault(cl, []).append(d)
    last_cluster = max(cluster_to_dates, key=lambda cl: max(cluster_to_dates[cl]))
    return clusters_new.count(last_cluster)

def calculate_recent_occurrences(df):
    df = df.copy()
    df["Nombre d'occurrences récentes"] = df.apply(compute_recent_occurrences, axis=1)
    min_val = df["Nombre d'occurrences récentes"].min()
    max_val = df["Nombre d'occurrences récentes"].max()
    denom = (max_val - min_val) if max_val != min_val else 1
    df["Color Scale"] = (df["Nombre d'occurrences récentes"] - min_val) / denom
    columns_to_return = ["IMMAT", "FAULT", "Nombre d'occurrences récentes", "Color Scale"]
    if "ATA" in df.columns:
        columns_to_return.insert(2, "ATA")
    df = df[columns_to_return].drop_duplicates().sort_values(by="Nombre d'occurrences récentes", ascending=False)
    return df

# =============================================================================
# FONCTIONS DE PLOTTING ET STYLE
# =============================================================================
def get_contrasting_text_color(color_str):
    if color_str.startswith('#'):
        hex_color = color_str.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    elif color_str.startswith('rgb'):
        inner = color_str[color_str.find('(')+1:color_str.find(')')]
        r, g, b = [int(p.strip()) for p in inner.split(',')]
    else:
        r, g, b = (0, 0, 0)
    luminance = (0.299*r + 0.587*g + 0.114*b)
    return 'black' if luminance > 186 else 'white'

def plot_timeline_with_clusters_meanshift_plotly(df, immat=None, ata=None, fault=None):
    global df_result
    if "ATA" in df.columns:
        df_selected = df[(df["IMMAT"] == immat) & (df["ATA"] == ata) & (df["FAULT"] == fault)]
    else:
        df_selected = df[(df["IMMAT"] == immat) & (df["FAULT"] == fault)]
    if df_selected.empty:
        return go.Figure(), pd.DataFrame()
    row = df_selected.iloc[0]
    potential_dates = pd.to_datetime(df.columns, format="%Y-%m-%d %H:%M:%S", errors='coerce')
    date_cols = df.columns[potential_dates.notna()]
    event_dates = []
    for col in date_cols:
        val = row[col]
        if pd.notna(val) and val != 0:
            dt = pd.to_datetime(col, format="%Y-%m-%d %H:%M:%S", errors='coerce')
            if pd.notnull(dt):
                event_dates.append(dt)
    if not event_dates:
        return go.Figure(), pd.DataFrame()
    event_dates.sort()
    X = np.array([d.toordinal() for d in event_dates]).reshape(-1, 1)
    bw_est = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))
    bandwidth = max(bw_est, np.std(X)/4, 1.0)
    clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)
    cluster_mapping = {}
    clusters = []
    current_label = 1
    for c in clusters_raw:
        if c not in cluster_mapping:
            cluster_mapping[c] = current_label
            current_label += 1
        clusters.append(cluster_mapping[c])
    cumulative_faults = np.arange(1, len(event_dates)+1)
    points_by_cluster = {}
    for d, cum, cl in zip(event_dates, cumulative_faults, clusters):
        points_by_cluster.setdefault(cl, {"x": [], "y": []})
        points_by_cluster[cl]["x"].append(d)
        points_by_cluster[cl]["y"].append(cum)
    fig = go.Figure()
    unique_clusters = sorted(points_by_cluster.keys())
    base_palette = px.colors.qualitative.Set1
    for i, cl in enumerate(unique_clusters):
        data_clust = points_by_cluster[cl]
        color = base_palette[i % len(base_palette)]
        fig.add_trace(go.Scatter(
            x=data_clust["x"],
            y=data_clust["y"],
            mode="markers",
            marker=dict(color=color, size=10),
            name=f"Cluster {cl}"
        ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='Plainte MTX'
    ))
    fault_cleaned = str(fault).replace(" ", "")
    if "ATA" in df_result.columns:
        matching_df_result = df_result[
            (df_result["IMMAT"] == immat) &
            ((df_result["ATA"] == ata) |
             (df_result["FAULT"].astype(str).str.replace(" ", "").str.contains(fault_cleaned, na=False)))
        ].copy()
    else:
        matching_df_result = df_result[
            (df_result["IMMAT"] == immat) &
            (df_result["FAULT"].astype(str).str.replace(" ", "").str.contains(fault_cleaned, na=False))
        ].copy()
    matching_df_result['FoundDateDT'] = pd.to_datetime(matching_df_result["FoundDate"],
                                                        format="%d/%m/%Y %H:%M", errors='coerce')
    matching_df_result = matching_df_result.sort_values(by="FoundDateDT", ascending=False).reset_index(drop=True)
    matching_df_result.drop(columns="FoundDateDT", inplace=True)
    color_palette = px.colors.qualitative.Set1
    matching_df_result["LineColor"] = [color_palette[i % len(color_palette)] for i in range(len(matching_df_result))]
    for idx, row_res in matching_df_result.iterrows():
        fd = pd.to_datetime(row_res["FoundDate"], format="%d/%m/%Y %H:%M", errors='coerce')
        if pd.notna(fd):
            fig.add_trace(go.Scatter(
                x=[fd, fd],
                y=[0, max(cumulative_faults)],
                mode="lines+markers",
                line=dict(color=row_res["LineColor"], dash="dot"),
                marker=dict(size=20, opacity=0),
                hoverinfo="text",
                hovertext=f"Date: {fd.strftime('%d/%m/%Y %H:%M')}",
                showlegend=False
            ))
    fig.update_layout(
        title="Timeline des occurrences",
        yaxis_title="Numéro de l'occurrence",
        template="plotly_white"
    )
    fig.update_xaxes(type="date")
    table_df = matching_df_result[["FAULT", "FoundDate", "Tasks_Barcode"]]
    return fig, table_df

def generate_row_styles(data):
    styles = []
    # On détermine quelles sont les colonnes de df_final correspondant à des dates
    potential_dates = pd.to_datetime(df_final.columns, errors='coerce')
    date_cols = [col for col, dt in zip(df_final.columns, potential_dates) if pd.notnull(dt)]
    
    for i, row in enumerate(data):
        # Couleur de fond basée sur le score déjà calculé ("Color Scale")
        intensity = int(255*(1-row["Color Scale"]))
        bg_color = f"rgb(255, {intensity}, {intensity})"
        
        # Style de base pour la ligne
        row_style = {
            "if": {"row_index": i},
            "backgroundColor": bg_color,
            "color": "black"
        }
        
        # Récupérer les clés pour filtrer df_final et df_result (IMMAT, FAULT et ATA éventuellement)
        immat = row["IMMAT"]
        fault = row["FAULT"]
        # Vérifier si ATA existe dans la ligne (pour Airbus par exemple)
        ata = row.get("ATA", None)
        
        # Filtrage dans df_final (les dates du plot)
        if ata is not None:
            matching_rows_final = df_final[(df_final["IMMAT"] == immat) &
                                           (df_final["FAULT"] == fault) &
                                           (df_final["ATA"] == ata)]
        else:
            matching_rows_final = df_final[(df_final["IMMAT"] == immat) &
                                           (df_final["FAULT"] == fault)]
        
        last_event_date = None
        if not matching_rows_final.empty:
            # Parcourir les lignes correspondantes pour extraire les dates (colonnes de date non nulles et non zéro)
            event_dates = []
            for idx, row_final in matching_rows_final.iterrows():
                for col in date_cols:
                    # On considère que la valeur non nulle de la cellule signifie la présence d’un événement
                    if row_final[col] != 0:
                        # La colonne représente une date (cf. generate_date_columns)
                        try:
                            dt = pd.to_datetime(col, errors='coerce')
                            if pd.notnull(dt):
                                event_dates.append(dt)
                        except Exception:
                            pass
            if event_dates:
                last_event_date = max(event_dates)
        
        # Filtrage dans df_result (les plaintes MTX)  
        if ata is not None:
            matching_rows_result = df_result[(df_result["IMMAT"] == immat) &
                                             ((df_result["ATA"] == ata) |
                                              (df_result["FAULT"].astype(str).str.replace(" ", "").str.contains(str(fault).replace(" ", ""), na=False)))]
        else:
            matching_rows_result = df_result[(df_result["IMMAT"] == immat) &
                                             (df_result["FAULT"].astype(str).str.replace(" ", "").str.contains(str(fault).replace(" ", ""), na=False))]
        
        last_found_date = None
        if not matching_rows_result.empty:
            # Convertir les FoundDate en datetime pour pouvoir comparer
            matching_rows_result = matching_rows_result.copy()
            matching_rows_result['FoundDateDT'] = pd.to_datetime(matching_rows_result["FoundDate"],
                                                                  format="%d/%m/%Y %H:%M", errors='coerce')
            valid_dates = matching_rows_result['FoundDateDT'].dropna()
            if not valid_dates.empty:
                last_found_date = valid_dates.max()
        
        # Si des lignes de df_result existent ET que la dernière date du plot est supérieure à la dernière FoundDate,
        # on ajoute une bordure rouge.
        if last_event_date and last_found_date and last_event_date > last_found_date:
            row_style["border"] = "2px solid red"
        
        styles.append(row_style)
    return styles


# =============================================================================
# INTERFACE DE CHARGEMENT DES DONNÉES (UPLOAD) - MODIFIÉE
# =============================================================================
app.layout = html.Div([
    html.H1("Détection de pannes récurrentes"),
    html.Div([
        # Colonne pour le type d'avion
        html.Div([
            html.Label("Type d'avion"),
            dcc.Dropdown(
                id='aircraft-type',
                options=[
                    {'label': 'A330', 'value': 'A330'},
                    {'label': 'A320', 'value': 'A320'},
                    {'label': 'B777', 'value': 'B777'}
                ],
                value=None,
                placeholder="Sélectionner un type d'avion"
            )
        ], style={'flex': '1', 'padding': '10px'}),
        # Colonne pour le fichier ECAM
        html.Div([
            html.Label("Fichier ECAM (Excel)"),
            dcc.Upload(
                id='upload-ecam',
                children=html.Div([
                    'Glisser un fichier ici ou cliquer pour téléverser'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'whiteSpace': 'normal',
                    'wordWrap': 'break-word',
                    'margin': '10px 0'
                },
                multiple=False
            ),
            html.Div(id='ecam-status', style={'margin': '5px 0'})
        ], style={'flex': '1', 'padding': '10px'}),
        # Colonne pour le fichier MTX
        html.Div([
            html.Label("Fichier MTX (Excel)"),
            dcc.Upload(
                id='upload-mtx',
                children=html.Div([
                    'Glisser un fichier ici ou cliquer pour téléverser'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'whiteSpace': 'normal',
                    'wordWrap': 'break-word',
                    'margin': '10px 0'
                },
                multiple=False
            ),
            html.Div(id='mtx-status', style={'margin': '5px 0'})
        ], style={'flex': '1', 'padding': '10px'}),
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    }),
    # Bouton d'analyse centré juste en dessous
    html.Div([
        html.Button("Lancer l'analyse", id="launch-analysis", n_clicks=0,
                    style={'padding': '10px 20px', 'fontSize': '16px'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px 0'}),
    # Conteneur du dashboard enveloppé dans un composant dcc.Loading
    dcc.Loading(
       id="loading-animation",
       type="default",  # Vous pouvez choisir "cube", "circle", etc.
       children=[html.Div(id='dashboard-container')]
    )
])



def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return io.BytesIO(decoded)

# =============================================================================
# CALLBACKS POUR CONFIRMER LE TÉLÉVERSEMENT DES FICHIERS
# =============================================================================
@app.callback(
    Output('ecam-status', 'children'),
    Output('upload-ecam', 'style'),
    Input('upload-ecam', 'contents'),
    State('upload-ecam', 'filename')
)
def update_ecam_status(contents, filename):
    if contents is not None and filename is not None:
        style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'solid', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': '#d4edda'
        }
        return f"Fichier reçu: {filename}", style
    else:
        style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px 0'
        }
        return "", style

@app.callback(
    Output('mtx-status', 'children'),
    Output('upload-mtx', 'style'),
    Input('upload-mtx', 'contents'),
    State('upload-mtx', 'filename')
)
def update_mtx_status(contents, filename):
    if contents is not None and filename is not None:
        style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'solid', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': '#d4edda'
        }
        return f"Fichier reçu: {filename}", style
    else:
        style = {
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px 0'
        }
        return "", style

# =============================================================================
# CALLBACK DE TRAITEMENT DES FICHIERS ET AFFICHAGE DU DASHBOARD
# =============================================================================
@app.callback(
    Output('dashboard-container', 'children'),
    [Input('launch-analysis', 'n_clicks')],
    [State('aircraft-type', 'value'),
     State('upload-ecam', 'contents'),
     State('upload-mtx', 'contents')]
)
def handle_files(n_clicks, aircraft_type, ecam_contents, mtx_contents):
    if n_clicks == 0 or not aircraft_type or not ecam_contents or not mtx_contents:
        return html.Div("")
    
    global df, df_result, df_final, filtered_df

    ecam_file = parse_contents(ecam_contents)
    mtx_file = parse_contents(mtx_contents)

    if aircraft_type in ["A330", "A320"]:
        df_60J, df_360J = process_airbus_excel_new_format(ecam_file)
        df_combined = merge_60j_into_360j(df_60J, df_360J)
    else:
        df_60J, df_360J = process_boeing_excel_new_format(ecam_file)
        df_combined = merge_60j_into_360j_boeing(df_60J, df_360J)
    
    df = df_combined.drop_duplicates()
    potential_dates = pd.to_datetime(df.columns, errors='coerce')
    date_cols = df.columns[potential_dates.notna()]
    df_filtre = df[(df[date_cols].ne(0).sum(axis=1) >= 3) & (df["USEFUL"] == True)]
    df_final = filter_rows_by_mean_gap_dates(df_filtre, facteur=2)

    df_mtx = pd.read_excel(mtx_file, sheet_name="Data")
    for col in df_mtx.select_dtypes(include=['object']):
        df_mtx[col] = df_mtx[col].str.strip()
    df_mtx["ATA"] = pd.to_numeric(df_mtx["ATA"], errors="coerce")
    df_mtx["ATA"] = df_mtx["ATA"].astype("Int64")
    df_mtx = df_mtx.rename(columns={"Tasks_Aircraft": "IMMAT", "Task_Name_Only": "FAULT"})
    df_result = df_mtx.copy()

    filtered_df = calculate_recent_occurrences(df_final)

    dashboard_layout = html.Div([
        html.H2("Dashboard chargé"),
        html.Div([
            html.Div([
                html.Details([
                    html.Summary("Sélectionner les faults"),
                    # Par défaut le menu n'est pas déroulé
                    dcc.Checklist(
                        id="remove-faults-option",
                        options=[{"label": "Retirer les pannes concernant plus de 3 avions", "value": "remove"}],
                        value=["remove"],
                        labelStyle={"display": "block"}
                    ),
                    html.Div([
                        html.Button("Tout cocher", id="select-all", n_clicks=0),
                        html.Button("Tout décocher", id="deselect-all", n_clicks=0)
                    ], style={'display': 'flex', 'gap': '10px', 'margin': '5px 0'}),
                    dcc.Checklist(
                        id="fault-filter",
                        options=[{"label": f, "value": f} for f in sorted(filtered_df["FAULT"].unique())],
                        value=sorted(filtered_df["FAULT"].unique()),
                        labelStyle={"display": "block"}
                    )
                ], open=False),
                dash_table.DataTable(
                    id='table',
                    columns=[{'name': c, 'id': c} for c in filtered_df.columns if c != "Color Scale"],
                    data=filtered_df.to_dict('records'),
                    filter_action='native',
                    sort_action='native',
                    row_selectable='single',
                    style_table={'overflowX': 'auto', 'maxHeight': '600px', 'overflowY': 'scroll'},
                    style_data_conditional=generate_row_styles(filtered_df.to_dict('records'))
                )
            ], style={'width': '30%', 'padding': '10px', 'overflowY': 'auto'}),
            html.Div([
                dcc.Graph(id='timeline-plot'),
                html.Hr(),
                html.H3("Plaintes MTX"),
                dash_table.DataTable(
                    id='df-result-table',
                    columns=[{'name': c, 'id': c} for c in ["FAULT", "FoundDate", "Tasks_Barcode"]],
                    data=[],
                    style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'scroll'},
                    style_cell={'textAlign': 'left'},
                )
            ], style={'width': '70%', 'padding': '10px'})
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
    return dashboard_layout

# =============================================================================
# CALLBACKS DU DASHBOARD (Mise à jour table, graphique, etc.)
# =============================================================================
@app.callback(
    [Output('table', 'data'),
     Output('table', 'style_data_conditional'),
     Output('fault-filter', 'options'),
     Output('fault-filter', 'value')],
    [Input('fault-filter', 'value'),
     Input('remove-faults-option', 'value'),
     Input('select-all', 'n_clicks'),
     Input('deselect-all', 'n_clicks')],
    State('fault-filter', 'options')
)
def update_table_and_faults(selected_faults, remove_option, select_all, deselect_all, current_options):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    df_table = filtered_df.copy()
    if "remove" in (remove_option or []):
        fault_counts = df_table.groupby("FAULT")["IMMAT"].nunique()
        faults_to_exclude = fault_counts[fault_counts > 3].index
        df_table = df_table[~df_table["FAULT"].isin(faults_to_exclude)]
    available_faults = sorted(df_table["FAULT"].unique())
    new_options = [{"label": f, "value": f} for f in available_faults]
    if trigger_id == 'select-all':
        new_selected = available_faults
    elif trigger_id == 'deselect-all':
        new_selected = []
    else:
        new_selected = [f for f in (selected_faults or []) if f in available_faults]
        if not new_selected:
            new_selected = available_faults
    if new_selected:
        df_table = df_table[df_table["FAULT"].isin(new_selected)]
    else:
        df_table = pd.DataFrame(columns=filtered_df.columns)
    records = df_table.to_dict('records')
    new_styles = generate_row_styles(records)
    return records, new_styles, new_options, new_selected

@app.callback(
    [Output('timeline-plot', 'figure'),
     Output('df-result-table', 'data'),
     Output('df-result-table', 'style_data_conditional')],
    [Input('table', 'selected_rows'),
     Input('timeline-plot', 'restyleData')],
    [State('table', 'data'),
     State('timeline-plot', 'figure')]
)
def display_plot(selected_rows, restyleData, table_data, current_fig):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'table':
        if not selected_rows:
            return go.Figure(), [], []
        selected_row = table_data[selected_rows[0]]
        immat = selected_row["IMMAT"]
        ata = selected_row.get("ATA", None)
        fault = selected_row["FAULT"]
        new_fig, result_df = plot_timeline_with_clusters_meanshift_plotly(df_final, immat=immat, ata=ata, fault=fault)
        result_df['FoundDateDT'] = pd.to_datetime(result_df['FoundDate'], format="%d/%m/%Y %H:%M", errors='coerce')
        result_df = result_df.sort_values(by='FoundDateDT', ascending=False)
        result_df.drop(columns='FoundDateDT', inplace=True)
        table_result_data = result_df.to_dict('records')
        style_result = []
        color_palette = px.colors.qualitative.Set1
        for i in range(len(result_df)):
            bg = color_palette[i % len(color_palette)]
            txt = get_contrasting_text_color(bg)
            style_result.append({
                'if': {'row_index': i},
                'backgroundColor': bg,
                'color': txt
            })
        return new_fig, table_result_data, style_result
    elif trigger_id == 'timeline-plot' and restyleData is not None:
        update_info = restyleData[0]
        trace_indices = restyleData[1]
        if 'visible' in update_info:
            new_visible = update_info['visible'][0]
            if new_visible != 'legendonly':
                idx = trace_indices[0]
                if current_fig and 'data' in current_fig and len(current_fig['data']) > idx:
                    x_data = current_fig['data'][idx].get('x', [])
                    parsed_dates = []
                    for val in x_data:
                        try:
                            dt = parse(val)
                            parsed_dates.append(dt)
                        except:
                            pass
                    if parsed_dates:
                        x_min = min(parsed_dates)
                        x_max = max(parsed_dates)
                        delta_days = (x_max - x_min).days
                        x_margin_days = delta_days * 0.1 if delta_days > 0 else 1
                        current_fig['layout']['xaxis']['range'] = [
                            (x_min - pd.Timedelta(days=x_margin_days)).isoformat(),
                            (x_max + pd.Timedelta(days=x_margin_days)).isoformat()
                        ]
        return current_fig, dash.no_update, dash.no_update
    else:
        return current_fig, dash.no_update, dash.no_update

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
