import base64
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.cluster import MeanShift, estimate_bandwidth

# --- Fonctions de traitement communes ---

def insert_useful_column(df):
    pannes_a_exclure = [
        "AUTO FLT A/THR OFF", 
        "AUTO FLT AP OFF", 
        "BRAKES HOT", 
        "SURV ROW/ROP LOST", 
        "NAV ALTI DISCREPANCY"
    ]
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
            bandwidth = max(estimate_bandwidth(X, quantile=0.2, n_samples=len(X)),
                            np.std(X)/4, 1.0)
            clusters_raw = MeanShift(bandwidth=bandwidth).fit_predict(X)
            last_cluster = clusters_raw[-1] if len(clusters_raw) > 0 else 0
            recent_occurrences = sum(clusters_raw == last_cluster)
        result.append(recent_occurrences)

    df["Occurrences récentes"] = result
    max_occurrences = max(result) if result else 1
    df["Color Scale"] = df["Occurrences récentes"] / max_occurrences

    columns_to_return = ["row_id", "IMMAT"]
    if "ATA" in df.columns:
        columns_to_return.append("ATA")
    columns_to_return.extend(["FAULT", "Occurrences récentes", "Color Scale"])
    return df[columns_to_return].sort_values(by="Occurrences récentes", ascending=False)

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

# --- Fonctions de traitement spécifiques selon le modèle ---

def process_A320(file_obj):
    xls = pd.ExcelFile(file_obj)
    dfs_cleaned = {}
    for sheet in xls.sheet_names:
        df_sheet = xls.parse(sheet)
        nb_cols = df_sheet.shape[1]
        df_sheet.columns = ["ATA", "IMMAT", "FAULT"] + [f"Day {i}" for i in range(1, nb_cols - 2)]
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

    return df_60J, df_360J

def process_B777(file_obj):
    xls = pd.ExcelFile(file_obj)
    dfs_cleaned = {}
    for sheet in xls.sheet_names:
        df_sheet = xls.parse(sheet)
        df_sheet = df_sheet.iloc[:, 1:]
        nb_cols = df_sheet.shape[1]
        df_sheet.columns = ["IMMAT", "FAULT"] + [f"Day {i}" for i in range(1, nb_cols - 1)]
        df_sheet = df_sheet.iloc[1:].reset_index(drop=True)
        df_sheet.iloc[:, 2:] = df_sheet.iloc[:, 2:].fillna(0)
        dfs_cleaned[sheet] = df_sheet

    df_60J, df_360J = dfs_cleaned["60 J"], dfs_cleaned["360 J"]

    df_60J = insert_useful_column(df_60J)
    df_360J = insert_useful_column(df_360J)

    df_60J["IMMAT"] = df_60J["IMMAT"].str.strip()
    df_60J["FAULT"] = df_60J["FAULT"].str.strip()
    df_360J["IMMAT"] = df_360J["IMMAT"].str.strip()
    df_360J["FAULT"] = df_360J["FAULT"].str.strip()

    return df_60J, df_360J

def parse_contents(contents, filename, facteur):
    # Dans Streamlit, 'contents' est de type bytes.
    decoded = contents
    file_obj = io.BytesIO(decoded)
    
    if "A320" in filename:
        df_60J, df_360J = process_A320(file_obj)
        st.write("Traitement pour A320 réalisé.")
    elif "B777" in filename:
        df_60J, df_360J = process_B777(file_obj)
        st.write("Traitement pour B777 réalisé.")
    else:
        st.error("Le nom du fichier ne correspond à aucun des modèles attendus (A320 ou B777).")
        return None, None

    df = df_360J.copy()
    df["IMMAT"] = df["IMMAT"].str.strip()
    df["FAULT"] = df["FAULT"].str.strip()
    df["row_id"] = df.index

    day_cols = df.filter(like="Day ")
    df_filtre = df[df["USEFUL"] & (day_cols.astype(bool).sum(axis=1) >= 3)]
    df_final = filter_rows_by_mean_gap(df_filtre, facteur)
    filtered_df = calculate_recent_occurrences(df_final)
    
    return df_final, filtered_df

# --- Interface Streamlit ---

st.title("Détection de récurrences")

uploaded_file = st.file_uploader("Sélectionnez un fichier Excel", type=["xlsx"])
facteur = st.number_input("Facteur de filtre :", value=1.5, step=0.1)

if uploaded_file is not None:
    # Lecture du fichier en bytes
    file_bytes = uploaded_file.read()
    df_final, filtered_df = parse_contents(file_bytes, uploaded_file.name, facteur)
    
    if filtered_df is not None:
        st.subheader("Tableau des récurrences")
        # Afficher le tableau (on peut utiliser st.dataframe)
        st.dataframe(filtered_df.drop(columns=["Color Scale"], errors="ignore"))
        
        # Sélection d'une fault à exclure via multiselect
        faults = sorted(filtered_df["FAULT"].unique())
        selected_faults = st.multiselect("Exclure certaines faults :", faults)
        
        if selected_faults:
            filtered_df = filtered_df[~filtered_df["FAULT"].isin(selected_faults)]
            st.dataframe(filtered_df.drop(columns=["Color Scale"], errors="ignore"))
        
        # Sélection d'une ligne via un index numérique (pour simplifier l'interactivité)
        st.subheader("Visualisation du timeline")
        row_index = st.number_input("Sélectionnez l'index de la ligne :", min_value=0, max_value=len(filtered_df)-1, value=0, step=1)
        selected_row = filtered_df.iloc[row_index]
        immat, fault = selected_row["IMMAT"], selected_row["FAULT"]
        
        # Pour tracer le timeline, on relit les données complètes
        df_final_reloaded = pd.read_json(df_final.to_json(orient='split'), orient='split')
        fig = plot_timeline_with_clusters_meanshift_plotly(df_final_reloaded, immat, fault)
        st.plotly_chart(fig, use_container_width=True)
        
        # Bouton de téléchargement en Excel
        st.subheader("Téléchargement")
        download_df = filtered_df.drop(columns=["Color Scale"], errors="ignore")
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            download_df.to_excel(writer, index=False, sheet_name="Filtered Data")
        towrite.seek(0)
        st.download_button(
            label="Télécharger le fichier Excel",
            data=towrite,
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
