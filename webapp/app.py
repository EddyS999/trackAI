# app.py
import os
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import plotly.express as px

from trackmania_rl.replay_utils import check_replay_vs_map, get_replay_and_map_data, load_ghost_positions
from webapp.stats_functions import *


# ------------------------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent
REPLAY_DIR = DATA_DIR / "replays"
DB_PATH    = DATA_DIR / "coach.sqlite"

for d in (DATA_DIR, REPLAY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------
#  Définition de la BDD et création conditionnelle de la table
# ------------------------------------------------------------------------------------
Base = declarative_base()

class Replay(Base):
    __tablename__ = "replays"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user        = Column(String,   nullable=False)
    map_short   = Column(String,   nullable=False)
    filename    = Column(String,   nullable=False)
    race_time   = Column(Integer)  # en ms
    upload_time = Column(DateTime, default=datetime.utcnow)



class Map(Base):
    __tablename__ = "maps"
    short       = Column(String, primary_key=True)      # ex. A01
    display_name = Column(String, nullable=False)       # ex. "A01 – White Series Race"
    vcp_file     = Column(String, nullable=False)       # ex. "A01_0.5m_.npy"


engine = create_engine(f"sqlite:///{DB_PATH}")

Base.metadata.create_all(engine, checkfirst=True)  # Crée la table si elle n'existe pas
Session = sessionmaker(bind=engine)

def seed_maps(session):
    """Insère les cartes de base si la table est vide."""
    if session.query(Map).count() == 0:
        seed = [
            {"short": "A01", "display_name": "A01 – White Series Race", "vcp_file": "A01_0.5m_.npy"},
            {"short": "A02", "display_name": "A02 - White Series Race",  "vcp_file": "A02_0.5m_.npy"},
            {"short": "ESL-Hock", "display_name": "ESL – Hockolicious",  "vcp_file": "ESL-Hockolicious_0.5m_cl2.npy"},
        ]
        session.bulk_insert_mappings(Map, seed)
        session.commit()

# Exécution au lanceme

# --------------------------------------------------------
# Récupération des cartes depuis la BDD
# --------------------------------------------------------
with Session() as db:
    seed_maps(db)  # Assure que les cartes de base sont présentes
    maps = db.query(Map).order_by(Map.short).all()

# On prépare un dict « display_name → objet Map » pour accéder facilement aux champs
MAP_CHOICES = {m.display_name: m for m in maps}



# ------------------------------------------------------------------------------------
#  Interface Streamlit
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="TrackMania Coach AI", layout="wide")
st.title("🏎️ StatMania Coach AI – Analyse de Replay")

# 1) Sélection de la carte
map_choice = st.sidebar.selectbox("1️⃣ Choisir la carte", list(MAP_CHOICES.keys()))
map_row    = MAP_CHOICES[map_choice]
short      = map_row.short
# map_choice = st.sidebar.selectbox("1️⃣ Choisir la carte", MAPS["display_name"])
# map_row    = MAPS.loc[MAPS["display_name"] == map_choice].iloc[0]
# short      = map_row.short

st.sidebar.markdown(f"**Carte :** {map_row.display_name}")
st.sidebar.markdown(f"**Short :** `{short}`")

# 2) Upload du replay du joueur
st.sidebar.header("2️⃣ Ajouter ton replay")
uploaded = st.sidebar.file_uploader("Glisse ton `.Replay.Gbx` ici", type=["gbx"])
user     = st.sidebar.text_input("Ton pseudo", value="player1")
run_it   = st.sidebar.button("✅ Analyser", disabled=(uploaded is None))

if run_it:
    # ← Sauvegarde du replay
    replay_name = f"{user}_{short}_{int(time.time())}.Replay.Gbx"
    replay_path = REPLAY_DIR / replay_name
    with open(replay_path, "wb") as f:
        f.write(uploaded.read())

    # ← 3) Vérification géométrique
    vcp_path = DATA_DIR / "maps" / short / "vcp" / map_row.vcp_file

    if not vcp_path.exists():
        st.error(
            f"❌ Fichier VCP introuvable :\n`{vcp_path}`\n"
            "As-tu bien généré le `.npy` avec `gbx_to_vcp` et placé le fichier "
            f"dans `webapp/maps/{short}/vcp/` ?"
        )
    else:
        try:
            ok = check_replay_vs_map(str(replay_path), str(vcp_path), max_avg_dist=100.0)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de la VCP : {e}")
            ok = False

        if not ok:
            st.error("⚠️ Ce replay ne correspond pas à la carte sélectionnée.")
            exit(0)
        else:
            st.success("✅ Le replay correspond bien à la carte !")

            # ← 4) Extraction des données du joueur
            player_data = get_replay_and_map_data(str(replay_path))

            # ← 5) Charger le replay de référence (agent)
            ref_dir   = DATA_DIR / "maps" / short / "reference_run"
            ref_files = list(ref_dir.glob("*.Replay.Gbx"))
            if ref_files:
                agent_data = get_replay_and_map_data(str(ref_files[0]))
            else:
                agent_data = None
                st.warning("⚠️ Pas de replay de référence trouvé pour cette map.")

            # ← 6) Ajout de graphiques pour comparaison

            # 6a) Bar chart des métriques principales
            st.subheader("🏆 Analyse de Performance vs Agent")

        # Calcul des métriques de performance
        time_diff = player_data["race_time_s"] - agent_data["race_time_s"] if agent_data else 0
        respawn_diff = player_data["num_respawns"] - agent_data["num_respawns"] if agent_data else 0
        stunts_diff = player_data.get("stunts_score", 0) - agent_data.get("stunts_score", 0) if agent_data else 0


        if agent_data:
            st.session_state.analysis_data = {
                'player_data': player_data,
                'agent_data': agent_data,
                'vcp_path': str(vcp_path),
                'replay_path': str(replay_path),
                'ref_files': ref_files,
                'analysis_ready': True
            }
        
        # Métriques en colonnes avec indicateurs visuels
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="⏱️ Temps de course",
                value=f"{player_data['race_time_s']/1000.0:.2f}s",
                delta=f"{time_diff:.3f}s" if agent_data else None,
                delta_color="inverse"
            )
            if agent_data:
                st.caption(f"🤖 Agent: {agent_data['race_time_s']:.3f}s")

        with col2:
            st.metric(
                label="🔄 Respawns",
                value=player_data['num_respawns'],
                delta=respawn_diff if agent_data else None,
                delta_color="inverse"
            )
            if agent_data:
                st.caption(f"🤖 Agent: {agent_data['num_respawns']}")

        with col3:
            st.metric(
                label="🎪 Score Stunts",
                value=player_data.get('stunts_score', 0),
                delta=stunts_diff if agent_data else None
            )
            if agent_data:
                st.caption(f"🤖 Agent: {agent_data.get('stunts_score', 0)}")

        # Graphique de comparaison horizontal avec pourcentages
        if agent_data:
            st.subheader("📊 Comparaison détaillée")
            
            # Création du graphique avec Plotly
            fig_comparison = go.Figure()
            
            categories = ['Temps de course', 'Respawns', 'Score Stunts']
            player_values = [
                player_data["race_time_s"],
                player_data["num_respawns"],
                player_data.get("stunts_score", 0)
            ]
            agent_values = [
                agent_data["race_time_s"],
                agent_data["num_respawns"],
                agent_data.get("stunts_score", 0)
            ]
            
            # Ajout des barres
            fig_comparison.add_trace(go.Bar(
                name='👤 Joueur',
                y=categories,
                x=player_values,
                orientation='h',
                marker_color='#FF6B6B',
                text=[f"{v:.3f}s" if i == 0 else str(v) for i, v in enumerate(player_values)],
                textposition='auto',
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='🤖 Agent',
                y=categories,
                x=agent_values,
                orientation='h',
                marker_color='#4ECDC4',
                text=[f"{v:.3f}s" if i == 0 else str(v) for i, v in enumerate(agent_values)],
                textposition='auto',
            ))
            
            fig_comparison.update_layout(
                title="Performance Joueur vs Agent",
                xaxis_title="Valeur",
                barmode='group',
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Analyse et conseils personnalisés
            st.subheader("🎯 Analyse et Conseils d'Amélioration")
            
            # Calcul du pourcentage de différence
            time_percent = ((time_diff / agent_data["race_time_s"]) * 100) if agent_data["race_time_s"] > 0 else 0
            
            # Conseils basés sur les métriques
            advice_col1, advice_col2 = st.columns(2)
            
            with advice_col1:
                st.markdown("### 🏁 Performance Globale")
                if time_percent <= 5:
                    st.success("🏆 **Excellent !** Tu es très proche du niveau de l'agent !")
                elif time_percent <= 15:
                    st.info("👍 **Bien joué !** Tu as une performance solide, quelques ajustements suffisent.")
                elif time_percent <= 30:
                    st.warning("⚡ **Potentiel d'amélioration** : Focus sur la régularité et la technique.")
                else:
                    st.error("🎯 **Beaucoup de marge** : Travaille les bases, la trajectoire et la gestion de vitesse.")
            
            with advice_col2:
                st.markdown("### 🔧 Conseils Spécifiques")
                
                # Conseil sur les respawns
                if respawn_diff > 0:
                    st.markdown(f"🔄 **Respawns (+{respawn_diff})** : Travaille la précision dans les passages techniques")
                elif respawn_diff == 0:
                    st.markdown("🔄 **Respawns** : Même stabilité que l'agent, bien joué !")
                else:
                    st.markdown("🔄 **Respawns** : Moins de respawns que l'agent, excellente régularité !")
                
                # Conseil sur les stunts
                if stunts_diff < 0:
                    st.markdown(f"🎪 **Stunts ({stunts_diff})** : Prends plus de risques pour augmenter ton score")
                elif stunts_diff > 0:
                    st.markdown(f"🎪 **Stunts (+{stunts_diff})** : Bon style ! Maintiens ce niveau")
                
                # Conseil sur le temps
                if time_diff > 10:
                    st.markdown("⏱️ **Temps** : Focus sur l'optimisation des trajectoires et la gestion de vitesse")
                elif time_diff > 5:
                    st.markdown("⏱️ **Temps** : Travaille les détails : freinages, trajectoires en courbe")
                elif time_diff > 0:
                    st.markdown("⏱️ **Temps** : Très proche ! Peaufine les micro-optimisations")
                else:
                    st.markdown("⏱️ **Temps** : Tu bats l'agent ! Incroyable performance !")

        # Graphique radar pour vue d'ensemble (optionnel si vous voulez l'ajouter)
        if agent_data:
            st.subheader("🕸️ Profil de Performance (Radar)")
            
            # Normalisation des données pour le radar
            max_time = max(player_data["race_time_s"], agent_data["race_time_s"])
            max_respawns = max(player_data["num_respawns"], agent_data["num_respawns"]) if agent_data["num_respawns"] > 0 else 1
            max_stunts = max(player_data.get("stunts_score", 0), agent_data.get("stunts_score", 0)) if agent_data.get("stunts_score", 0) > 0 else 1
            
            # Calcul des scores normalisés (0-100)
            player_speed = 100 - (player_data["race_time_s"] / max_time * 100)  # Vitesse (temps inversé)
            agent_speed = 100 - (agent_data["race_time_s"] / max_time * 100)
            
            player_stability = 100 - (player_data["num_respawns"] / max_respawns * 100)  # Stabilité (respawns inversés)
            agent_stability = 100 - (agent_data["num_respawns"] / max_respawns * 100)
            
            player_style = (player_data.get("stunts_score", 0) / max_stunts * 100)  # Style (stunts)
            agent_style = (agent_data.get("stunts_score", 0) / max_stunts * 100)
            
            categories_radar = ['Vitesse', 'Stabilité', 'Style']
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[player_speed, player_stability, player_style],
                theta=categories_radar,
                fill='toself',
                name='👤 Joueur',
                line_color='#FF6B6B'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[agent_speed, agent_stability, agent_style],
                theta=categories_radar,
                fill='toself',
                name='🤖 Agent',
                line_color='#4ECDC4'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                height=400
            )
    
            st.plotly_chart(fig_radar, use_container_width=True)




            # st.subheader("📊 Comparaison des métriques clés")
            # metrics = {
            #     "Temps total (s)": [
            #         player_data["race_time_s"],
            #         agent_data["race_time_s"] if agent_data else None,
            #     ],
            #     "Respawns": [
            #         player_data["num_respawns"],
            #         agent_data["num_respawns"] if agent_data else None,
            #     ],
            #     "Stunts score": [
            #         player_data.get("stunts_score", 0),
            #         agent_data.get("stunts_score", 0) if agent_data else None,
            #     ],
            # }
            # df_metrics = pd.DataFrame(metrics, index=["Joueur", "Agent"])
            # st.bar_chart(df_metrics)

            # 6b) Courbe des temps aux checkpoints
            if agent_data:
                st.subheader("⏱️ Temps aux checkpoints (s)")
                cp_p = player_data["cp_times_ms"]
                cp_a = agent_data["cp_times_ms"]
                max_n = max(len(cp_p), len(cp_a))
                cp_p += [None] * (max_n - len(cp_p))
                cp_a += [None] * (max_n - len(cp_a))
                df_cp = pd.DataFrame({
                    "Joueur": pd.Series(cp_p),
                    "Agent":  pd.Series(cp_a),
                })
                df_cp.index.name = "Checkpoint #"
                st.line_chart(df_cp / 1000.0)

            # 6c) Heatmap ou distribution des vitesses (optionnel)
            # … à ajouter ici si vous avez les données de speed

            # 6d) Trajectoire superposée sur plan 2D

            display_trajectory_analysis(vcp_path, replay_path ,ref_files)

        elif hasattr(st.session_state, "analysis_data") and st.session_state.analysis_data.get("analysis_ready"):
            data = st.session_state.analysis_data
            display_trajectory_analysis(data['vcp_path'], data['replay_path'], data['ref_files'])

            # st.subheader("📍 Trajectoire sur la map (vue X–Z)")
            # vcp = np.load(vcp_path)          # shape (M,3)
            #     # charger positions joueur / agent
            # pos_j = load_ghost_positions(replay_path)[:, [0,2]]

            # if ref_files:
            #     pos_a = load_ghost_positions(str(ref_files[0]))[:, [0,2]]
            # else:
            #     pos_a = None
            
            # fig = go.Figure()

            # fig.add_trace(go.Scatter(
            #     x=vcp[:,0], y=vcp[:,2],
            #     mode="lines",
            #     line=dict(color="gray", width=2),
            #     name="VCP réf. agent",
            # ))

            # fig.add_trace(go.Scatter(
            #     x=pos_j[:,0], y=pos_j[:,1],
            #     mode="lines",
            #     line=dict(color="red", width=2),
            #     name="Traj. Joueur",
            #     hovertemplate="x: %{x:.1f}  z: %{y:.1f}<extra></extra>",
            # ))

            # if pos_a is not None:
            #     fig.add_trace(go.Scatter(
            #     x=pos_a[:,0], y=pos_a[:,1],
            #     mode="lines",
            #     line=dict(color="blue", width=2, dash="dash"),
            #     name="Traj. Agent",
            # ))

            # fig.update_layout(
            #     xaxis_title="X (m)",
            #     yaxis_title="Z (m)",
            #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            #     margin=dict(l=20, r=20, t=30, b=20),
            #     width=600,
            #     height=600,
            # )
            # fig.update_yaxes(scaleanchor="x", scaleratio=1)  # rapport 1:1

            # st.plotly_chart(fig, use_container_width=True)


            # pos_a = load_ghost_positions(str(ref_files[0]))[:, [0,2]]

            # fig, ax = plt.subplots(figsize=(6,6))
            #     # si tu as un screenshot : ax.imshow(img, extent=[xmin,xmax,zmin,zmax], alpha=0.5)
            # ax.plot(vcp[:,0], vcp[:,2], color='gray', lw=2, label='Réf. agent')
            # ax.plot(pos_j[:,0], pos_j[:,1], color='red',  lw=1, label='Joueur')
            # ax.plot(pos_a[:,0], pos_a[:,1], color='blue', lw=1, label='Agent')
            # ax.set_aspect('equal', 'box')
            # ax.legend(loc='upper right')
            # st.pyplot(fig)

            # ← 7) Affichage textuel résumé
            st.header("🏁 Résultats détaillés")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ton replay")
                for k, v in player_data.items():
                    st.markdown(f"- **{k} :** {v}")
            with col2:
                st.subheader("Référence agent")
                if agent_data:
                    for k, v in agent_data.items():
                        st.markdown(f"- **{k} :** {v}")
                    delta = player_data["race_time_s"] - agent_data["race_time_s"]
                    st.markdown(f"**Différence** (joueur − agent) : {delta:.3f} s")

            # ← 8) Enregistrement en BDD
            with Session() as db:
                rec = Replay(
                    user      = user,
                    map_short = short,
                    filename  = str(replay_path),
                    race_time = player_data["race_time_s"]
                )
                db.add(rec)
                db.commit()

# 9) Historique des runs
st.header("📊 Historique de tes runs")
with Session() as db:
    df = pd.read_sql_table("replays", engine)
    df = df.query("user == @user and map_short == @short")

if df.empty:
    st.info("Tu n'as pas encore déposé de replay pour cette map.")
else:
    df["upload_time"] = pd.to_datetime(df["upload_time"])
    df = df.sort_values("upload_time")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df["upload_time"],
        y=df["race_time"] / 1000,
        mode="lines+markers",
        name="Temps de run"
    ))
    fig_hist.update_layout(
        xaxis_title="Date d'ajout",
        yaxis_title="Temps (s)",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.caption("🔧 Prototype - v.0.0.1")