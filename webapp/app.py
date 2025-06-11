# app.py
import os
import time
import random
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from trackmania_rl.replay_utils import check_replay_vs_map, get_replay_and_map_data

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

engine = create_engine(f"sqlite:///{DB_PATH}")
if not DB_PATH.exists():
    Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ------------------------------------------------------------------------------------
#  Tableau des cartes (remplacer par une vraie BDD plus tard)
# ------------------------------------------------------------------------------------
MAPS = pd.DataFrame({
    "short": ["A01", "ESL-Hock", "Easy-Race"],
    "display_name": [
        "A01 – White Series Race",
        "ESL – Hockolicious",
        "Easy Race (community)"
    ],
    "vcp_file": [
        "A01_0.5m_.npy",
        "ESL-Hockolicious_0.5m_cl2.npy",
        "EASY-Race_0.5m_cl.npy"
    ],
})

# ------------------------------------------------------------------------------------
#  Interface Streamlit
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="TrackMania Coach AI", layout="wide")
st.title("🏎️ StatMania Coach AI – Analyse de Replay")

# 1) Sélection de la carte
map_choice = st.sidebar.selectbox("1️⃣ Choisir la carte", MAPS["display_name"])
map_row    = MAPS.loc[MAPS["display_name"] == map_choice].iloc[0]
short      = map_row.short

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
            st.subheader("📊 Comparaison des métriques clés")
            metrics = {
                "Temps total (s)": [
                    player_data["race_time_s"],
                    agent_data["race_time_s"] if agent_data else None,
                ],
                "Respawns": [
                    player_data["num_respawns"],
                    agent_data["num_respawns"] if agent_data else None,
                ],
                "Stunts score": [
                    player_data.get("stunts_score", 0),
                    agent_data.get("stunts_score", 0) if agent_data else None,
                ],
            }
            df_metrics = pd.DataFrame(metrics, index=["Joueur", "Agent"])
            st.bar_chart(df_metrics)

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
    st.line_chart(df.set_index("upload_time")["race_time"] / 1000)

st.caption("🔧 Prototype – adaptez ces graphiques à vos données réelles !")
