# streamlit_app.py
# -----------------
# Prototype minimal de coach TrackMania avec Streamlit
# Auteur : vous ✨  (libre d’en faire ce que vous voulez)

import os
import io
import time
import random
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ------------------------------------------------------------------------------------
#  Configuration & utilitaires
# ------------------------------------------------------------------------------------
DATA_DIR = Path("./webapp/")
REPLAY_DIR = DATA_DIR / "replays"
DB_PATH = DATA_DIR / "coach.sqlite"
REFERENCE_DIR = DATA_DIR / "reference_runs"  # contiendra les fantômes / stats de l’agent

for p in [DATA_DIR, REPLAY_DIR, REFERENCE_DIR]:
    p.mkdir(exist_ok=True, parents=True)

# Liste de cartes – ici un simple DataFrame. Remplacez par votre BDD.
MAPS = pd.DataFrame(
    {
        "short": ["A01", "ESL-Hock", "Easy-Race"],
        "gbx_file": [
            "A01-Race.Challenge.Gbx",
            "ESL-Hockolicious.Challenge.Gbx",
            "EASY-Race.Challenge.Gbx",
        ],
        "display_name": [
            "A01 – White Series Race",
            "ESL – Hockolicious",
            "Easy Race (community)",
        ],
        "ref_time_ms": [31000, 47000, 40000],  # temps cible de l’agent (exemple)
    }
)

# ------------------------------------------------------------------------------------
#  DB SQLAlchemy : table Replay
# ------------------------------------------------------------------------------------
Base = declarative_base()


class Replay(Base):
    __tablename__ = "replays"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user = Column(String, nullable=False)
    map_short = Column(String, nullable=False)
    race_time_ms = Column(Integer)        # temps total
    upload_time = Column(DateTime, default=datetime.utcnow)
    filename = Column(String, nullable=False)  # chemin sur disque
    tip_1 = Column(String)
    tip_2 = Column(String)
    tip_3 = Column(String)


engine = create_engine(f"sqlite:///{DB_PATH}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ------------------------------------------------------------------------------------
#  Fonctions d’« analyse » factices : à remplacer par votre pipeline RL
# ------------------------------------------------------------------------------------
def extract_race_time_ms(gbx_bytes: bytes) -> int:
    """
    Extrait naïvement la durée du replay depuis le header GBX.
    On se contente ici de renvoyer un temps bidon (entre 30 000 et 120 000 ms).
    """
    # 👉 À remplacer par pygbx ou un parseur maison pour récupérer le vrai temps.
    return random.randint(30000, 120000)


def make_tips(player_time: int, reference_time: int) -> list[str]:
    """
    Fabrique 3 conseils texto selon la différence de temps.
    """
    delta = player_time - reference_time
    tips = []
    if delta > 10000:
        tips.append("🚀 Utilise l’accélérateur plus tôt dans les lignes droites.")
        tips.append("💡 Travaille tes trajectoires : vise le centre de la route avant chaque virage.")
        tips.append("📉 Essaye de garder ta vitesse au-dessus de 250 km/h sur le premier split.")
    elif delta > 3000:
        tips.append("✅ Bon rythme ! Tu peux gratter encore quelques dixièmes en réduisant les freinages.")
        tips.append("💡 Dans le dernier virage, reste plus à l’intérieur pour réduire la distance.")
        tips.append("🕒 Utilise un release plutôt qu’un frein court au deuxième check-point.")
    else:
        tips.append("🔥 Excellent ! Tu es proche de la ligne de référence.")
        tips.append("⚙️ Concentre-toi sur la régularité pour stabiliser tes perfs.")
        tips.append("🏁 Tente un start plus agressif : boost + steering léger.")
    return tips[:3]


# ------------------------------------------------------------------------------------
#  Interface Streamlit
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="TrackMania Coach AI", layout="wide")
st.title("🏎️ StatMania Coach AI – ")

session_state = st.session_state.setdefault

st.sidebar.header("1️⃣ Choisir une carte")
map_choice = st.sidebar.selectbox(
    "Map",
    MAPS["display_name"],
    index=0,
)
map_row = MAPS.loc[MAPS["display_name"] == map_choice].iloc[0]
st.sidebar.markdown(
    f"**Fichier :** `{map_row.gbx_file}`  \n**Temps de référence agent :** {map_row.ref_time_ms/1000:.2f} s"
)

# Section upload
st.sidebar.header("2️⃣ Déposer un replay")
replay_file = st.sidebar.file_uploader(
    "Glisse ton `.Replay.Gbx` ici :",
    type=["Gbx", "gbx"],
)

user_name = st.sidebar.text_input("Ton pseudo", value="Akirastroworld")

analyze_btn = st.sidebar.button("Analyze replay 🔍", disabled=replay_file is None)

# ------------------------------------------------------------------------------------
#  Analyse on-click
# ------------------------------------------------------------------------------------
if analyze_btn and replay_file:
    with st.spinner("Analyse en cours…"):
        file_bytes = replay_file.read()
        race_time_ms = extract_race_time_ms(file_bytes)

        # Sauvegarde sur disque
        filename = (
            f"{user_name}_{map_row.short}_{int(time.time())}.Replay.Gbx"
        )
        file_path = REPLAY_DIR / filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Génération des tips
        tips = make_tips(race_time_ms, map_row.ref_time_ms)

        # Enregistrement en DB
        with Session() as db:
            rep = Replay(
                user=user_name,
                map_short=map_row.short,
                race_time_ms=race_time_ms,
                filename=str(file_path),
                tip_1=tips[0],
                tip_2=tips[1],
                tip_3=tips[2],
            )
            db.add(rep)
            db.commit()
            st.success(f"Replay enregistré ({race_time_ms/1000:.2f}s).")

    st.header("🚀 Conseils personnalisés")
    for t in tips:
        st.write("- " + t)

# ------------------------------------------------------------------------------------
#  Historique & progression
# ------------------------------------------------------------------------------------
st.header("📈 Historique de tes temps")

with Session() as db:
    hist = (
        pd.read_sql_table("replays", db.bind)
        .query("user == @user_name and map_short == @map_row.short")
        .sort_values("upload_time")
    )

if len(hist) == 0:
    st.info("Aucun replay enregistré pour l’instant.")
else:
    # Affichage tableau
    st.dataframe(
        hist[["upload_time", "race_time_ms"]]
        .rename(columns={"upload_time": "Date", "race_time_ms": "Temps (ms)"})
        .set_index("Date"),
        use_container_width=True,
    )

    # Graphique
    fig, ax = plt.subplots()
    ax.plot(
        pd.to_datetime(hist["upload_time"]),
        hist["race_time_ms"] / 1000,
        marker="o",
    )
    ax.axhline(map_row.ref_time_ms / 1000, ls="--", label="Agent ref")
    ax.set_ylabel("Temps (s)")
    ax.set_title(f"Progression sur {map_row.display_name}")
    ax.legend()
    st.pyplot(fig)

st.caption(
    "💡 Ce prototype utilise une analyse fictive. "
    "Connectez ici votre pipeline TrackMania RL pour produire les vraies métriques "
    "et suggestions (vitesses, trajectoires, contrôles…)."
)
