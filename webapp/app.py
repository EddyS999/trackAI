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


# ------------------------------------------------------------------------------------
#  Fonctions utilitaires
# ------------------------------------------------------------------------------------

def store_analysis_data_in_session(player_data, agent_data, vcp_path, replay_path, ref_files):
    """
    Stocke les données d'analyse dans session_state pour éviter les rechargements
    """
    st.session_state.analysis_data = {
        'player_data': player_data,
        'agent_data': agent_data,
        'vcp_path': vcp_path,
        'replay_path': replay_path,
        'ref_files': ref_files,
        'analysis_ready': True
    }

def main_analysis_section():
    """
    Section principale d'analyse avec gestion de session_state
    """
    # Vérifier si nous avons des données d'analyse en cours
    if hasattr(st.session_state, 'analysis_data') and st.session_state.analysis_data.get('analysis_ready', False):
        # Utiliser les données stockées
        data = st.session_state.analysis_data
        
        # Afficher les métriques de performance (code existant)
        # ... votre code existant pour les métriques ...
        
        # Appeler l'analyse de trajectoire avec les données stockées
        display_trajectory_analysis(
            data['vcp_path'], 
            data['replay_path'], 
            data['ref_files']
        )
    
    else:
        st.info("⬆️ Veuillez d'abord charger un replay pour voir l'analyse détaillée.")



def display_trajectory_analysis(vcp_path, replay_path, ref_files):
    """
    Fonction principale pour l'affichage de l'analyse de trajectoire améliorée
    """
    st.subheader("📍 Analyse détaillée de trajectoire (vue X–Z)")
    
    # Ajout du sélecteur de distance pour les segments avec session state
    col1, col2 = st.columns([3, 1])
    with col2:
        # Utiliser session_state pour conserver la valeur
        if 'segment_distance' not in st.session_state:
            st.session_state.segment_distance = 500
            
        segment_distance = st.selectbox(
            "🎯 Distance des segments (m)",
            options=[100, 500, 800],
            index=[100, 500, 800].index(st.session_state.segment_distance),
            help="Choisir la taille des segments d'analyse pour regrouper les données",
            key="segment_selector"
        )
        
        # Mettre à jour la session state
        st.session_state.segment_distance = segment_distance
    
    # Chargement des données avec spinner
    with st.spinner("📂 Chargement des données de trajectoire..."):
        vcp = np.load(vcp_path)
        pos_j = load_ghost_positions(replay_path)[:, [0,2]]
        
        if ref_files:
            pos_a = load_ghost_positions(str(ref_files[0]))[:, [0,2]]
        else:
            pos_a = None
            st.warning("Aucun fichier de référence agent trouvé")
            return
        
        time.sleep(0.3)  # Simulation temps de chargement
    
    # Analyse des trajectoires avec spinner amélioré
    with st.spinner("🔄 Analyse des trajectoires en cours..."):
        segments, sync_player, sync_agent, distances, deviations = calculate_trajectory_deviations(
            pos_j, pos_a, vcp, segment_length=segment_distance
        )
        time.sleep(0.5)
        
    with st.spinner("🔍 Identification des zones problématiques..."):
        problematic_segments = analyze_trajectory_issues(segments)
        time.sleep(0.3)
    
    # Graphique amélioré avec spinner
    with st.spinner("📊 Génération du graphique de trajectoire..."):
        fig = create_enhanced_trajectory_plot(vcp, pos_j, pos_a, segments, problematic_segments)
        time.sleep(0.4)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques globales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Écart moyen", f"{np.mean(deviations):.1f}m")
    with col2:
        st.metric("Écart maximum", f"{np.max(deviations):.1f}m")
    with col3:
        st.metric("Zones problématiques", len(problematic_segments))
    
    # Graphique des écarts le long du parcours avec spinner
    with st.spinner("📈 Génération du graphique d'écarts..."):
        st.subheader("📊 Écarts le long du parcours")
        fig_deviations = go.Figure()
        fig_deviations.add_trace(go.Scatter(
            x=distances, y=deviations,
            mode="lines",
            name="Écart à la trajectoire optimale",
            line=dict(color="orange", width=2)
        ))
        
        # Marquer les segments problématiques
        for prob_seg in problematic_segments:
            segment = prob_seg['segment']
            fig_deviations.add_vrect(
                x0=segment['start_distance'], 
                x1=segment['end_distance'],
                fillcolor="red" if prob_seg['severity'] > 2 else "orange",
                opacity=0.2,
                line_width=0
            )
        
        fig_deviations.update_layout(
            xaxis_title="Distance parcourue (m)",
            yaxis_title="Écart (m)",
            height=300
        )
        time.sleep(0.3)
    
    st.plotly_chart(fig_deviations, use_container_width=True)
    
    # Conseils détaillés par segment avec spinner
    with st.spinner("💡 Génération des conseils d'amélioration..."):
        time.sleep(0.4)  # Simulation du temps de calcul
        
        if problematic_segments:
            st.subheader("💡 Conseils d'amélioration")
            
            # Information sur le nombre de segments
            st.info(f"🎯 **Analyse par segments de {segment_distance}m** - {len(problematic_segments)} zone(s) à améliorer détectée(s)")
            
            # Trier par sévérité
            problematic_segments.sort(key=lambda x: x['severity'], reverse=True)
            
            # Limiter l'affichage pour éviter trop d'onglets
            max_segments_to_show = min(len(problematic_segments), 8)
            
            if len(problematic_segments) > max_segments_to_show:
                st.warning(f"⚠️ Affichage limité aux {max_segments_to_show} segments les plus problématiques sur {len(problematic_segments)} détectés.")
            
            for i, prob_seg in enumerate(problematic_segments[:max_segments_to_show]):
                segment = prob_seg['segment']
                
                # Icône selon la sévérité
                severity_icon = "🔴" if prob_seg['severity'] > 2 else "🟡" if prob_seg['severity'] > 1 else "🟢"
                
                with st.expander(f"{severity_icon} Segment {segment['id']+1} - Distance {segment['start_distance']:.0f}m à {segment['end_distance']:.0f}m ({segment['end_distance']-segment['start_distance']:.0f}m)"):
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**🔍 Problèmes identifiés:**")
                        for issue in prob_seg['issues']:
                            st.write(f"• {issue}")
                        
                        st.write("**📊 Statistiques:**")
                        st.write(f"• Écart moyen: {segment['mean_deviation']:.1f}m")
                        st.write(f"• Écart maximum: {segment['max_deviation']:.1f}m")
                        st.write(f"• Régularité: {segment['std_deviation']:.1f}m")
                        st.write(f"• Longueur: {segment['end_distance']-segment['start_distance']:.0f}m")
                    
                    with col2:
                        st.write("**💡 Conseils:**")
                        for advice in prob_seg['advice']:
                            st.write(f"💡 {advice}")
                        
                        # Conseil spécifique selon le type de problème
                        if segment['mean_deviation'] > 5:
                            st.error("🎯 **Conseil prioritaire:** Cette zone nécessite une attention particulière. "
                                "Entraînez-vous spécifiquement sur ce segment en mode libre.")
                        elif segment['mean_deviation'] > 3:
                            st.warning("🎯 **Conseil prioritaire:** Cette zone nécessite une attention particulière. "
                                "Entraînez-vous spécifiquement sur ce segment en mode libre.")
                        
                        if segment['std_deviation'] > 2:
                            st.info("🎮 **Conseil technique:** Travaillez la fluidité de vos mouvements. "
                                "Évitez les corrections brusques au volant.")
        
        else:
            st.success("🎉 Excellente trajectoire ! Aucune zone problématique majeure détectée.")
            st.info("💡 Continuez à vous entraîner pour maintenir cette régularité !")


def create_enhanced_trajectory_plot(vcp, pos_player, pos_agent, segments, problematic_segments):
    """
    Crée un graphique de trajectoire amélioré avec mise en évidence des zones problématiques
    Couleur verte pour la trajectoire du joueur
    """
    fig = go.Figure()
    
    # VCP de référence (gris)
    fig.add_trace(go.Scatter(
        x=vcp[:,0], y=vcp[:,2],
        mode="lines",
        line=dict(color="gray", width=2),
        name="VCP réf. agent",
    ))
    
    # Trajectoire du joueur en VERT
    fig.add_trace(go.Scatter(
        x=pos_player[:,0], y=pos_player[:,1],
        mode="lines",
        line=dict(color="green", width=3),
        name="Traj. Joueur",
        hovertemplate="x: %{x:.1f}  z: %{y:.1f}<extra></extra>",
    ))
    
    # Trajectoire de l'agent en bleu
    if pos_agent is not None:
        fig.add_trace(go.Scatter(
            x=pos_agent[:,0], y=pos_agent[:,1],
            mode="lines",
            line=dict(color="blue", width=2, dash="dash"),
            name="Traj. Agent",
        ))
    
    # Marquer les zones problématiques en ROUGE/ORANGE (distinctes du vert)
    for prob_seg in problematic_segments:
        segment = prob_seg['segment']
        player_pos = segment['player_positions']
        
        # Couleur selon la sévérité
        color = "red" if prob_seg['severity'] > 2 else "orange"
        
        fig.add_trace(go.Scatter(
            x=player_pos[:,0], y=player_pos[:,1],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"Zone problématique {segment['id']+1}",
            hovertemplate=f"Segment {segment['id']+1}<br>Écart moyen: {segment['mean_deviation']:.1f}m<extra></extra>",
        ))
    
    fig.update_layout(
        xaxis_title="X (m)",
        yaxis_title="Z (m)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=30, b=20),
        width=600,
        height=600,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def calculate_trajectory_curvature(positions):
    """Calcule la courbure approximative d'une trajectoire"""
    if len(positions) < 3:
        return np.array([])
    
    # Calculer les vecteurs directionnels
    vectors = np.diff(positions, axis=0)
    
    # Calculer les angles entre vecteurs successifs
    angles = []
    for i in range(len(vectors) - 1):
        v1, v2 = vectors[i], vectors[i + 1]
        # Normaliser
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        # Angle
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return np.array(angles)

def calculate_trajectory_deviations(pos_player, pos_agent, vcp_points, segment_length=500):
    """
    Calcule les écarts entre trajectoires par segments et identifie les zones problématiques
    
    Args:
        pos_player: positions du joueur (N, 2)
        pos_agent: positions de l'agent (M, 2) 
        vcp_points: virtual checkpoints (K, 3)
        segment_length: longueur des segments à analyser en mètres (défaut: 500m)
    
    Returns:
        segments, sync_player, sync_agent, common_distances, deviations
    """
    # Synchroniser les trajectoires sur une base commune (distance parcourue)
    def get_cumulative_distance(positions):
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        return np.concatenate([[0], np.cumsum(distances)])
    
    # Calculer les distances cumulées
    dist_player = get_cumulative_distance(pos_player)
    dist_agent = get_cumulative_distance(pos_agent)
    
    # Créer une base commune de distance
    max_dist = min(dist_player[-1], dist_agent[-1])
    common_distances = np.linspace(0, max_dist, int(max_dist / 0.5))  # tous les 0.5m
    
    # Interpoler les positions sur cette base commune
    interp_player_x = interp1d(dist_player, pos_player[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_player_z = interp1d(dist_player, pos_player[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_agent_x = interp1d(dist_agent, pos_agent[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_agent_z = interp1d(dist_agent, pos_agent[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Positions synchronisées
    sync_player = np.column_stack([interp_player_x(common_distances), interp_player_z(common_distances)])
    sync_agent = np.column_stack([interp_agent_x(common_distances), interp_agent_z(common_distances)])
    
    # Calculer les écarts point par point
    deviations = np.sqrt(np.sum((sync_player - sync_agent)**2, axis=1))
    
    # Analyser par segments (adaptation pour les différentes tailles)
    segments = []
    points_per_segment = int(segment_length / 0.5)  # nombre de points par segment
    num_segments = int(len(common_distances) / points_per_segment)
    
    for i in range(num_segments):
        start_idx = i * points_per_segment
        end_idx = min((i + 1) * points_per_segment, len(common_distances))
        
        if end_idx - start_idx < 10:  # Ignorer les segments trop petits
            continue
            
        segment_deviations = deviations[start_idx:end_idx]
        segment_distance = common_distances[end_idx-1] - common_distances[start_idx]
        
        segment_info = {
            'id': i,
            'start_distance': common_distances[start_idx],
            'end_distance': common_distances[end_idx-1],
            'distance_range': segment_distance,
            'mean_deviation': np.mean(segment_deviations),
            'max_deviation': np.max(segment_deviations),
            'std_deviation': np.std(segment_deviations),
            'player_positions': sync_player[start_idx:end_idx],
            'agent_positions': sync_agent[start_idx:end_idx],
            'deviations': segment_deviations
        }
        
        segments.append(segment_info)
    
    return segments, sync_player, sync_agent, common_distances, deviations


def analyze_trajectory_issues(segments, threshold_deviation=2.0, threshold_consistency=1.5):
    """
    Identifie les segments problématiques et génère des conseils
    Seuils adaptés selon la taille des segments
    """
    problematic_segments = []
    
    for segment in segments:
        issues = []
        advice = []
        
        # Adapter les seuils selon la longueur du segment
        segment_length = segment['distance_range']
        
        # Seuils adaptatifs
        if segment_length >= 800:
            dev_threshold = threshold_deviation * 1.5  # Plus tolérant sur les longs segments
            cons_threshold = threshold_consistency * 1.3
        elif segment_length >= 500:
            dev_threshold = threshold_deviation * 1.2
            cons_threshold = threshold_consistency * 1.1
        else:
            dev_threshold = threshold_deviation
            cons_threshold = threshold_consistency
        
        # Écart moyen trop important
        if segment['mean_deviation'] > dev_threshold:
            issues.append("Écart important avec la ligne optimale")
            advice.append("Essayez de vous rapprocher de la trajectoire de référence")
        
        # Inconsistance dans la trajectoire
        if segment['std_deviation'] > cons_threshold:
            issues.append("Trajectoire instable/zigzagante")
            advice.append("Travaillez la régularité de votre pilotage")
        
        # Analyse de la forme de la trajectoire
        player_pos = segment['player_positions']
        agent_pos = segment['agent_positions']
        
        if len(player_pos) > 3 and len(agent_pos) > 3:
            # Calculer la courbure relative
            player_angles = calculate_trajectory_curvature(player_pos)
            agent_angles = calculate_trajectory_curvature(agent_pos)
            
            if len(player_angles) > 0 and len(agent_angles) > 0:
                angle_diff = np.mean(np.abs(player_angles - agent_angles))
                
                if angle_diff > 0.3:  # seuil en radians
                    issues.append("Courbure de trajectoire différente")
                    if np.mean(player_angles) > np.mean(agent_angles):
                        advice.append("Votre trajectoire est trop serrée, élargissez vos courbes")
                    else:
                        advice.append("Votre trajectoire est trop large, resserrez vos courbes")
        
        # Détecter les sorties de piste potentielles (adapté à la longueur)
        max_dev = segment['max_deviation']
        danger_threshold = 5.0 if segment_length < 500 else 7.0
        
        if max_dev > danger_threshold:
            issues.append("Risque de sortie de piste")
            advice.append("Attention aux limites de piste, ralentissez si nécessaire")
        
        # Conseils spécifiques selon la longueur du segment
        if segment_length >= 800:
            advice.append(f"Segment long ({segment_length:.0f}m): concentrez-vous sur la constance")
        elif segment_length >= 500:
            advice.append(f"Segment moyen ({segment_length:.0f}m): optimisez vitesse et trajectoire")
        
        if issues:
            problematic_segments.append({
                'segment': segment,
                'issues': issues,
                'advice': advice,
                'severity': len(issues) + (1 if segment['mean_deviation'] > dev_threshold * 1.5 else 0)
            })
    
    return problematic_segments

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
    "short": ["A01", "A02","ESL-Hock"],
    "display_name": [
        "A01 – White Series Race",
        "A02 - White Series Race", 
        "ESL – Hockolicious",
    ],
    "vcp_file": [
        "A01_0.5m_.npy",
        "ESL-Hockolicious_0.5m_cl2.npy",
        "A02_0.5m_.npy",
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