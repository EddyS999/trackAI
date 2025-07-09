
import time

import numpy as np
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
