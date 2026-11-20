"""
Streamlit Dashboard for Driver Monitoring System
Provides analytics, session playback, and real-time monitoring visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob


# Page configuration
st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .alert-critical {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_session_data(logs_dir: str = "logs"):
    """Load all session data from logs directory."""
    sessions = []
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        return []
    
    # Find all summary files
    for summary_file in logs_path.glob("summary_*.json"):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            # Load corresponding events
            session_id = summary.get('session_id', '')
            events_file = logs_path / f"events_{session_id}.json"
            events = []
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events = json.load(f)
            
            sessions.append({
                'summary': summary,
                'events': events,
                'session_id': session_id
            })
        except Exception as e:
            st.warning(f"Error loading session: {e}")
    
    # Sort by start time (newest first)
    sessions.sort(key=lambda x: x['summary'].get('start_time', ''), reverse=True)
    return sessions


def create_gauge_chart(value, title, max_val=1.0, thresholds=None):
    """Create a gauge chart for metrics."""
    if thresholds is None:
        thresholds = [0.3, 0.6, 1.0]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, thresholds[0]], 'color': "#C8E6C9"},
                {'range': [thresholds[0], thresholds[1]], 'color': "#FFF9C4"},
                {'range': [thresholds[1], thresholds[2]], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds[1]
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_event_timeline(events):
    """Create a timeline visualization of events."""
    if not events:
        return None
    
    df = pd.DataFrame(events)
    if df.empty:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    color_map = {
        'drowsiness': '#F44336',
        'yawn': '#FF9800',
        'distraction': '#9C27B0'
    }
    
    fig = px.scatter(
        df,
        x='timestamp',
        y='event_type',
        color='event_type',
        color_discrete_map=color_map,
        size_max=20,
        title='Event Timeline'
    )
    
    fig.update_traces(marker=dict(size=15, symbol='diamond'))
    fig.update_layout(
        height=300,
        xaxis_title="Time",
        yaxis_title="Event Type",
        showlegend=True
    )
    
    return fig


def create_metrics_over_time(events):
    """Create line charts for EAR/MAR over time."""
    if not events:
        return None
    
    df = pd.DataFrame(events)
    if df.empty:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Eye Aspect Ratio (EAR)', 'Mouth Aspect Ratio (MAR)'),
        shared_xaxes=True,
        vertical_spacing=0.15
    )
    
    # EAR values
    ear_df = df[df['ear_value'].notna()]
    if not ear_df.empty:
        fig.add_trace(
            go.Scatter(x=ear_df['timestamp'], y=ear_df['ear_value'],
                      mode='lines+markers', name='EAR',
                      line=dict(color='#1E88E5')),
            row=1, col=1
        )
        # Add threshold line
        fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                     annotation_text="Threshold", row=1, col=1)
    
    # MAR values
    mar_df = df[df['mar_value'].notna()]
    if not mar_df.empty:
        fig.add_trace(
            go.Scatter(x=mar_df['timestamp'], y=mar_df['mar_value'],
                      mode='lines+markers', name='MAR',
                      line=dict(color='#FF9800')),
            row=2, col=1
        )
        # Add threshold line
        fig.add_hline(y=0.6, line_dash="dash", line_color="red",
                     annotation_text="Threshold", row=2, col=1)
    
    fig.update_layout(height=400, showlegend=True)
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Driver Monitoring System Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/car.png", width=80)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "📈 Session Analysis", "⚙️ Settings"],
        index=0
    )
    
    # Load session data
    sessions = load_session_data()
    
    if page == "📊 Overview":
        show_overview(sessions)
    elif page == "📈 Session Analysis":
        show_session_analysis(sessions)
    elif page == "⚙️ Settings":
        show_settings()


def show_overview(sessions):
    """Show overview page with aggregate statistics."""
    
    st.header("📊 System Overview")
    
    if not sessions:
        st.info("No sessions recorded yet. Run the Driver Monitoring System to generate data.")
        st.code("python main.py", language="bash")
        return
    
    # Aggregate statistics
    total_sessions = len(sessions)
    total_duration = sum(s['summary'].get('duration_seconds', 0) for s in sessions)
    total_events = sum(s['summary'].get('total_events', 0) for s in sessions)
    total_drowsy = sum(s['summary'].get('drowsy_frames', 0) for s in sessions)
    total_yawns = sum(s['summary'].get('yawn_count', 0) for s in sessions)
    total_distractions = sum(s['summary'].get('distraction_count', 0) for s in sessions)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", total_sessions, 
                  delta=f"+{1 if len(sessions) > 0 else 0} today")
    
    with col2:
        hours = total_duration / 3600
        st.metric("Total Drive Time", f"{hours:.1f}h")
    
    with col3:
        st.metric("Total Alerts", total_events,
                  delta_color="inverse")
    
    with col4:
        avg_events_per_session = total_events / max(total_sessions, 1)
        st.metric("Avg Alerts/Session", f"{avg_events_per_session:.1f}")
    
    st.divider()
    
    # Event breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Event Breakdown")
        
        event_data = {
            'Event Type': ['Drowsiness', 'Yawning', 'Distraction'],
            'Count': [total_drowsy, total_yawns, total_distractions]
        }
        df = pd.DataFrame(event_data)
        
        fig = px.pie(df, values='Count', names='Event Type',
                    color='Event Type',
                    color_discrete_map={
                        'Drowsiness': '#F44336',
                        'Yawning': '#FF9800',
                        'Distraction': '#9C27B0'
                    },
                    hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Recent Sessions")
        
        recent_data = []
        for s in sessions[:5]:
            summary = s['summary']
            recent_data.append({
                'Session': summary.get('session_id', 'N/A')[:15],
                'Duration': f"{summary.get('duration_seconds', 0)/60:.1f}m",
                'Events': summary.get('total_events', 0),
                'Status': '🔴' if summary.get('critical_alerts', 0) > 0 else '🟢'
            })
        
        if recent_data:
            st.dataframe(pd.DataFrame(recent_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Safety score
    st.subheader("🛡️ Safety Score")
    
    # Calculate safety score (inverse of alert rate)
    total_frames = sum(s['summary'].get('total_frames', 1) for s in sessions)
    alert_rate = total_events / max(total_frames, 1)
    safety_score = max(0, min(100, 100 - (alert_rate * 1000)))
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safety_score,
            title={'text': "Overall Safety Score", 'font': {'size': 20}},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50" if safety_score > 70 else "#FF9800" if safety_score > 40 else "#F44336"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFCDD2"},
                    {'range': [40, 70], 'color': "#FFF9C4"},
                    {'range': [70, 100], 'color': "#C8E6C9"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def show_session_analysis(sessions):
    """Show detailed session analysis."""
    
    st.header("📈 Session Analysis")
    
    if not sessions:
        st.info("No sessions to analyze. Run the monitoring system first.")
        return
    
    # Session selector
    session_options = {
        f"{s['session_id']} - {s['summary'].get('start_time', 'N/A')[:19]}": i 
        for i, s in enumerate(sessions)
    }
    
    selected = st.selectbox("Select Session", list(session_options.keys()))
    session_idx = session_options[selected]
    session = sessions[session_idx]
    
    summary = session['summary']
    events = session['events']
    
    st.divider()
    
    # Session summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Duration", f"{summary.get('duration_seconds', 0)/60:.1f} min")
    with col2:
        st.metric("Total Frames", summary.get('total_frames', 0))
    with col3:
        st.metric("Drowsy Events", summary.get('drowsy_frames', 0))
    with col4:
        st.metric("Yawn Count", summary.get('yawn_count', 0))
    with col5:
        st.metric("Distractions", summary.get('distraction_count', 0))
    
    st.divider()
    
    # Event timeline
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Event Timeline")
        timeline_fig = create_event_timeline(events)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("No events recorded in this session.")
    
    with col2:
        st.subheader("📋 Event Log")
        if events:
            for event in events[:10]:
                event_type = event.get('event_type', 'unknown')
                severity = event.get('severity', 'warning')
                timestamp = event.get('timestamp', '')[:19]
                
                icon = {'drowsiness': '😴', 'yawn': '🥱', 'distraction': '👀'}.get(event_type, '⚠️')
                color = 'red' if severity == 'critical' else 'orange'
                
                st.markdown(f"**{icon} {event_type.title()}** - {timestamp}")
        else:
            st.success("No fatigue events detected! 🎉")
    
    st.divider()
    
    # Metrics visualization
    st.subheader("📊 Metrics Over Time")
    metrics_fig = create_metrics_over_time(events)
    if metrics_fig:
        st.plotly_chart(metrics_fig, use_container_width=True)
    else:
        st.info("Insufficient data for metrics visualization.")
    
    # Raw data
    with st.expander("🔍 View Raw Event Data"):
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.info("No events to display.")


def show_settings():
    """Show settings page."""
    
    st.header("⚙️ Settings")
    
    st.subheader("Detection Thresholds")
    st.info("These are the current detection thresholds. Edit `src/utils/config.py` to change them.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👁️ Eye Tracking")
        st.text("EAR Threshold: 0.25")
        st.text("Consecutive Frames: 30")
        
        st.markdown("### 👄 Mouth Tracking")
        st.text("MAR Threshold: 0.6")
        st.text("Yawn Duration: 2.0s")
    
    with col2:
        st.markdown("### 🔄 Head Pose")
        st.text("Yaw Threshold: 30°")
        st.text("Pitch Threshold: 20°")
        st.text("Distraction Frames: 45")
        
        st.markdown("### 🔔 Alerts")
        st.text("Alert Cooldown: 3.0s")
    
    st.divider()
    
    st.subheader("📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logs_path = Path("logs")
        if logs_path.exists():
            log_files = list(logs_path.glob("*"))
            st.text(f"Log files: {len(log_files)}")
        else:
            st.text("Log directory: Not created yet")
    
    with col2:
        recordings_path = Path("recordings")
        if recordings_path.exists():
            recordings = list(recordings_path.glob("*.mp4"))
            st.text(f"Recordings: {len(recordings)}")
        else:
            st.text("Recordings directory: Not created yet")
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **Driver Monitoring System v1.0**
    
    A real-time computer vision system for detecting driver fatigue and distraction.
    
    - 🔍 Face detection with MediaPipe
    - 👁️ Drowsiness detection (EAR)
    - 🥱 Yawn detection (MAR)
    - 🔄 Head pose estimation
    - 📹 Video recording
    - 📊 Analytics dashboard
    """)


if __name__ == "__main__":
    main()
