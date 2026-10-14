import streamlit as st
import cv2
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import sys

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from chatbot import WildlifeChatbot
from vector_store import WildGuardVectorStore

# Page Configuration
st.set_page_config(
    page_title="WildGuard AI - Command Center",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the reference UI (Modern Clean Dark/Glass Neumorphic styling)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background-color: #0b0f14;
        color: #e2e8f0;
    }
    
    /* Top Header Bar */
    .header-bar {
        background: #111722;
        padding: 12px 24px;
        border-radius: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        border: 1px solid #1e293b;
    }
    
    .brand-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .brand-title span {
        color: #10b981;
    }
    
    .brand-sub {
        font-size: 11px;
        color: #94a3b8;
    }
    
    .badge-info {
        background: #1e293b;
        color: #cbd5e1;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Cards */
    .wg-card {
        background: #111722;
        border-radius: 16px;
        padding: 18px;
        border: 1px solid #1e293b;
        margin-bottom: 16px;
    }

    .wg-title {
        font-size: 15px;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Risk Metrics */
    .stat-val {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }

    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .badge-high {
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .badge-medium {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Video Overlay Banner */
    .danger-overlay {
        background: rgba(220, 38, 38, 0.85);
        backdrop-filter: blur(8px);
        color: white;
        padding: 10px 16px;
        border-radius: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: -55px;
        position: relative;
        z-index: 10;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Show tiger sightings from CAM_01 in the last 7 days"},
        {
            "role": "assistant",
            "content": """**Here are the tiger sightings from CAM_01 in the last 7 days:**
1. **12 Oct 2026, 02:47 AM** — Confidence: 92% (Near village boundary fence)
2. **09 Oct 2026, 11:12 PM** — Confidence: 88% (Near perimeter road)
3. **07 Oct 2026, 01:05 AM** — Confidence: 86% (Near village boundary)

🚨 **Status:** Perimeter siren and floodlight array triggered on latest encounter."""
        }
    ]

# Vector database connection
db = WildGuardVectorStore(storage_path="data/qdrant_db")
chatbot = WildlifeChatbot(storage_path="data/qdrant_db")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 24px;'>
            <div style='font-size: 38px;'>🐾</div>
            <div style='font-size: 20px; font-weight: 700; color: #fff;'>WildGuard <span style='color: #10b981;'>AI</span></div>
            <div style='font-size: 11px; color: #94a3b8;'>Intelligent Intrusion System</div>
        </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["📊 Dashboard", "📹 Live Monitor", "🚨 Alerts (3)", "📑 Sightings Log", "💬 AI Ranger", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: #111722; padding: 12px; border-radius: 12px; border: 1px solid #1e293b; display: flex; align-items: center; gap: 10px;'>
            <div style='font-size: 24px;'>👤</div>
            <div>
                <div style='font-size: 13px; font-weight: 600; color: #fff;'>Payal Dhumal</div>
                <div style='font-size: 11px; color: #10b981;'>🟢 Active Operator</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- TOP HEADER BAR -----------------
now_dt = datetime.now()
st.markdown(f"""
    <div class='header-bar'>
        <div>
            <div class='brand-title'>🐾 WildGuard <span>AI</span></div>
            <div class='brand-sub'>Wildlife Detection and Intelligent Alert System</div>
        </div>
        <div style='display: flex; gap: 12px;'>
            <div class='badge-info'>📅 {now_dt.strftime('%d %b %Y, %I:%M:%S %p')} IST</div>
            <div class='badge-info'>🌙 23°C Clear Night</div>
            <div class='badge-info' style='background: #ef4444; color: white;'>🔔 3 Alerts</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------- MAIN LAYOUT -----------------
col_left, col_right = st.columns([1.6, 1.0], gap="medium")

# ==================== LEFT COLUMN: CAMERA FEED & RECENT SIGHTINGS ====================
with col_left:
    # 1. LIVE CAMERA FEED CARD
    st.markdown("""
        <div class='wg-card'>
            <div class='wg-title'>
                <span>📹 <b>Live Camera Feed — CAM_01</b> <span style='color: #10b981; font-size: 12px; margin-left: 8px;'>● Live</span></span>
                <span style='font-size: 12px; color: #94a3b8;'>📍 Village Boundary (Sector 4) | 26.0173° N, 92.9376° E</span>
            </div>
    """, unsafe_allow_html=True)

    # Display Detected Video / Snapshot
    evidence_img_path = "output/sample_detected_frame.jpg"
    if os.path.exists(evidence_img_path):
        st.image(evidence_img_path, use_container_width=True)
    else:
        st.info("No video feed active.")

    # High Risk Overlay Banner
    st.markdown("""
        <div class='danger-overlay'>
            <span style='font-size: 22px;'>⚠️</span>
            <div>
                <div style='font-size: 13px; font-weight: 700; letter-spacing: 0.5px;'>HIGH RISK — CRITICAL THREAT</div>
                <div style='font-size: 11px; opacity: 0.9;'>Apex predator approaching settlement (Sector 4 Perimeter Fence)</div>
            </div>
        </div>
        <br>
        <div style='display: flex; justify-content: space-between; align-items: center; color: #94a3b8; font-size: 13px; margin-top: 10px;'>
            <div>⏸️ <b>LIVE FEED ACTIVE</b> (30 FPS)</div>
            <div>🔊 Siren: <b>ONLINE</b> | 📸 Capture: <b>ACTIVE</b></div>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. RECENT SIGHTINGS TABLE CARD
    st.markdown("""
        <div class='wg-card'>
            <div class='wg-title'>
                <span>📑 <b>Recent Sightings Evidence</b></span>
                <span style='font-size: 12px; color: #3b82f6; cursor: pointer;'>View All →</span>
            </div>
    """, unsafe_allow_html=True)

    incidents = db.get_all_incidents()[:4]
    
    # Table Header
    cols = st.columns([1.2, 1.8, 1.2, 1.2, 2.0, 1.2])
    cols[0].caption("IMAGE")
    cols[1].caption("TIME")
    cols[2].caption("ANIMAL")
    cols[3].caption("CAMERA")
    cols[4].caption("ZONE")
    cols[5].caption("RISK")

    for inc in incidents:
        c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.8, 1.2, 1.2, 2.0, 1.2])
        
        # Thumbnail
        thumb = inc.get("snapshot_path")
        if thumb and os.path.exists(thumb):
            c1.image(thumb, width=50)
        else:
            c1.write("🐾")

        c2.markdown(f"<span style='font-size: 11px;'>{inc['timestamp']}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='font-size: 12px; font-weight: 600;'>{inc['species']}</span>", unsafe_allow_html=True)
        c4.markdown(f"<span style='font-size: 11px;'>{inc['camera_id']}</span>", unsafe_allow_html=True)
        c5.markdown(f"<span style='font-size: 11px;'>{inc['zone']}</span>", unsafe_allow_html=True)
        
        # Risk Badge
        lvl = inc['threat_level']
        badge_class = "badge-critical" if lvl in ["CRITICAL", "HIGH"] else ("badge-medium" if lvl == "MEDIUM" else "badge-low")
        c6.markdown(f"<span class='{badge_class}'>{lvl}</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==================== RIGHT COLUMN: RISK OVERVIEW & AI RANGER CHAT ====================
with col_right:
    # 1. RISK OVERVIEW CARD
    st.markdown("""
        <div class='wg-card'>
            <div class='wg-title'>
                <span>📊 <b>Risk Overview</b></span>
                <span style='font-size: 12px; color: #94a3b8;'>Last 7 Days ▼</span>
            </div>
            <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;'>
                <div>
                    <span class='stat-val'>27</span>
                    <span style='color: #94a3b8; font-size: 12px; margin-left: 6px;'>Total Sightings <b style='color: #10b981;'>↑ 12%</b></span>
                </div>
            </div>
            <div style='display: flex; gap: 8px;'>
                <div style='flex: 1; background: #1e293b; padding: 10px; border-radius: 10px; text-align: center;'>
                    <div style='color: #ef4444; font-weight: 700; font-size: 16px;'>8</div>
                    <div style='font-size: 10px; color: #94a3b8;'>High Risk</div>
                </div>
                <div style='flex: 1; background: #1e293b; padding: 10px; border-radius: 10px; text-align: center;'>
                    <div style='color: #eab308; font-weight: 700; font-size: 16px;'>11</div>
                    <div style='font-size: 10px; color: #94a3b8;'>Medium Risk</div>
                </div>
                <div style='flex: 1; background: #1e293b; padding: 10px; border-radius: 10px; text-align: center;'>
                    <div style='color: #10b981; font-weight: 700; font-size: 16px;'>8</div>
                    <div style='font-size: 10px; color: #94a3b8;'>Low Risk</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. CAMERA SENSOR STATUS CARD
    st.markdown("""
        <div class='wg-card'>
            <div class='wg-title'>
                <span>📹 <b>Camera Sensor Status</b></span>
                <span style='font-size: 12px; color: #3b82f6;'>View All →</span>
            </div>
            <div style='display: flex; flex-direction: column; gap: 10px;'>
                <div style='background: #1e293b; padding: 10px 14px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='font-size: 12px; font-weight: 600; color: #fff;'>CAM_01 • Village Boundary (Sector 4)</div>
                        <div style='font-size: 10px; color: #94a3b8;'>26.0173° N, 92.9376° E</div>
                    </div>
                    <span class='badge-low'>● Live</span>
                </div>
                <div style='background: #1e293b; padding: 10px 14px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='font-size: 12px; font-weight: 600; color: #fff;'>CAM_02 • Forest Corridor (Sector 12)</div>
                        <div style='font-size: 10px; color: #94a3b8;'>26.0450° N, 92.9810° E</div>
                    </div>
                    <span class='badge-critical'>● Offline</span>
                </div>
                <div style='background: #1e293b; padding: 10px 14px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <div style='font-size: 12px; font-weight: 600; color: #fff;'>CAM_03 • Waterhole Zone (Sector 7)</div>
                        <div style='font-size: 10px; color: #94a3b8;'>25.9921° N, 92.8943° E</div>
                    </div>
                    <span class='badge-low'>● Live</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. AI RANGER CHATBOT (POWERED BY GEMINI 3.8 FLASH)
    st.markdown("""
        <div class='wg-card'>
            <div class='wg-title'>
                <span>🤖 <b>AI Ranger Assistant</b> <span style='font-size: 11px; color: #38bdf8; margin-left: 6px;'>✦ Powered by Gemini 3.8 Flash</span></span>
                <span style='font-size: 11px; color: #94a3b8; cursor: pointer;'>Clear Chat</span>
            </div>
    """, unsafe_allow_html=True)

    # Chat Messages Box
    chat_container = st.container(height=300)
    for msg in st.session_state.messages:
        with chat_container:
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style='text-align: right; margin-bottom: 8px;'>
                        <span style='background: #1e3a8a; color: white; padding: 8px 14px; border-radius: 16px; font-size: 12px; display: inline-block;'>
                            {msg['content']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='text-align: left; margin-bottom: 8px;'>
                        <div style='background: #1e293b; color: #f1f5f9; padding: 10px 14px; border-radius: 16px; font-size: 12px; display: inline-block; border-left: 3px solid #38bdf8;'>
                            {msg['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about sightings, patterns, risk analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            st.markdown(f"""
                <div style='text-align: right; margin-bottom: 8px;'>
                    <span style='background: #1e3a8a; color: white; padding: 8px 14px; border-radius: 16px; font-size: 12px; display: inline-block;'>
                        {prompt}
                    </span>
                </div>
            """, unsafe_allow_html=True)
        
        # Get response from chatbot (Gemini 3.8 Flash)
        with st.spinner("AI Ranger analyzing database evidence..."):
            response = chatbot.ask(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
