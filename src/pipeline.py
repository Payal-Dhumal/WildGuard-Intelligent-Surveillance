import cv2
import os
import json
from datetime import datetime
from ultralytics import YOLO
from incident_manager import IncidentManager
from llm_advisor import LLMAdvisor
from vector_store import WildGuardVectorStore
from quantized_embedding import QuantizedEmbeddingGemmaCPU

def draw_camera_banner(frame, camera_name, zone_name, timestamp_str, is_night):
    """
    Overlays a professional CCTV / Camera Trap status banner on top of the frame.
    """
    h, w, _ = frame.shape
    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text metadata
    mode_str = "[NIGHT MODE]" if is_night else "[DAY MODE]"
    left_text = f"CAM: {camera_name} | {zone_name}"
    right_text = f"{timestamp_str} {mode_str}"

    cv2.putText(frame, left_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Calculate right text size to position it cleanly
    (t_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, right_text, (w - t_w - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
    return frame

def run_wildguard_pipeline(video_path, model_path, camera_id="CAM_01", output_video_path=None):
    print("=" * 65)
    print("      WILDGUARD AI - LIVE MONITORING & INTELLIGENCE")
    print("=" * 65)

    # 1. Load Camera Metadata
    cam_file = os.path.join("data", "cameras.json")
    with open(cam_file, "r") as f:
        cameras = json.load(f)
    camera_info = cameras.get(camera_id, cameras["CAM_01"])

    print(f"\n[1] Active Camera Sensor: {camera_id} - {camera_info['name']}")
    print(f"    Zone: {camera_info['zone']} | GPS: ({camera_info['latitude']}, {camera_info['longitude']})")
    print(f"    Distance to Human Settlement: {camera_info['proximity_to_settlement_m']} meters")

    # 2. Initialize Models & Storage
    print(f"\n[2] Loading Multi-Class Wildlife Model: {model_path}")
    model = YOLO(model_path)
    
    print("[3] Initializing Incident Manager & Qdrant Vector Store...")
    incident_mgr = IncidentManager(min_conf=0.50, alarm_conf=0.65, min_frames=8)
    vector_store = WildGuardVectorStore(storage_path="data/qdrant_db")
    embedding_engine = QuantizedEmbeddingGemmaCPU(dimension=512)
    llm_advisor = LLMAdvisor()

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    total_alarms = 0
    os.makedirs(os.path.join("output", "evidence"), exist_ok=True)

    print(f"\n[4] Commencing Surveillance Analysis on '{video_path}'...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        is_night = (now_dt.hour < 6 or now_dt.hour >= 18)

        # Track objects
        results = model.track(frame, persist=True, verbose=False)

        current_frame_dets = []
        for box in results[0].boxes:
            track_id = int(box.id[0]) if box.id is not None else frame_idx
            cls_id = int(box.cls[0])
            species = model.names[cls_id]
            conf = float(box.conf[0])
            current_frame_dets.append({
                "track_id": track_id,
                "species": species,
                "confidence": conf
            })

        # Render YOLO bounding boxes & tracking
        annotated_frame = results[0].plot()

        # Add professional camera metadata banner
        annotated_frame = draw_camera_banner(
            annotated_frame, 
            camera_name=camera_info['name'], 
            zone_name=camera_info['zone'], 
            timestamp_str=timestamp_str, 
            is_night=is_night
        )

        # Update aggregator
        triggered_incidents = incident_mgr.update(current_frame_dets, camera_zone=camera_info['zone'])

        for inc in triggered_incidents:
            total_alarms += 1
            inc["camera_id"] = camera_id
            
            # Save visual evidence snapshot
            snapshot_file = os.path.join("output", "evidence", f"{inc['incident_id']}.jpg")
            cv2.imwrite(snapshot_file, annotated_frame)

            print("!" * 65)
            print(f"🚨 EMERGENCY ALERT: [{inc['species'].upper()}] DETECTED!")
            print(f"📍 Location: {camera_info['name']} ({camera_info['zone']})")
            print(f"🕒 Timestamp: {inc['timestamp']} | Confidence: {inc['max_confidence']*100:.1f}%")
            print(f"📸 Forensic Evidence Saved: {snapshot_file}")
            print("!" * 65)

            # LLM Evaluation
            analysis = llm_advisor.evaluate_incident(inc)

            # Persist directly into Qdrant Vector Store
            try:
                summary_text = (
                    f"Species: {inc['species']}, Threat Level: {analysis['threat_level']}, "
                    f"Threat Score: {analysis['threat_score']}, Zone: {camera_info['zone']}, "
                    f"Summary: {analysis['situation_summary']}"
                )
                multimodal_vector = embedding_engine.embed_multimodal_event(
                    text_context=summary_text,
                    image_path=snapshot_file
                )
                vector_store.insert_incident(
                    vector=multimodal_vector,
                    incident_data=inc,
                    llm_eval=analysis,
                    camera_info=camera_info,
                    snapshot_path=snapshot_file
                )
            except Exception as e:
                print(f"[Pipeline] Vector store insertion error: {e}")

            print(f"\n📊 THREAT LEVEL: {analysis['threat_level']} (Score: {analysis['threat_score']}/10)")
            print(f"🔊 PHYSICAL SIREN TRIGGER: {'YES - SOUND SIREN' if analysis['trigger_siren'] else 'NO - SILENT LOG'}")
            print(f"\n📝 SITUATION BRIEFING:\n{analysis['situation_summary']}")
            print(f"\n🛡️ TACTICAL RANGER DIRECTIVE:\n{analysis['ranger_instructions']}")
            print(f"\n📢 PUBLIC / HAMLET ADVISORY:\n{analysis['community_broadcast']}")
            print("=" * 65 + "\n")

        if out_writer:
            out_writer.write(annotated_frame)

    cap.release()
    if out_writer:
        out_writer.release()

    print(f"Surveillance session concluded. Analyzed {frame_idx} frames.")
    print(f"Active Sirens Triggered: {total_alarms}")

if __name__ == "__main__":
    video = os.path.join("data", "tiger_video.mp4")
    model = os.path.join("models", "wildlife_multi_yolov8.pt")
    output_vid = os.path.join("output", "monitored_feed.mp4")
    run_wildguard_pipeline(video, model, camera_id="CAM_01", output_video_path=output_vid)
