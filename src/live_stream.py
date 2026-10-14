import cv2
import os
import sys
import json
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from incident_manager import IncidentManager
from llm_advisor import LLMAdvisor
from quantized_embedding import QuantizedEmbeddingGemmaCPU
from vector_store import WildGuardVectorStore

class LiveStreamManager:
    """
    Manages real-time video streaming with live YOLOv8 tracking,
    camera banner overlay, and automatic threat detection.
    Supports:
      1. Default sample wildlife video (looped)
      2. Uploaded user video file
      3. Live local Webcam (Device 0)
    """

    def __init__(self, model_path="models/wildlife_multi_yolov8.pt"):
        print(f"[LiveStreamManager] Loading model from {model_path}...")
        self.lock = threading.Lock()
        self.model = YOLO(model_path)
        self.incident_mgr = IncidentManager(min_conf=0.45, alarm_conf=0.50, min_frames=1, exit_grace_period=2.0)
        self.llm_advisor = LLMAdvisor()
        self.embedding_engine = QuantizedEmbeddingGemmaCPU(dimension=512)
        self.vector_store = WildGuardVectorStore(storage_path="data/qdrant_db", vector_dim=512)

        # Load camera metadata
        cam_file = os.path.join("data", "cameras.json")
        if os.path.exists(cam_file):
            with open(cam_file, "r") as f:
                self.cameras = json.load(f)
        else:
            self.cameras = {"CAM_01": {"name": "Village Boundary", "zone": "Sector 4"}}

        # Geofence Polygon State (Default: Central observation danger corridor)
        self.default_polygon = [[100, 120], [540, 120], [540, 440], [100, 440]]
        self.geofence_polygon = self._load_geofence()

        # Cache first frame for user interactive geofence drawing
        self.first_frame = None
        self.first_frame_bytes = None

        # Use the user's latest uploaded video or fallback to sample
        user_uploads = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".mp4")] if os.path.exists("uploads") else []
        if user_uploads:
            self.current_source = user_uploads[0]
        elif os.path.exists("data/sample_wildlife.mp4"):
            self.current_source = "data/sample_wildlife.mp4"
        else:
            self.current_source = None

        self.current_cam_id = "CAM_01"
        self.cap = None
        self.is_webcam = False
        self.latest_alert = None
        self.latest_frame = None

        # Pre-capture first frame if source exists
        if self.current_source:
            self._capture_first_frame()

    def _load_geofence(self):
        geo_file = os.path.join("data", "geofence.json")
        if os.path.exists(geo_file):
            try:
                with open(geo_file, "r") as f:
                    pts = json.load(f)
                    if isinstance(pts, list) and len(pts) >= 3:
                        return pts
            except Exception:
                pass
        return list(self.default_polygon)

    def _capture_first_frame(self):
        try:
            with self.lock:
                src = 0 if self.is_webcam else self.current_source
                if not src:
                    return
                c = cv2.VideoCapture(src)
                ret, frame = c.read()
                if ret and frame is not None:
                    self.first_frame = cv2.resize(frame, (640, 480))
                    ret_enc, buf = cv2.imencode('.jpg', self.first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret_enc:
                        self.first_frame_bytes = buf.tobytes()
                c.release()
        except Exception as e:
            print(f"[LiveStreamManager] Error capturing first frame: {e}")

    def set_geofence(self, points):
        """
        Sets a custom polygon geofence from user clicks.
        points: list of [x, y] coordinates in 640x480 frame space.
        """
        if not points or len(points) < 3:
            raise ValueError("Polygon geofence requires at least 3 points.")
        
        self.geofence_polygon = [[int(pt[0]), int(pt[1])] for pt in points]
        try:
            with open(os.path.join("data", "geofence.json"), "w") as f:
                json.dump(self.geofence_polygon, f)
        except Exception as e:
            print(f"[LiveStreamManager] Error saving geofence: {e}")

        # Reset tracking session state when geofence is modified
        self.incident_mgr.reset_session()
        self.latest_alert = None
        print(f"[LiveStreamManager] Updated geofence with {len(self.geofence_polygon)} points.")
        return self.geofence_polygon

    def reset_geofence(self):
        """Resets geofence to default."""
        self.geofence_polygon = list(self.default_polygon)
        geo_file = os.path.join("data", "geofence.json")
        if os.path.exists(geo_file):
            os.remove(geo_file)
        self.incident_mgr.reset_session()
        self.latest_alert = None
        return self.geofence_polygon

    def get_first_frame_bytes(self):
        """Returns JPEG bytes of first frame for user drawing."""
        if self.first_frame_bytes is None:
            self._capture_first_frame()
        if self.first_frame_bytes is None and self.latest_frame is not None:
            _, buf = cv2.imencode('.jpg', self.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self.first_frame_bytes = buf.tobytes()
        return self.first_frame_bytes

    def set_source(self, source_path_or_cam_index, cam_id="CAM_01"):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            self.current_source = source_path_or_cam_index
            self.current_cam_id = cam_id
            self.is_webcam = (source_path_or_cam_index == 0 or source_path_or_cam_index == "0")
            
            # Reset in-memory tracking session state for new video processing session
            self.incident_mgr.reset_session()
            self.latest_alert = None
            self.latest_frame = None
            self.first_frame = None
            self.first_frame_bytes = None

        self._capture_first_frame()
        print(f"[LiveStreamManager] Switched source to: {self.current_source} (Camera: {cam_id})")

    def _get_capture(self):
        if self.cap is None or not self.cap.isOpened():
            src = 0 if self.is_webcam else self.current_source
            if src is not None:
                self.cap = cv2.VideoCapture(src)
        return self.cap

    def generate_frames(self):
        """
        Continuous generator that yields MJPEG multipart frames for HTTP streaming.
        """
        frame_idx = 0
        cam_info = self.cameras.get(self.current_cam_id, {"name": "Village Boundary", "zone": "Sector 4"})

        while True:
            with self.lock:
                cap = self._get_capture()
                if cap is None or not cap.isOpened():
                    ret = False
                    frame = None
                else:
                    ret, frame = cap.read()

            if not ret or frame is None:
                # Video has reached the end!
                # Do NOT restart/loop uploaded videos
                if self.latest_frame is not None:
                    # Freeze on the final annotated frame and overlay "ANALYSIS COMPLETE"
                    frozen_frame = self.latest_frame.copy()
                    h, w, _ = frozen_frame.shape
                    
                    # Dark bottom completion banner
                    cv2.rectangle(frozen_frame, (0, h - 45), (w, h), (15, 23, 42), -1)
                    cv2.putText(
                        frozen_frame, 
                        "VIDEO PLAYBACK COMPLETED - INCIDENT LOGGED", 
                        (30, h - 16), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.52, 
                        (16, 185, 129), 
                        2, 
                        cv2.LINE_AA
                    )
                    
                    ret_enc, buffer = cv2.imencode('.jpg', frozen_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret_enc:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(1.0)
                    continue
                else:
                    time.sleep(0.5)
                    continue

            frame_idx += 1
            now_dt = datetime.now()
            timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            is_night = (now_dt.hour < 6 or now_dt.hour >= 18)

            # Resize frame smoothly for fast, responsive web streaming
            frame = cv2.resize(frame, (640, 480))
            if self.first_frame is None:
                self.first_frame = frame.copy()
                ret_enc, buf = cv2.imencode('.jpg', self.first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret_enc:
                    self.first_frame_bytes = buf.tobytes()

            # Run YOLO tracking every frame
            results = self.model.track(frame, persist=True, verbose=False)

            current_frame_dets = []
            poly_np = np.array(self.geofence_polygon, dtype=np.int32) if (self.geofence_polygon and len(self.geofence_polygon) >= 3) else None

            for box in results[0].boxes:
                track_id = int(box.id[0]) if box.id is not None else frame_idx
                cls_id = int(box.cls[0])
                species = self.model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None

                in_danger_zone = True
                anchor_pt = None
                if xyxy:
                    # Bottom-center anchor point (animal ground/feet contact point)
                    anchor_x = int((xyxy[0] + xyxy[2]) / 2)
                    anchor_y = int(xyxy[3])
                    anchor_pt = (anchor_x, anchor_y)
                    if poly_np is not None:
                        # cv2.pointPolygonTest: >= 0 inside or on edge, < 0 outside
                        test_res = cv2.pointPolygonTest(poly_np, (float(anchor_x), float(anchor_y)), measureDist=False)
                        in_danger_zone = (test_res >= 0)

                current_frame_dets.append({
                    "track_id": track_id,
                    "species": species,
                    "confidence": conf,
                    "box": xyxy,
                    "anchor": anchor_pt,
                    "in_danger_zone": in_danger_zone
                })

            # Base annotated frame from YOLO tracks
            annotated_frame = results[0].plot()

            # 1. Overlay Virtual Polygon Geofence Danger Zone
            if poly_np is not None:
                # Semi-transparent red warning overlay inside the polygon
                poly_mask = annotated_frame.copy()
                cv2.fillPoly(poly_mask, [poly_np], (0, 30, 210))
                cv2.addWeighted(poly_mask, 0.22, annotated_frame, 0.78, 0, annotated_frame)

                # High-contrast border line
                cv2.polylines(annotated_frame, [poly_np], isClosed=True, color=(0, 60, 255), thickness=2, lineType=cv2.LINE_AA)

                # Geofence perimeter vertices
                for pt in self.geofence_polygon:
                    cv2.circle(annotated_frame, (pt[0], pt[1]), 4, (0, 220, 255), -1)

                # Danger Zone badge
                gx, gy = self.geofence_polygon[0]
                cv2.rectangle(annotated_frame, (max(5, gx - 4), max(5, gy - 20)), (max(5, gx + 195), max(5, gy)), (15, 23, 42), -1)
                cv2.putText(annotated_frame, "DANGER ZONE GEOFENCE", (max(10, gx), max(16, gy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 80, 255), 1, cv2.LINE_AA)

            # 2. Render Zone Status badges and ground contact anchors for each animal
            for det in current_frame_dets:
                if det["anchor"] and det["box"]:
                    ax, ay = det["anchor"]
                    in_zone = det["in_danger_zone"]
                    sp = det["species"]
                    bx1, by1 = int(det["box"][0]), int(det["box"][1])

                    # Ground contact point
                    dot_col = (0, 0, 255) if in_zone else (0, 220, 100)
                    cv2.circle(annotated_frame, (ax, ay), 5, dot_col, -1)
                    cv2.circle(annotated_frame, (ax, ay), 7, (255, 255, 255), 1)

                    # Badge on top of bounding box
                    badge_lbl = f"{sp}: IN DANGER ZONE" if in_zone else f"{sp}: OUTSIDE ZONE"
                    badge_bg = (0, 0, 210) if in_zone else (20, 120, 50)
                    cv2.rectangle(annotated_frame, (bx1, max(15, by1 - 22)), (bx1 + 175, max(15, by1)), badge_bg, -1)
                    cv2.putText(annotated_frame, badge_lbl, (bx1 + 4, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw camera overlay banner
            h, w, _ = annotated_frame.shape
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 36), (15, 20, 26), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

            mode_str = "[NIGHT MODE]" if is_night else "[DAY MODE]"
            left_text = f"CAM: {cam_info['name']} | {cam_info['zone']}"
            right_text = f"{timestamp_str} {mode_str}"
            cv2.putText(annotated_frame, left_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, right_text, (w - 240, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # Check if incident is triggered (Risk calculated ONLY when animal enters danger zone)
            triggered = self.incident_mgr.update(current_frame_dets, camera_zone=cam_info["zone"])
            for inc in triggered:
                inc["camera_id"] = self.current_cam_id
                
                # Save ACTUAL captured frame snapshot with bounding box into static/evidence
                os.makedirs("static/evidence", exist_ok=True)
                snapshot_filename = f"{inc['incident_id']}.jpg"
                snapshot_filepath = os.path.join("static", "evidence", snapshot_filename)
                cv2.imwrite(snapshot_filepath, annotated_frame)
                snapshot_url = f"/static/evidence/{snapshot_filename}"

                # LLM Evaluation
                analysis = self.llm_advisor.evaluate_incident(inc)

                # Multimodal Quantized Embedding & Qdrant Indexing on CPU
                try:
                    summary_text = (
                        f"Species: {inc['species']}, Threat Level: {analysis['threat_level']}, "
                        f"Threat Score: {analysis['threat_score']}, Zone: {cam_info['zone']}, "
                        f"Summary: {analysis['situation_summary']}"
                    )
                    multimodal_vector = self.embedding_engine.embed_multimodal_event(
                        text_context=summary_text,
                        image_path=snapshot_filepath
                    )
                    self.vector_store.insert_incident(
                        vector=multimodal_vector,
                        incident_data=inc,
                        llm_eval=analysis,
                        camera_info=cam_info,
                        snapshot_path=snapshot_url
                    )
                except Exception as e:
                    print(f"[LiveStreamManager] Error indexing vector to Qdrant: {e}")

                self.latest_alert = {
                    "species": inc["species"],
                    "threat_level": analysis["threat_level"],
                    "threat_score": analysis["threat_score"],
                    "siren_triggered": analysis["trigger_siren"],
                    "timestamp": inc["timestamp"],
                    "location": f"{cam_info['name']} ({cam_info['zone']})",
                    "summary": analysis["situation_summary"],
                    "snapshot_path": snapshot_url
                }

            # If there's an active threat banner, draw it right on frame bottom
            if self.latest_alert and self.latest_alert.get("threat_level") in ["CRITICAL", "HIGH"]:
                b_overlay = annotated_frame.copy()
                cv2.rectangle(b_overlay, (20, h - 55), (w - 20, h - 12), (0, 0, 180), -1)
                cv2.addWeighted(b_overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                alert_line1 = f"HIGH RISK: {self.latest_alert['species'].upper()} DETECTED NEAR SETTLEMENT"
                cv2.putText(annotated_frame, alert_line1, (35, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

            # Store latest frame in memory for quick snapshots
            self.latest_frame = annotated_frame

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Maintain smooth ~25 FPS stream
            time.sleep(0.04)
