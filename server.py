import os
import sys
import shutil
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from chatbot import WildlifeChatbot
from vector_store import WildGuardVectorStore
from live_stream import LiveStreamManager

app = FastAPI(title="WildGuard AI Command Center")

# Ensure required folders exist
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("output/evidence", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Initialize Chatbot, VectorStore, and LiveStreamManager
vector_store = WildGuardVectorStore(storage_path="data/qdrant_db")
bot = WildlifeChatbot(storage_path="data/qdrant_db")
streamer = LiveStreamManager("models/wildlife_multi_yolov8.pt")

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = "default"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    incidents = vector_store.get_all_incidents()
    counts = vector_store.get_incident_counts()
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "incidents": incidents,
            "total": counts["total"],
            "high_risk": counts["high_risk"],
            "medium_risk": counts["medium_risk"],
            "low_risk": counts["low_risk"]
        }
    )

# 1. LIVE CAMERA MJPEG STREAM ENDPOINT
@app.get("/video_feed")
async def video_feed():
    """
    Continuous live video stream with real-time YOLO tracking
    and on-screen camera metadata.
    """
    return StreamingResponse(
        streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 2. UPLOAD VIDEO ENDPOINT
@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), camera_id: str = Form("CAM_01")):
    """
    Upload an MP4 or AVI video to immediately analyze and stream on the dashboard.
    """
    import re
    safe_filename = re.sub(r'[^\w\.-]', '_', file.filename)
    file_path = os.path.join("uploads", safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Switch active streamer source to the uploaded video using absolute path
    abs_path = os.path.abspath(file_path)
    streamer.set_source(abs_path, cam_id=camera_id)
    return JSONResponse({
        "status": "success",
        "message": f"Successfully loaded '{safe_filename}' for camera '{camera_id}'!",
        "video_path": file_path
    })

# 3. SWITCH TO WEBCAM OR SAMPLE VIDEO
@app.post("/api/switch_source")
async def switch_source(source_type: str = Form(...), camera_id: str = Form("CAM_01")):
    """
    source_type: 'webcam' or 'sample'
    """
    if source_type == "webcam":
        streamer.set_source(0, cam_id=camera_id)
        msg = "Live Webcam activated!"
    else:
        sample_path = "data/sample_wildlife.mp4"
        streamer.set_source(sample_path, cam_id=camera_id)
        msg = "Switched to Wildlife Camera Feed (Tiger Sector 4)!"

    return JSONResponse({"status": "success", "message": msg})

# 4. GEOFENCE API ENDPOINTS
class GeofenceRequest(BaseModel):
    points: list[list[int]]

@app.get("/api/first_frame")
async def get_first_frame():
    """
    Returns the first frame of the current video as JPEG for user polygon drawing.
    """
    frame_bytes = streamer.get_first_frame_bytes()
    if not frame_bytes:
        return JSONResponse({"status": "error", "message": "No frame available yet"}, status_code=404)
    from fastapi import Response
    return Response(content=frame_bytes, media_type="image/jpeg")

@app.get("/api/get_geofence")
async def get_geofence():
    """Returns current active polygon vertices."""
    return {"points": streamer.geofence_polygon}

@app.post("/api/set_geofence")
async def set_geofence(req: GeofenceRequest):
    """Saves user-drawn polygon vertices and resets tracking session."""
    try:
        pts = streamer.set_geofence(req.points)
        return {"status": "success", "points": pts}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.post("/api/reset_geofence")
async def reset_geofence():
    """Resets geofence polygon to default."""
    pts = streamer.reset_geofence()
    return {"status": "success", "points": pts}

# 5. GET LATEST ALERT AND REAL-TIME STATS
@app.get("/api/latest_alert")
async def get_latest_alert():
    """
    Returns the most recent incident and active alert state
    for real-time dashboard updates without page reloads.
    """
    latest = vector_store.get_latest_incident()
    counts = vector_store.get_incident_counts()
    
    return {
        "latest": latest,
        "total": counts["total"],
        "high_risk": counts["high_risk"],
        "medium_risk": counts["medium_risk"],
        "low_risk": counts["low_risk"],
        "active_streamer_alert": streamer.latest_alert,
        "geofence_polygon": streamer.geofence_polygon
    }

# 6. CHAT ENDPOINT (GEMINI 3.8 FLASH & CONTEXTUAL RANGER RAG)
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    sess_id = req.session_id or "default"
    answer = bot.ask(req.question, session_id=sess_id)
    return {"answer": answer}

@app.post("/api/chat/reset")
async def chat_reset(session_id: str = "default"):
    bot.reset_session(session_id)
    return {"status": "success", "message": f"Session '{session_id}' reset."}

# 7. RECENT INCIDENTS ENDPOINT
@app.get("/api/incidents")
async def get_incidents():
    return vector_store.get_all_incidents()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
