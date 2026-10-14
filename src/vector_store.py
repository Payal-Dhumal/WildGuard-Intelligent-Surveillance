import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class WildGuardVectorStore:
    """
    Qdrant Vector Database Manager for WildGuard AI.
    Stores multimodal embeddings alongside structured incident payloads:
    - Species
    - Timestamps
    - Confidence score
    - Risk & Threat score
    - Danger Zone status
    - Snapshot paths
    """

    COLLECTION_NAME = "wildguard_incidents"
    _instance = None

    def __new__(cls, storage_path="data/qdrant_db", vector_dim=512):
        if cls._instance is None:
            cls._instance = super(WildGuardVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_path="data/qdrant_db", vector_dim=512):
        if getattr(self, "_initialized", False):
            return
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.vector_dim = vector_dim

        # Initialize embedded local Qdrant Client (No external Docker required!)
        self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection()
        self._initialized = True

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            print(f"[Qdrant] Collection '{self.COLLECTION_NAME}' created with Cosine distance (dim={self.vector_dim}).")

    def insert_incident(self, vector: list, incident_data: dict, llm_eval: dict, camera_info: dict, snapshot_path: str):
        """
        Idempotently inserts or updates an incident vector and payload in Qdrant.
        """
        incident_id = incident_data.get("incident_id", str(uuid.uuid4()))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, incident_id))

        payload = {
            "incident_id": incident_id,
            "species": incident_data.get("species", "Unknown"),
            "confidence": float(incident_data.get("max_confidence", incident_data.get("confidence", 0.0))),
            "timestamp": incident_data.get("timestamp", ""),
            "threat_level": llm_eval.get("threat_level", "LOW"),
            "threat_score": int(llm_eval.get("threat_score", 1)),
            "siren_triggered": 1 if llm_eval.get("trigger_siren") else 0,
            "danger_zone_status": "INSIDE_DANGER_ZONE" if llm_eval.get("trigger_siren") else "SAFE_PERIMETER",
            "camera_id": incident_data.get("camera_id", "CAM_01"),
            "camera_name": camera_info.get("name", "Village Boundary"),
            "zone": camera_info.get("zone", "Sector 4"),
            "latitude": camera_info.get("latitude", 26.0173),
            "longitude": camera_info.get("longitude", 92.9376),
            "proximity_to_settlement_m": camera_info.get("proximity_to_settlement_m", 80),
            "is_night": 1 if incident_data.get("is_night") else 0,
            "snapshot_path": snapshot_path or "",
            "situation_summary": llm_eval.get("situation_summary", ""),
            "ranger_instructions": llm_eval.get("ranger_instructions", ""),
            "community_broadcast": llm_eval.get("community_broadcast", "")
        }

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )
        print(f"[Qdrant] Successfully indexed vector & payload for Incident '{incident_id}' ({payload['species']}).")
        return payload

    def search_similar(self, query_vector: list, limit=5, score_threshold=0.0):
        """
        Performs semantic vector search across all indexed wildlife incidents.
        """
        try:
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return [point.payload for point in results.points]
        except Exception as e:
            print(f"[Qdrant] Search error: {e}")
            return []

    def get_all_incidents(self, limit=200) -> list[dict]:
        """
        Retrieves all logged incidents directly from Qdrant point payloads,
        sorted chronologically by timestamp (most recent first).
        """
        try:
            scroll_result, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            incidents = [pt.payload for pt in scroll_result if pt.payload]
            incidents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return incidents
        except Exception as e:
            print(f"[Qdrant] Error fetching incidents: {e}")
            return []

    def get_latest_incident(self) -> dict | None:
        """Returns the most recent incident payload from Qdrant."""
        incidents = self.get_all_incidents(limit=1)
        return incidents[0] if incidents else None

    def get_incident_counts(self) -> dict:
        """Calculates total, high, medium, and low risk summary stats from Qdrant payloads."""
        incidents = self.get_all_incidents(limit=500)
        total = len(incidents)
        high_risk = sum(1 for i in incidents if i.get("threat_level") in ["CRITICAL", "HIGH"])
        medium_risk = sum(1 for i in incidents if i.get("threat_level") == "MEDIUM")
        low_risk = sum(1 for i in incidents if i.get("threat_level") == "LOW")
        return {
            "total": total,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk
        }

    def query_incidents_text_context(self, limit=15) -> str:
        """Formats recent incident records from Qdrant into structured text for LLM grounding."""
        incidents = self.get_all_incidents(limit=limit)
        if not incidents:
            return "No incidents logged in the system yet."
        lines = []
        for inc in incidents:
            line = (
                f"[{inc.get('timestamp')}] ID: {inc.get('incident_id')} | "
                f"Species: {inc.get('species')} ({float(inc.get('confidence', 0))*100:.1f}%) | "
                f"Zone: {inc.get('camera_name')} ({inc.get('zone')}) | "
                f"Threat: {inc.get('threat_level')} ({inc.get('threat_score')}/10) | "
                f"Status: {inc.get('danger_zone_status')} | "
                f"Summary: {inc.get('situation_summary')}"
            )
            lines.append(line)
        return "\n".join(lines)
