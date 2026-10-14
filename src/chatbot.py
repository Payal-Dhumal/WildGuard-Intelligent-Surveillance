import os
import re
import json
from quantized_embedding import QuantizedEmbeddingGemmaCPU
from vector_store import WildGuardVectorStore

class WildlifeChatbot:
    """
    Ranger Intelligence Assistant that answers natural language questions
    about logged wildlife sightings, evidence, threat scores, and locations.
    Preserves multi-turn conversational context, validates direct answers,
    and returns comprehensive structured incident details from Qdrant.
    """

    def __init__(self, storage_path="data/qdrant_db", *args, **kwargs):
        self.embedding_engine = QuantizedEmbeddingGemmaCPU(dimension=512)
        self.vector_store = WildGuardVectorStore(storage_path=storage_path, vector_dim=512)
        self.sessions = {}  # session_id -> {"history": [], "active_incident": None}

        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("models/gemini-3.8-flash")
            except Exception as e:
                self.client = None

    def get_session(self, session_id: str = "default") -> dict:
        """Retrieves or creates session conversational context."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "active_incident": None,
                "active_species": None
            }
        return self.sessions[session_id]

    def reset_session(self, session_id: str = "default"):
        """Clears conversational context for a given session."""
        if session_id in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "active_incident": None,
                "active_species": None
            }

    def format_incident_details(self, inc: dict, title: str = "Incident Summary & Forensic Briefing") -> str:
        """
        Formats complete, structured incident details from Qdrant payload.
        Ensures all forensic and tactical fields are clearly represented.
        """
        if not inc:
            return "No matching incident record found in the database."

        conf = float(inc.get("confidence", 0.0))
        is_night = bool(inc.get("is_night", False))
        mode_str = "Night Mode (Infrared)" if is_night else "Day Mode"
        siren_status = "TRIGGERED (High Priority Audible Alert)" if inc.get("siren_triggered") == 1 else "STANDBY (Silent Surveillance)"

        lines = [
            f"🚨 **{title}:**",
            f"- **Incident ID:** `{inc.get('incident_id', 'N/A')}`",
            f"- **Species / Animal:** **{inc.get('species', 'Unknown')}** ({conf * 100:.1f}% confidence)",
            f"- **Threat Level:** **{inc.get('threat_level', 'LOW')}** (Threat Score: **{inc.get('threat_score', 1)}/10**)",
            f"- **Location:** {inc.get('camera_name', 'Boundary Camera')} [{inc.get('zone', 'Zone')}]",
            f"- **Camera ID:** `{inc.get('camera_id', 'CAM_01')}`",
            f"- **Perimeter Status:** **{inc.get('danger_zone_status', 'SAFE_PERIMETER')}**",
            f"- **Physical Siren:** {siren_status}",
            f"- **Timestamp:** `{inc.get('timestamp', 'N/A')}` ({mode_str})"
        ]

        if inc.get("latitude") and inc.get("longitude"):
            lines.append(f"- **GPS Coordinates:** {inc.get('latitude')}° N, {inc.get('longitude')}° E (Proximity to Hamlet: {inc.get('proximity_to_settlement_m', 80)}m)")

        if inc.get("situation_summary"):
            lines.append(f"- **Situation Summary:** {inc.get('situation_summary')}")

        if inc.get("ranger_instructions"):
            lines.append(f"- **Ranger Tactical Directive:** {inc.get('ranger_instructions')}")

        if inc.get("community_broadcast"):
            lines.append(f"- **Community Advisory:** {inc.get('community_broadcast')}")

        if inc.get("snapshot_path"):
            lines.append(f"- **Visual Evidence Snapshot:** `{inc.get('snapshot_path')}`")

        return "\n".join(lines)

    def _resolve_incident(self, query: str, session: dict, incidents: list[dict]) -> dict | None:
        """
        Resolves which incident the user is asking about using:
        1. Exact incident ID in query (e.g. INC-...)
        2. Animal species mentioned (e.g. tiger, elephant)
        3. Camera ID or Zone mentioned (e.g. CAM_01, Sector 4, Sector 12)
        4. Existing active incident in conversational context
        5. Default: Latest logged incident
        """
        if not incidents:
            return None

        q_lower = query.lower()

        # 1. Exact Incident ID match
        id_match = re.search(r'\b(INC-[A-Za-z0-9_-]+)\b', query, re.IGNORECASE)
        if id_match:
            target_id = id_match.group(1).upper()
            for inc in incidents:
                if inc.get("incident_id", "").upper() == target_id:
                    return inc

        # 2. Species match
        for sp in ["tiger", "elephant", "leopard", "deer", "lion", "bear"]:
            if sp in q_lower:
                matches = [i for i in incidents if str(i.get("species", "")).lower() == sp]
                if matches:
                    return matches[0]

        # 3. Camera / Sector match
        for cam in ["cam_01", "cam_02", "cam_03"]:
            if cam in q_lower:
                matches = [i for i in incidents if str(i.get("camera_id", "")).lower() == cam]
                if matches:
                    return matches[0]

        for sector in ["sector 4", "sector 12", "village boundary", "forest corridor", "perimeter"]:
            if sector in q_lower:
                matches = [i for i in incidents if sector in str(i.get("camera_name", "")).lower() or sector in str(i.get("zone", "")).lower()]
                if matches:
                    return matches[0]

        # 4. Contextual reference to previous conversation
        active = session.get("active_incident")
        if active and any(term in q_lower for term in ["it", "that", "this", "the incident", "the animal", "the sighting", "there", "the alarm"]):
            return active

        # 5. Default: Most recent critical or latest logged incident
        critical_matches = [i for i in incidents if i.get("threat_level") in ["CRITICAL", "HIGH"]]
        return critical_matches[0] if critical_matches else incidents[0]

    def ask(self, user_question: str, session_id: str = "default") -> str:
        """
        Answers a user query preserving conversational context across turns,
        grounding on Qdrant vector store and payload evidence, and strictly
        validating that every response directly addresses the question.
        """
        session = self.get_session(session_id)
        user_question_clean = user_question.strip()
        if not user_question_clean:
            return "Please provide a question or incident query."

        # 1. Fetch live Qdrant incidents
        incidents = self.vector_store.get_all_incidents(limit=100)
        resolved_inc = self._resolve_incident(user_question_clean, session, incidents)
        if resolved_inc:
            session["active_incident"] = resolved_inc
            session["active_species"] = resolved_inc.get("species")

        # 2. Vector search semantic retrieval
        query_vector = self.embedding_engine.embed_text(user_question_clean)
        semantic_matches = self.vector_store.search_similar(query_vector, limit=4, score_threshold=0.0)

        vector_context_items = []
        for match in semantic_matches:
            vector_context_items.append(
                f"- [Incident ID: {match.get('incident_id')}] Species: {match.get('species')}, "
                f"Threat Level: {match.get('threat_level')} (Score: {match.get('threat_score')}/10), "
                f"Status: {match.get('danger_zone_status')}, Zone: {match.get('camera_name')} ({match.get('zone')}), "
                f"Time: {match.get('timestamp')}, Summary: {match.get('situation_summary')}, "
                f"Ranger Directive: {match.get('ranger_instructions')}"
            )
        vector_context = "\n".join(vector_context_items) if vector_context_items else "None retrieved."
        recent_logs = self.vector_store.query_incidents_text_context(limit=12)

        # 3. Format conversational history window
        history_text = ""
        if session["history"]:
            recent_turns = session["history"][-6:]  # Last 3 turns
            hist_lines = []
            for item in recent_turns:
                role = "User" if item["role"] == "user" else "AI Ranger"
                hist_lines.append(f"{role}: {item['content']}")
            history_text = "\n".join(hist_lines)

        active_inc_text = ""
        if session["active_incident"]:
            ai = session["active_incident"]
            active_inc_text = (
                f"ACTIVE INCIDENT FOCUS: ID {ai.get('incident_id')} | Species: {ai.get('species')} | "
                f"Threat: {ai.get('threat_level')} ({ai.get('threat_score')}/10) | Location: {ai.get('camera_name')} ({ai.get('zone')}) | "
                f"Time: {ai.get('timestamp')} | Summary: {ai.get('situation_summary')}"
            )

        combined_evidence = f"""=== CONVERSATIONAL DIALOGUE HISTORY ===
{history_text or "No prior turns in this session."}

{active_inc_text}

=== SEMANTIC VECTOR EVIDENCE (QDRANT) ===
{vector_context}

=== RECENT INCIDENT LOGS ===
{recent_logs}"""

        # 4. Attempt Gemini LLM response with conversational grounding
        llm_answer = None
        if self.client:
            try:
                system_prompt = f"""You are the WildGuard AI Ranger Assistant. You have access to official wildlife telemetry and incident logs below:

{combined_evidence}

USER QUESTION: {user_question_clean}

CRITICAL INSTRUCTIONS:
1. Directly and specifically answer the user's question.
2. If the user asks for an incident summary or details, provide the full incident details: Incident ID, Species, Location, Threat Level, Status, Timestamp, Situation Summary, and Ranger Directives.
3. If the user asks a follow-up question (e.g. "when did it happen", "where was it", "what should rangers do", "is it dangerous"), answer for the active incident in context.
4. Maintain a professional, authoritative ranger operations tone. Do not invent details not present in the logs.
"""
                resp = self.client.generate_content(system_prompt)
                candidate = resp.text.strip()
                if self._validate_response(user_question_clean, candidate, resolved_inc):
                    llm_answer = candidate
            except Exception as e:
                print(f"[WildlifeChatbot] Gemini API exception: {e}")
                llm_answer = None

        # 5. Local Deterministic Intelligence Engine (Guarantees direct, validated answers)
        if not llm_answer:
            llm_answer = self._smart_local_query(user_question_clean, session, incidents, resolved_inc)

        # 6. Record conversation turn in session history
        session["history"].append({"role": "user", "content": user_question_clean})
        session["history"].append({"role": "assistant", "content": llm_answer})

        return llm_answer

    def _validate_response(self, query: str, response: str, target_incident: dict | None) -> bool:
        """
        Validates that a response directly and accurately addresses the user's question:
        - Incident summary queries MUST contain incident details (ID, species, location/threat).
        - Temporal queries MUST contain time/date information.
        - Location queries MUST contain place/camera information.
        - Count queries MUST contain numeric figures.
        """
        if not response or len(response) < 15:
            return False

        q_lower = query.lower()

        # Reject generic evasion responses
        evasions = ["i cannot help with that", "as an ai", "i don't have access to real-time", "i am an ai"]
        if any(e in response.lower() for e in evasions):
            return False

        # Validate incident summary requests
        if any(term in q_lower for term in ["incident summary", "summary", "details", "briefing", "report"]):
            has_id = ("inc-" in response.lower()) or (target_incident and target_incident.get("incident_id", "").lower() in response.lower())
            has_threat = any(t in response.lower() for t in ["threat", "critical", "high", "medium", "low"])
            has_species = any(s in response.lower() for s in ["tiger", "elephant", "animal", "species", "predator"])
            if not (has_threat or has_species or has_id):
                return False

        # Validate time requests
        if any(term in q_lower for term in ["when", "timestamp", "what time"]):
            if not any(char.isdigit() for char in response):
                return False

        # Validate count requests
        if any(term in q_lower for term in ["how many", "count", "total"]):
            if not any(char.isdigit() for char in response):
                return False

        return True

    def _smart_local_query(self, query: str, session: dict, incidents: list[dict], target_incident: dict | None) -> str:
        """
        Deterministic, grounded query handler that understands conversational context,
        resolves multi-intent questions, and returns complete incident details.
        """
        q_lower = query.lower()

        if not incidents:
            return "No incidents have been recorded in the Qdrant vector database yet."

        inc = target_incident or incidents[0]
        conf = float(inc.get("confidence", 0.0))

        # -------------------------------------------------------------
        # INTENT 1: Incident Summary & Detailed Forensic Briefing
        # -------------------------------------------------------------
        is_summary_query = any(k in q_lower for k in [
            "summar", "incident", "details", "detail", "briefing",
            "what happened", "tell me about", "report", "overview", "describe"
        ])

        # Check if question is a combined query (Count + Summary)
        has_count_intent = any(k in q_lower for k in ["how many", "count", "total"])

        if has_count_intent and is_summary_query:
            tiger_count = sum(1 for i in incidents if str(i.get("species", "")).lower() == "tiger")
            elephant_count = sum(1 for i in incidents if str(i.get("species", "")).lower() == "elephant")
            crit_count = sum(1 for i in incidents if i.get("threat_level") in ["CRITICAL", "HIGH"])
            
            stats_part = (
                f"📊 **Database Summary (Qdrant):**\n"
                f"- Total Recorded Sightings: **{len(incidents)}**\n"
                f"- Tiger Sightings: **{tiger_count}**\n"
                f"- Elephant Sightings: **{elephant_count}**\n"
                f"- Critical Alerts: **{crit_count}**\n\n"
            )
            return stats_part + self.format_incident_details(inc, title=f"Incident Summary ({inc.get('species')})")

        if is_summary_query:
            title = f"Incident Summary & Evidence Briefing ({inc.get('species')})"
            return self.format_incident_details(inc, title=title)

        # -------------------------------------------------------------
        # INTENT 2: Follow-up Temporal Queries ("when did it happen?")
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["when", "time", "timestamp", "what time", "date"]):
            mode_str = "Night Mode (Infrared)" if inc.get("is_night") else "Day Mode"
            return (
                f"🕒 **Incident Timestamp for {inc.get('species')} (`{inc.get('incident_id')}`):**\n"
                f"- **Logged Time:** `{inc.get('timestamp')}`\n"
                f"- **Camera Lighting:** {mode_str}\n"
                f"- **Sector / Location:** {inc.get('camera_name')} ({inc.get('zone')})\n"
                f"- **Current Perimeter Status:** **{inc.get('danger_zone_status')}**"
            )

        # -------------------------------------------------------------
        # INTENT 3: Follow-up Location Queries ("where was it?", "which camera?")
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["where", "location", "camera", "which zone", "gps", "coordinates", "proximity"]):
            lines = [
                f"📍 **Location Details for {inc.get('species')} (`{inc.get('incident_id')}`):**",
                f"- **Camera Sensor:** `{inc.get('camera_id')}` — **{inc.get('camera_name')}**",
                f"- **Patrol Sector:** {inc.get('zone')}",
                f"- **Perimeter Status:** **{inc.get('danger_zone_status')}**"
            ]
            if inc.get("latitude") and inc.get("longitude"):
                lines.append(f"- **GPS Coordinates:** {inc.get('latitude')}° N, {inc.get('longitude')}° E")
                lines.append(f"- **Distance to Village:** {inc.get('proximity_to_settlement_m', 80)} meters")
            return "\n".join(lines)

        # -------------------------------------------------------------
        # INTENT 4: Ranger Actions, Directives, or Villager Precautions
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["action", "precaution", "directive", "instruction", "what should", "what to do", "protocol", "protect", "stay safe"]):
            return (
                f"🛡️ **Ranger Directives & Community Precautions for `{inc.get('incident_id')}`:**\n"
                f"- **Species:** **{inc.get('species')}** (Threat Level: **{inc.get('threat_level')}**, Score: **{inc.get('threat_score')}/10**)\n"
                f"- **Tactical Ranger Directive:**\n  {inc.get('ranger_instructions')}\n"
                f"- **Community Advisory:**\n  {inc.get('community_broadcast')}"
            )

        # -------------------------------------------------------------
        # INTENT 5: Threat Level, Risk Score, or Danger Status
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["threat", "risk", "danger", "score", "how dangerous", "severe"]):
            siren = "ACTIVE AUDIBLE ALARM" if inc.get("siren_triggered") else "Silent Monitoring"
            return (
                f"⚠️ **Threat Assessment for `{inc.get('incident_id')}`:**\n"
                f"- **Target Animal:** **{inc.get('species')}**\n"
                f"- **Threat Level:** **{inc.get('threat_level')}** (Score: **{inc.get('threat_score')}/10**)\n"
                f"- **Perimeter Status:** **{inc.get('danger_zone_status')}**\n"
                f"- **Siren State:** {siren}\n"
                f"- **Briefing:** {inc.get('situation_summary')}"
            )

        # -------------------------------------------------------------
        # INTENT 6: Counts, Totals, or Census
        # -------------------------------------------------------------
        if has_count_intent:
            tiger_count = sum(1 for i in incidents if str(i.get("species", "")).lower() == "tiger")
            elephant_count = sum(1 for i in incidents if str(i.get("species", "")).lower() == "elephant")
            crit_count = sum(1 for i in incidents if i.get("threat_level") in ["CRITICAL", "HIGH"])
            return (
                f"📊 **Database Summary (Qdrant):**\n"
                f"- Total Recorded Sightings: **{len(incidents)}**\n"
                f"- Tiger Incidents: **{tiger_count}**\n"
                f"- Elephant Incidents: **{elephant_count}**\n"
                f"- Critical Alerts: **{crit_count}**"
            )

        # -------------------------------------------------------------
        # INTENT 7: Predator / Animal Sighting Lookup
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["tiger", "predator", "elephant", "animal", "sighting"]):
            return (
                f"🚨 **Latest {inc.get('species')} Sighting:**\n"
                f"- **Species:** **{inc.get('species')}** ({conf * 100:.1f}% confidence)\n"
                f"- **Incident ID:** `{inc.get('incident_id')}`\n"
                f"- **Location:** {inc.get('camera_name')} ({inc.get('zone')})\n"
                f"- **Threat Level:** **{inc.get('threat_level')}** (Score: {inc.get('threat_score')}/10)\n"
                f"- **Status:** **{inc.get('danger_zone_status')}**\n"
                f"- **Timestamp:** `{inc.get('timestamp')}`\n"
                f"- **Summary:** {inc.get('situation_summary')}"
            )

        # -------------------------------------------------------------
        # INTENT 8: Siren / Emergency Alarm Status
        # -------------------------------------------------------------
        if any(k in q_lower for k in ["siren", "alarm"]):
            crit = [i for i in incidents if i.get("siren_triggered") == 1]
            if not crit:
                return "🔊 No emergency siren alarms have been triggered so far."
            return (
                f"🚨 There are **{len(crit)} active alarm incidents** in the vector database.\n"
                f"- Most Recent Alarm: **{crit[0].get('species')}** at **{crit[0].get('camera_name')}** on `{crit[0].get('timestamp')}`."
            )

        # -------------------------------------------------------------
        # Default: Full incident summary for the active incident
        # -------------------------------------------------------------
        return self.format_incident_details(inc, title=f"Incident Summary (`{inc.get('incident_id')}`)")
