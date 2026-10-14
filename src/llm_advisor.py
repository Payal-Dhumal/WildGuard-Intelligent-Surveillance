import json
import os

class LLMAdvisor:
    """
    Intelligent Reasoning Engine for WildGuard AI.
    Analyzes incident packets using an LLM (Gemini / OpenAI / Local rule-grounded engine)
    to generate tactical situational advisories and structured threat ratings.
    """

    def __init__(self, api_key=None):
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("models/gemini-3.8-flash")
                print("Connected to Gemini 3.8 Flash LLM.")
            except Exception as e:
                print(f"Gemini client initialization failed ({e}), falling back to grounded rule engine.")

    def evaluate_incident(self, incident: dict) -> dict:
        """
        Takes an incident packet and returns a comprehensive, structured evaluation.
        """
        # If live LLM API is available, ask the LLM
        if self.client:
            try:
                return self._call_gemini(incident)
            except Exception as e:
                print(f"LLM API call error: {e}. Using deterministic reasoning engine.")

        # Robust, deterministic fallback reasoning engine (Zero API cost & 100% reliable)
        return self._rule_grounded_evaluation(incident)

    def _call_gemini(self, incident: dict) -> dict:
        prompt = f"""
You are the Tactical Dispatch Officer for WildGuard AI, an intelligent wildlife alert system protecting rural communities and forest reserves.
Analyze the following verified wildlife incident and return a strict JSON response:

Incident Data:
- Species: {incident['species']} ({incident['category']})
- Max Confidence: {incident['max_confidence']}
- Location / Zone: {incident['camera_zone']}
- Time: {incident['timestamp']} (Night-time: {incident['is_night']})
- Duration spotted: {incident['duration_seconds']} seconds

Respond ONLY with valid JSON in this exact structure:
{{
  "threat_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "threat_score": 1 to 10,
  "trigger_siren": true | false,
  "situation_summary": "1-2 sentence tactical briefing of the situation.",
  "ranger_instructions": "Direct tactical instructions for forest rangers.",
  "community_broadcast": "Short, clear advisory for villagers / public broadcast."
}}
"""
        response = self.client.generate_content(prompt)
        text = response.text.strip()
        # Clean markdown codeblocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

    def _rule_grounded_evaluation(self, incident: dict) -> dict:
        """
        Deterministic, professional reasoning engine based on wildlife conflict protocols.
        """
        species = incident["species"]
        is_night = incident["is_night"]
        zone = incident["camera_zone"]
        conf = incident["max_confidence"]

        if species in ["Tiger", "Lion", "Jaguar"]:
            score = 10 if is_night else 9
            threat_level = "CRITICAL"
            trigger_siren = True
            summary = (
                f"Apex predator ({species}) confirmed in {zone} with {conf*100:.1f}% confidence. "
                f"{'Night-time sighting indicates active prowling near human borders.' if is_night else 'Daytime presence signals potential territorial encroachment.'}"
            )
            ranger_instructions = (
                f"1. Dispatch Rapid Response Unit to {zone}.\n"
                f"2. Activate perimeter spotlight array.\n"
                f"3. Prepare tranquilizer and non-lethal acoustic deterrents.\n"
                f"4. Establish a 500-meter safety cordon."
            )
            community_broadcast = (
                f"EMERGENCY ALERT: Confirmed {species} sighted near {zone}. "
                f"Remain inside homes immediately. Bring all domestic livestock indoors. Do not venture into boundary trails."
            )

        elif species in ["Cheetah", "Elephant"]:
            score = 8 if species == "Elephant" else 7
            threat_level = "HIGH"
            trigger_siren = True
            summary = (
                f"{species} spotted in {zone} ({conf*100:.1f}% confidence). "
                f"{'High risk of crop depredation or accidental human trampling.' if species == 'Elephant' else 'High speed predator active in boundary zone.'}"
            )
            ranger_instructions = (
                f"1. Send patrol vehicle to monitor herd/individual trajectory.\n"
                f"2. Utilize low-frequency acoustic horns if moving toward village farmland."
            )
            community_broadcast = (
                f"HIGH ALERT: {species} active near {zone}. Avoid farm perimeters and stay vigilant."
            )

        elif species in ["Rhino", "Buffalo"]:
            score = 5
            threat_level = "MEDIUM"
            trigger_siren = False
            summary = f"Large herbivore ({species}) detected in {zone}. Defensive aggression risk if startled."
            ranger_instructions = "Log coordinates and ensure boundary cattle fences remain intact."
            community_broadcast = f"ADVISORY: {species} present in {zone}. Maintain safe viewing distance."

        else:
            score = 2
            threat_level = "LOW"
            trigger_siren = False
            summary = f"{species} detected in natural habitat. Routine wildlife movement."
            ranger_instructions = "Standard logging for biodiversity records. No tactical intervention required."
            community_broadcast = "Normal status. No action required."

        return {
            "threat_level": threat_level,
            "threat_score": score,
            "trigger_siren": trigger_siren,
            "situation_summary": summary,
            "ranger_instructions": ranger_instructions,
            "community_broadcast": community_broadcast
        }
