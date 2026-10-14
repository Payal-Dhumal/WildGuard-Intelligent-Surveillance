import time
from datetime import datetime

class IncidentManager:
    """
    Manages real-time wildlife tracking events in memory.
    
    Guarantees:
    1. In-memory state tracking only (zero deduplication state persisted to DB).
    2. Single-alarm per continuous danger-zone intrusion session.
    3. If the animal exits the danger zone / camera view and re-enters after exit_grace_period,
       the previous intrusion is marked as exited, the risk is reassessed, and a new alarm is allowed.
    4. Eliminates duplicate insights and alarms for the same animal within a continuous sighting.
    """

    DANGEROUS_SPECIES = {
        "Tiger": {"type": "Apex Predator", "base_threat": "CRITICAL"},
        "Lion": {"type": "Apex Predator", "base_threat": "CRITICAL"},
        "Jaguar": {"type": "Apex Predator", "base_threat": "CRITICAL"},
        "Cheetah": {"type": "Predator", "base_threat": "HIGH"},
        "Elephant": {"type": "Megaherbivore / Herd", "base_threat": "HIGH"},
        "Rhino": {"type": "Megaherbivore", "base_threat": "MEDIUM"},
        "Buffalo": {"type": "Large Herbivore", "base_threat": "MEDIUM"},
    }

    HARMLESS_SPECIES = {"Zebra", "Fox", "Panda", "Deer", "Dog", "Cat"}

    def __init__(self, min_conf=0.45, alarm_conf=0.50, min_frames=1, exit_grace_period=2.0):
        """
        :param min_conf: Minimum confidence to consider detection valid
        :param alarm_conf: Confidence threshold required to trigger an alarm
        :param min_frames: Minimum frames in danger zone before firing an alarm (1 = immediate)
        :param exit_grace_period: Seconds of absence before an animal is confirmed to have
                                  'exited' the danger zone. Once exited, re-entry triggers a new intrusion.
        """
        self.min_conf = min_conf
        self.alarm_conf = alarm_conf
        self.min_frames = min_frames
        self.exit_grace_period = exit_grace_period

        # In-memory tracking state:
        # { track_id: {
        #     "species": str,
        #     "state": "INTRUSION_ACTIVE" | "ALARMED" | "EXITED",
        #     "consecutive_frames": int,
        #     "max_conf": float,
        #     "entry_time": float,
        #     "last_seen_time": float,
        #     "camera_zone": str,
        #     "intrusion_session": int
        #   }
        # }
        self.tracked_animals = {}
        self.session_intrusion_counter = 0

    def reset_session(self):
        """
        Resets in-memory tracking state for a new video processing session.
        """
        self.tracked_animals.clear()
        self.session_intrusion_counter = 0

    def update(self, detections, camera_zone="Sector 4 - Village North Border", frame_width=640, frame_height=480):
        """
        Processes detections for the current frame.
        detections format: list of dicts:
        [{'track_id': int, 'species': str, 'confidence': float, 'box': [x1, y1, x2, y2], 'in_danger_zone': bool}]
        
        Returns a list of newly triggered incident packets (if any).
        """
        now = time.time()
        triggered_incidents = []
        current_frame_track_ids = set()

        for det in detections:
            track_id = det.get('track_id')
            species = det.get('species')
            conf = det.get('confidence')
            box = det.get('box', None)
            in_danger_zone = det.get('in_danger_zone', True)

            if conf < self.min_conf or track_id is None:
                continue

            # Check if this animal is already known in memory
            animal = self.tracked_animals.get(track_id)

            if in_danger_zone:
                # Animal is INSIDE the Virtual Polygon Geofence!
                current_frame_track_ids.add(track_id)

                if animal is None:
                    # Initial entry into danger zone - trigger immediately
                    self.session_intrusion_counter += 1
                    self.tracked_animals[track_id] = {
                        "species": species,
                        "state": "INTRUSION_ACTIVE",
                        "consecutive_frames": 1,
                        "max_conf": conf,
                        "entry_time": now,
                        "last_seen_time": now,
                        "camera_zone": camera_zone,
                        "intrusion_session": self.session_intrusion_counter,
                        "last_box": box
                    }
                    animal = self.tracked_animals[track_id]
                elif animal["state"] == "EXITED":
                    # Re-entry into polygon danger zone after confirmed exit!
                    self.session_intrusion_counter += 1
                    animal["state"] = "INTRUSION_ACTIVE"
                    animal["consecutive_frames"] = 1
                    animal["max_conf"] = conf
                    animal["entry_time"] = now
                    animal["last_seen_time"] = now
                    animal["camera_zone"] = camera_zone
                    animal["intrusion_session"] = self.session_intrusion_counter
                    animal["last_box"] = box
                else:
                    # Continuous intrusion inside danger zone
                    animal["consecutive_frames"] += 1
                    animal["max_conf"] = max(animal["max_conf"], conf)
                    animal["last_seen_time"] = now
                    animal["last_box"] = box

                # Evaluate single-alarm condition inside danger zone:
                # Trigger ALARM IMMEDIATELY when a dangerous animal crosses into danger zone
                if (
                    animal["state"] == "INTRUSION_ACTIVE"
                    and animal["species"] in self.DANGEROUS_SPECIES
                    and animal["consecutive_frames"] >= self.min_frames
                    and animal["max_conf"] >= self.alarm_conf
                ):
                    animal["state"] = "ALARMED"
                    incident_packet = self._create_incident_packet(track_id, animal)
                    triggered_incidents.append(incident_packet)
            else:
                # Animal is OUTSIDE the Danger Zone polygon.
                # If it was previously tracked inside, treat absence from danger zone as stepping out
                pass

        # Handle exit detection for animals not currently in the danger zone
        for tid, animal in list(self.tracked_animals.items()):
            if tid not in current_frame_track_ids:
                time_since_last_seen = now - animal["last_seen_time"]
                
                # If absent from the danger zone for longer than exit_grace_period, mark as EXITED
                if time_since_last_seen >= self.exit_grace_period:
                    if animal["state"] in ["INTRUSION_ACTIVE", "ALARMED"]:
                        animal["state"] = "EXITED"

        return triggered_incidents

    def _create_incident_packet(self, track_id, animal_data):
        now_dt = datetime.now()
        species = animal_data["species"]
        profile = self.DANGEROUS_SPECIES.get(species, {"type": "Unknown", "base_threat": "LOW"})

        return {
            "incident_id": f"INC-{int(time.time())}-{track_id}-S{animal_data['intrusion_session']}",
            "track_id": track_id,
            "species": species,
            "category": profile["type"],
            "base_threat": profile["base_threat"],
            "max_confidence": round(animal_data["max_conf"], 3),
            "persistence_frames": animal_data["consecutive_frames"],
            "duration_seconds": round(animal_data["last_seen_time"] - animal_data["entry_time"], 1),
            "camera_zone": animal_data["camera_zone"],
            "intrusion_session": animal_data["intrusion_session"],
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour_of_day": now_dt.hour,
            "is_night": now_dt.hour < 6 or now_dt.hour >= 18
        }
