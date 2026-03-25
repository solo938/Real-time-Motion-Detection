import time
from dataclasses import dataclass, field
from typing import Optional
 
 
@dataclass
class _Track:
    label: str
    start_time: float
    last_seen: float
    latest_depth_z: float = 0.5
    ended: bool = False
 
 
class ActionTimer:
    
 
    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout = timeout_seconds
        self._tracks: dict[str, _Track] = {}
 
    def update(
        self,
        detections,          
        now: Optional[float] = None,
    ) -> list[dict]:
        """
        Call once per frame.  Returns a list of ActionEvent dicts.
        """
        if now is None:
            now = time.time()

        active_labels: dict[str, float] = {}
        for det in detections:
            # use track_id if available so two people get separate timers
            tid = getattr(det, 'track_id', None)
            lbl = f"{det.label_name}#{tid}" if tid is not None else det.label_name
            if lbl not in active_labels or det.depth_z < active_labels[lbl]:
                active_labels[lbl] = det.depth_z
 
        events = []
 
        # Update existing tracks
        for label, track in list(self._tracks.items()):
            if label in active_labels:
                track.last_seen = now
                track.latest_depth_z = active_labels[label]
                events.append(self._make_event(track, now, "active"))
            else:
                elapsed_since_seen = now - track.last_seen
                if elapsed_since_seen > self.timeout:
                    # Track has timed out → emit ended event, then remove
                    events.append(self._make_event(track, track.last_seen, "ended"))
                    del self._tracks[label]
                
 

        for label, depth_z in active_labels.items():
            if label not in self._tracks:
                self._tracks[label] = _Track(
                    label=label,
                    start_time=now,
                    last_seen=now,
                    latest_depth_z=depth_z,
                )
                events.append(self._make_event(self._tracks[label], now, "active"))
 
        return events
 
    def get_active(self, now: Optional[float] = None) -> list[dict]:
        """Return current state of all active tracks (no side effects)."""
        if now is None:
            now = time.time()
        return [self._make_event(t, now, "active") for t in self._tracks.values()]
 
    def reset(self):
        """Clear all tracks (call between videos)."""
        self._tracks.clear()
 
    @staticmethod
    def _make_event(track: _Track, now: float, state: str) -> dict:
        return {
            "label": track.label,
            "state": state,
            "start_time": round(track.start_time, 3),
            "duration": round(now - track.start_time, 3),
            "depth_z": round(track.latest_depth_z, 4),
        }