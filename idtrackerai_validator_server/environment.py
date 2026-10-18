"""Incubator temperature and humidity at a frame of the loaded experiment, from flyhostel's
environment database (`python -m flyhostel.data.environment sync` fills it)."""
import datetime
import logging
import os
from threading import Lock

from flyhostel.data.environment import ENVIRONMENT_DB, load_readings, interpolate
from flyhostel.utils.utils import get_frame_time_origin

logger = logging.getLogger(__name__)

# Readings loaded per experiment: from its frame_time origin to this many days later
SPAN_DAYS = 30


class IncubatorEnvironment:
    def __init__(self, db=ENVIRONMENT_DB):
        self.db = db
        self.origin = None
        self.readings = None
        self.mtime = None
        self.lock = Lock()

    def open(self, basedir):
        """frame_time 0 of the experiment in `basedir` (unix s, UTC); None if unknown"""
        with self.lock:
            self.readings = None
            try:
                origin = get_frame_time_origin(basedir)
                self.origin = origin.replace(tzinfo=datetime.timezone.utc).timestamp()
            except Exception as error:
                logger.warning("No frame time origin for %s, no temperature/humidity: %s", basedir, error)
                self.origin = None

    def _load(self):
        """Readings of the experiment, again when the database changed (ongoing experiments)"""
        try:
            mtime = os.path.getmtime(self.db)
        except OSError:
            return None
        if self.readings is None or mtime != self.mtime:
            self.readings = load_readings(self.origin, self.origin + SPAN_DAYS * 86400, db=self.db)
            self.mtime = mtime
        return self.readings

    def at(self, frame_time):
        """{temperature, humidity} at frame_time (ms), None where unknown"""
        out = {"temperature": None, "humidity": None}
        if frame_time is None or self.origin is None:
            return out
        with self.lock:
            try:
                readings = self._load()
            except Exception as error:
                logger.warning("Cannot read %s: %s", self.db, error)
                return out
        if readings is None:
            return out
        values = interpolate(readings, [self.origin + frame_time / 1000])
        for key, value in values.items():
            value = float(value[0])
            out[key] = None if value != value else round(value, 2)
        return out
