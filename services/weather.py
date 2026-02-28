"""
weather.py — OpenWeatherMap integration for weather-based prediction features.

Set WEATHER_API_KEY env var (free: 1000 calls/day at openweathermap.org).
"""
import os
import json
import time
import logging

logger = logging.getLogger("criccric")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Average climate data by city + month (fallback when no API key)
CLIMATE_DEFAULTS = {
    "Mumbai": {"temp": 28, "humidity": 75, "wind": 12, "dew_city": True},
    "Kolkata": {"temp": 27, "humidity": 80, "wind": 10, "dew_city": True},
    "Chennai": {"temp": 30, "humidity": 70, "wind": 11, "dew_city": True},
    "Delhi": {"temp": 25, "humidity": 55, "wind": 9, "dew_city": False},
    "Bangalore": {"temp": 24, "humidity": 65, "wind": 8, "dew_city": False},
    "Hyderabad": {"temp": 27, "humidity": 60, "wind": 10, "dew_city": True},
    "Ahmedabad": {"temp": 28, "humidity": 50, "wind": 10, "dew_city": False},
    "Pune": {"temp": 26, "humidity": 55, "wind": 9, "dew_city": False},
    "Jaipur": {"temp": 26, "humidity": 45, "wind": 10, "dew_city": False},
    "London": {"temp": 15, "humidity": 70, "wind": 15, "dew_city": False},
    "Melbourne": {"temp": 20, "humidity": 55, "wind": 12, "dew_city": False},
    "Sydney": {"temp": 22, "humidity": 60, "wind": 11, "dew_city": False},
    "Dubai": {"temp": 35, "humidity": 40, "wind": 12, "dew_city": False},
    "Sharjah": {"temp": 34, "humidity": 45, "wind": 10, "dew_city": False},
    "Colombo": {"temp": 28, "humidity": 78, "wind": 8, "dew_city": True},
    "Dhaka": {"temp": 27, "humidity": 78, "wind": 7, "dew_city": True},
    "Cape Town": {"temp": 20, "humidity": 65, "wind": 18, "dew_city": False},
    "Johannesburg": {"temp": 18, "humidity": 50, "wind": 12, "dew_city": False},
}

DEW_CITIES = {city for city, data in CLIMATE_DEFAULTS.items() if data.get("dew_city")}


class WeatherService:
    """Fetch weather data for cricket ground cities."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("WEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self._cache = {}
        self._cache_ttl = 1800  # 30 minutes

    def get_weather(self, city):
        """
        Get current weather for a city.

        Returns dict with: temperature, humidity, wind_speed, cloud_cover, rain_likely
        """
        if not city:
            return self._default_weather()

        # Check cache
        cache_key = city.lower().strip()
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try API
        if self.api_key and HAS_REQUESTS:
            try:
                resp = requests.get(
                    f"{self.base_url}/weather",
                    params={"q": city, "appid": self.api_key, "units": "metric"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "wind_speed": data["wind"]["speed"] * 3.6,  # m/s to km/h
                        "cloud_cover": data.get("clouds", {}).get("all", 0),
                        "rain_likely": data.get("rain", {}).get("1h", 0) > 0,
                        "source": "api",
                    }
                    self._set_cached(cache_key, result)
                    return result
            except Exception as e:
                logger.debug(f"Weather API failed for {city}: {e}")

        # Fallback to climate defaults
        return self._climate_fallback(city)

    def compute_weather_features(self, city, is_day_night=False):
        """
        Compute weather-based features for the prediction model.

        Returns dict with: temperature, humidity, wind_speed, rain_likely, dew_factor
        """
        weather = self.get_weather(city)

        dew_factor = 0
        if is_day_night:
            humidity = weather.get("humidity", 0)
            city_clean = city.strip() if city else ""
            if humidity > 70 and any(dc.lower() in city_clean.lower()
                                    for dc in DEW_CITIES):
                dew_factor = 1

        return {
            "temperature": weather.get("temperature", 25),
            "humidity": weather.get("humidity", 60),
            "wind_speed": weather.get("wind_speed", 10),
            "rain_likely": int(weather.get("rain_likely", False)),
            "cloud_cover": weather.get("cloud_cover", 0),
            "dew_factor": dew_factor,
        }

    def _climate_fallback(self, city):
        """Use average climate data as fallback."""
        city_clean = city.strip() if city else ""
        for known_city, data in CLIMATE_DEFAULTS.items():
            if known_city.lower() in city_clean.lower():
                return {
                    "temperature": data["temp"],
                    "humidity": data["humidity"],
                    "wind_speed": data["wind"],
                    "cloud_cover": 50,
                    "rain_likely": False,
                    "source": "climate_avg",
                }
        return self._default_weather()

    def _default_weather(self):
        return {
            "temperature": 25, "humidity": 60, "wind_speed": 10,
            "cloud_cover": 50, "rain_likely": False, "source": "default",
        }

    def _get_cached(self, key):
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
        return None

    def _set_cached(self, key, data):
        self._cache[key] = (time.time(), data)
