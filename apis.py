"""
Production-ready API provider for travel data
"""

from typing import List, Optional
from datetime import date, datetime, timedelta
import requests
import os
from time import sleep
from model import Attraction, Hotel, Location, Weather


class RealAPIProvider:
    """
    Production API provider connecting to: 
    - Geoapify:  Attractions
    - Open-Meteo: Weather (no key needed)
    - Nominatim: Geocoding (no key needed)
    - OSRM: Routing (no key needed)
    - Booking.com: Hotels (via RapidAPI)
    """
    
    def __init__(self):
        """Load API keys and initialize endpoints"""
        # API Keys
        self.geoapify_key = os.getenv("GEOAPIFY_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        
        # API Endpoints
        self.geoapify_base = "https://api.geoapify.com/v2"
        self.nominatim_base = "https://nominatim.openstreetmap.org"
        self.osrm_base = "http://router.project-osrm.org"
        
        # Cache
        self._geocode_cache = {}
    
    # =========================================================================
    # GEOCODING
    # =========================================================================
    
    def geocode(self, place_name: str) -> Location:
        """Convert city name to coordinates using Nominatim"""
        if place_name in self._geocode_cache:
            return self._geocode_cache[place_name]
        
        default_location = Location(name=place_name, lat=48.8566, lon=2.3522)
        
        try: 
            response = requests.get(
                f"{self.nominatim_base}/search",
                params={"q": place_name, "format":  "json", "limit": 1},
                headers={"User-Agent": "TravelPlannerAgent/1.0"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data: 
                return default_location
            
            result = data[0]
            location = Location(
                name=result.get("display_name", place_name),
                lat=float(result["lat"]),
                lon=float(result["lon"])
            )
            
            self._geocode_cache[place_name] = location
            sleep(1)  # Respect rate limit
            
            return location
            
        except Exception: 
            return default_location
    
    # =========================================================================
    # ATTRACTIONS
    # =========================================================================
    
    def get_attractions(
        self, 
        destination: str, 
        interests: List[str], 
        max_results: int = 20
    ) -> List[Attraction]: 
        """Fetch attractions using Geoapify Places API"""
        if not self.geoapify_key:
            raise ValueError("GEOAPIFY_API_KEY not configured")
        
        location = self.geocode(destination)
        
        # Map interests to Geoapify categories
        interest_to_categories = {
            "art": "entertainment. museum,entertainment.culture",
            "culture": "entertainment.museum,entertainment. culture,tourism. attraction",
            "history": "entertainment.museum,tourism.attraction",
            "nature": "leisure.park",
            "architecture": "tourism.attraction",
            "food": "catering.restaurant",
        }
        
        categories = set()
        for interest in interests: 
            cats = interest_to_categories.get(interest. lower(), "")
            if cats:
                categories.update(cats.split(","))
        
        if not categories:
            categories = {"tourism.attraction", "entertainment.museum", "leisure.park"}
        
        categories_param = ",".join(categories)
        
        try:
            response = requests.get(
                f"{self.geoapify_base}/places",
                params={
                    "categories": categories_param,
                    "filter":  f"circle:{location.lon},{location.lat},5000",
                    "limit": 50,
                    "apiKey": self.geoapify_key
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            features = data.get("features", [])
            if not features:
                return []
            
            attractions = []
            for feature in features[: max_results]:
                attraction = self._parse_geoapify_place(feature)
                if attraction: 
                    attractions.append(attraction)
            
            return attractions
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch attractions: {e}")
    
    def _parse_geoapify_place(self, feature: dict) -> Optional[Attraction]:
        """Convert Geoapify place to Attraction"""
        try:
            props = feature.get("properties", {})
            place_id = props.get("place_id", "")
            name = props.get("name", "")
            
            if not place_id or not name:
                return None
            
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                return None
            
            lon, lat = coords[0], coords[1]
            
            categories = props.get("categories", [])
            category = self._map_geoapify_category(categories)
            
            description = props.get("description", "")
            if not description:
                address = props.get("formatted", "Tourist attraction")
                description = f"Located at {address}"
            
            duration_map = {
                "museum": 2.5,
                "landmark": 1.5,
                "religious": 1.0,
                "historical": 2.0,
                "nature": 2.0,
                "park":  1.5,
            }
            duration = duration_map.get(category, 1.5)
            
            cost = 15.0 if category == "museum" else (10.0 if category in ["landmark", "historical"] else 0.0)
            
            return Attraction(
                id=place_id,
                name=name,
                location=Location(name=name, lat=lat, lon=lon),
                category=category,
                description=description[: 200],
                rating=4.0,
                duration_hours=duration,
                cost=cost,
                weather_dependent=(category in ["nature", "park", "landmark"])
            )
        except Exception:
            return None
    
    def _map_geoapify_category(self, categories:  List[str]) -> str:
        """Map Geoapify categories to standard categories"""
        cats_str = ",".join(categories).lower()
        
        if "museum" in cats_str: 
            return "museum"
        elif "religion" in cats_str or "place_of_worship" in cats_str:
            return "religious"
        elif "park" in cats_str:
            return "park"
        elif "historic" in cats_str:
            return "historical"
        else:
            return "landmark"
    
    # =========================================================================
    # WEATHER
    # =========================================================================
    
    def get_weather_forecast(self, destination: str, target_date: date) -> Weather:
        """Get weather forecast from Open-Meteo (free, no key needed)"""
        location = self.geocode(destination)
        days_ahead = (target_date - date.today()).days
        
        if days_ahead < 0 or days_ahead > 16:
            return self._default_weather(target_date)
        
        try: 
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location.lat,
                    "longitude": location.lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
                    "timezone":  "auto",
                    "forecast_days": 16
                },
                timeout=10
            )
            response. raise_for_status()
            data = response.json()
            
            dates = data["daily"]["time"]
            target_str = target_date.isoformat()
            
            if target_str not in dates:
                return self._default_weather(target_date)
            
            idx = dates.index(target_str)
            
            temp_high = int(data["daily"]["temperature_2m_max"][idx])
            temp_low = int(data["daily"]["temperature_2m_min"][idx])
            precip_chance = int(data["daily"]["precipitation_probability_max"][idx])
            weather_code = data["daily"]["weathercode"][idx]
            
            condition = self._map_weather_code(weather_code)
            suitable = condition in ["sunny", "cloudy"] and precip_chance < 30
            
            return Weather(
                date=target_date,
                condition=condition,
                temp_high=temp_high,
                temp_low=temp_low,
                precipitation_chance=precip_chance,
                suitable_for_outdoor=suitable
            )
            
        except Exception: 
            return self._default_weather(target_date)
    
    def _map_weather_code(self, code: int) -> str:
        """Map WMO weather codes to simple conditions"""
        if code == 0:
            return "sunny"
        elif code in range(1, 4):
            return "cloudy"
        elif code in range(51, 100):
            return "snowy" if code in range(71, 78) else "rainy"
        else:
            return "cloudy"
    
    def _default_weather(self, target_date: date) -> Weather:
        """Fallback weather based on season"""
        month = target_date.month
        
        if month in [6, 7, 8]:  # Summer
            return Weather(date=target_date, condition="sunny", temp_high=28, temp_low=18, precipitation_chance=20, suitable_for_outdoor=True)
        elif month in [12, 1, 2]:  # Winter
            return Weather(date=target_date, condition="cloudy", temp_high=10, temp_low=3, precipitation_chance=40, suitable_for_outdoor=False)
        else:  # Spring/Fall
            return Weather(date=target_date, condition="cloudy", temp_high=18, temp_low=10, precipitation_chance=30, suitable_for_outdoor=True)
    
    # =========================================================================
    # DISTANCES
    # =========================================================================
    
    def get_distance_matrix(self, locations:  List[Location], mode: str = "walking") -> dict:
        """Calculate distance matrix using OSRM"""
        profile_map = {"walking": "foot", "driving": "car", "transit": "car"}
        profile = profile_map.get(mode, "foot")
        
        coords = ";".join([f"{loc.lon},{loc. lat}" for loc in locations])
        
        try:
            response = requests.get(
                f"{self.osrm_base}/table/v1/{profile}/{coords}",
                params={"annotations": "distance,duration"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "Ok":
                return self._fallback_distance_matrix(locations)
            
            distances = data["distances"]
            durations = data["durations"]
            
            matrix = {}
            for i in range(len(locations)):
                for j in range(len(locations)):
                    matrix[f"{i},{j}"] = {
                        "distance_km": round(distances[i][j] / 1000, 2),
                        "duration_minutes": round(durations[i][j] / 60, 1)
                    }
            
            return matrix
            
        except Exception:
            return self._fallback_distance_matrix(locations)
    
    def _fallback_distance_matrix(self, locations: List[Location]) -> dict:
        """Fallback using Haversine formula"""
        import math
        
        matrix = {}
        speed_kmh = 4.0
        
        for i, origin in enumerate(locations):
            for j, dest in enumerate(locations):
                if i == j:
                    matrix[f"{i},{j}"] = {"distance_km": 0.0, "duration_minutes": 0.0}
                else:
                    lat1, lon1 = math.radians(origin.lat), math.radians(origin.lon)
                    lat2, lon2 = math.radians(dest.lat), math.radians(dest.lon)
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    distance_km = 6371 * 2 * math.asin(math.sqrt(a))
                    duration_min = (distance_km / speed_kmh) * 60
                    
                    matrix[f"{i},{j}"] = {
                        "distance_km":  round(distance_km, 2),
                        "duration_minutes":  round(duration_min, 1)
                    }
        
        return matrix
    
    # =========================================================================
    # HOTELS
    # =========================================================================
    
    def get_hotels(self, destination: str, budget_per_night: float, max_results: int = 10) -> List[Hotel]:
        """Search hotels using Booking.com API via RapidAPI"""
        if not self.rapidapi_key:
            raise ValueError("RAPIDAPI_KEY not configured")
        
        try:
            # Get destination ID
            response = requests.get(
                "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination",
                headers={
                    "X-RapidAPI-Key": self. rapidapi_key,
                    "X-RapidAPI-Host":  "booking-com15.p. rapidapi.com"
                },
                params={"query": destination},
                timeout=15
            )
            response.raise_for_status()
            dest_data = response.json()
            
            if not dest_data.get("data") or len(dest_data["data"]) == 0:
                return []
            
            dest_info = dest_data["data"][0]
            dest_id = dest_info["dest_id"]
            search_type = dest_info. get("search_type", "city")
            
            # Search hotels
            checkin = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
            checkout = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
            
            response = requests.get(
                "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
                headers={
                    "X-RapidAPI-Key": self.rapidapi_key,
                    "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
                },
                params={
                    "dest_id": dest_id,
                    "search_type": search_type,
                    "arrival_date": checkin,
                    "departure_date": checkout,
                    "adults": "1",
                    "children_age": "0",
                    "room_qty": "1",
                    "page_number": "1",
                    "units": "metric",
                    "temperature_unit": "c",
                    "languagecode": "en-us",
                    "currency_code": "USD"
                },
                timeout=20
            )
            response.raise_for_status()
            hotels_data = response.json()
            
            hotels_list = hotels_data.get("data", {}).get("hotels", [])
            if not hotels_list:
                return []
            
            hotels = []
            for hotel_item in hotels_list:
                try:
                    prop = hotel_item.get("property", {})
                    if not prop: 
                        continue
                    
                    name = prop.get("name", "Unknown Hotel")
                    hotel_id = str(prop.get("id", "unknown"))
                    
                    price_breakdown = prop.get("priceBreakdown", {}) or hotel_item.get("priceBreakdown", {})
                    gross_price = price_breakdown.get("grossPrice", {})
                    price_value = gross_price.get("value") or price_breakdown.get("grossAmountPerNight", {}).get("value")
                    
                    if not price_value:
                        continue
                    
                    price = float(price_value)
                    
                    if price > budget_per_night:
                        continue
                    
                    review_score = prop.get("reviewScore", 0)
                    rating = round(review_score / 2, 1) if review_score > 0 else 4.0
                    
                    lat = prop.get("latitude", 48.8566)
                    lon = prop.get("longitude", 2.3522)
                    
                    hotel = Hotel(
                        id=hotel_id,
                        name=name,
                        location=Location(name=name, lat=float(lat), lon=float(lon)),
                        rating=rating,
                        price_per_night=round(price, 2)
                    )
                    
                    hotels.append(hotel)
                    
                    if len(hotels) >= max_results:
                        break
                        
                except Exception:
                    continue
            
            hotels.sort(key=lambda h: h.price_per_night)
            return hotels[:max_results]
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch hotels: {e}")