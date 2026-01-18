from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date, time

class Location(BaseModel):
    name: str
    lat: float
    lon: float

class Attraction(BaseModel):
    id: str
    name: str
    location: Location
    category: str  # museum, landmark, nature, etc.
    description: str
    rating: float = Field(ge=0, le=5)
    duration_hours: float 
    cost:  float 
    weather_dependent: bool  # Indoor vs outdoor

class Hotel(BaseModel):
    id: str
    name: str
    location: Location
    rating: float = Field(ge=0, le=5)
    price_per_night: float

class Weather(BaseModel):
    date: date
    condition:  Literal["sunny", "cloudy", "rainy", "snowy"]
    temp_high: int
    temp_low: int
    precipitation_chance: int
    suitable_for_outdoor: bool

class Activity(BaseModel):
    type: Literal["attraction", "meal", "transport", "hotel"]
    name: str
    location: Location
    start_time: time
    end_time: time
    duration_hours: float
    cost:  float
    notes: Optional[str] = None

class DayPlan(BaseModel):
    date: date
    day_number: int
    activities: List[Activity]
    total_cost: float
    total_distance_km: float

class Itinerary(BaseModel):
    destination: str
    start_date: date
    end_date: date
    num_days: int
    days:  List[DayPlan]
    hotel:  Hotel
    total_cost: float
    cost_breakdown: dict 

class UserPreferences(BaseModel):
    destination: str
    num_days: int
    budget: float
    interests: List[str]
    pace: Literal["relaxed", "moderate", "packed"] = "moderate"
    start_date: Optional[date] = None