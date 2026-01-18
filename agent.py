"""
Multi-agent travel planner using LangGraph
"""

from typing import Optional, List
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from datetime import date, time, timedelta, datetime
from model import Attraction, Hotel, Weather, UserPreferences, DayPlan, Itinerary, Activity
from apis import RealAPIProvider
import uuid


class TravelPlannerState(MessagesState):
    """State flowing through the LangGraph workflow"""
    
    user_preferences: Optional[UserPreferences] = None
    available_attractions: List[Attraction] = []
    available_hotels: List[Hotel] = []
    weather_forecast: dict = {}
    selected_attractions: List[Attraction] = []
    selected_hotel: Optional[Hotel] = None
    distance_matrix: dict = {}
    daily_routes: dict = {}
    daily_schedules: List[DayPlan] = []
    total_cost: float = 0.0
    cost_breakdown: dict = {}
    final_itinerary: Optional[Itinerary] = None


class TravelPlanningAgent: 
    """
    Multi-agent travel planner using LangGraph. 
    
    Workflow:
    1. extract_preferences - Parse user request
    2. research - Fetch attractions, hotels, weather
    3. select - Choose best options
    4. optimize_routes - Group by location/day
    5. build_schedules - Create hour-by-hour plans
    6. check_budget - Validate costs
    7. generate_itinerary - Format final output
    """
    
    def __init__(self, provider: RealAPIProvider):
        self.provider = provider
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os. getenv("GROQ_API_KEY"),
            temperature=0
        )
        self.structured_llm = self.llm.with_structured_output(UserPreferences)
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Construct the LangGraph workflow"""
        workflow = StateGraph(TravelPlannerState)
        
        workflow.add_node("extract_preferences", self. extract_preferences_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("select", self.select_node)
        workflow.add_node("optimize_routes", self. optimize_routes_node)
        workflow.add_node("build_schedules", self.build_schedules_node)
        workflow.add_node("check_budget", self.check_budget_node)
        workflow.add_node("generate_itinerary", self. generate_itinerary_node)
        
        workflow.add_edge(START, "extract_preferences")
        workflow.add_edge("extract_preferences", "research")
        workflow.add_edge("research", "select")
        workflow.add_edge("select", "optimize_routes")
        workflow.add_edge("optimize_routes", "build_schedules")
        workflow.add_edge("build_schedules", "check_budget")
        workflow.add_conditional_edges(
            "check_budget",
            self._should_continue_after_budget,
            {"generate":  "generate_itinerary", "revise": END}
        )
        workflow.add_edge("generate_itinerary", END)
        
        return workflow. compile(checkpointer=MemorySaver())
    
    # =========================================================================
    # NODE 1: Extract Preferences
    # =========================================================================
    
    def extract_preferences_node(self, state: TravelPlannerState):
        """Extract structured preferences from natural language"""
        messages = state["messages"]
        last_message = messages[-1]. content if messages else ""
        
        # Better prompt with examples
        preferences = self.structured_llm. invoke([
            SystemMessage(content="""Extract travel preferences from the user's message. 

    Rules:
    - destination: City name (e.g., "Paris", "Tokyo")
    - num_days: Integer number of days (e.g., 5)
    - budget: Total budget in USD as a number (e.g., 3000)
    - interests: List of interests (e.g., ["art", "history", "food"])
    - pace: One of "relaxed", "moderate", or "packed"
    - start_date:  LEAVE AS NULL - we'll set it automatically

    Defaults if not specified:
    - num_days: 5
    - budget: 2000
    - interests: ["culture"]
    - pace: "moderate"
    - start_date: null

    IMPORTANT: Do NOT try to parse relative dates like "next week".  Set start_date to null.
    """),
            HumanMessage(content=last_message)
        ])
        
        # Always set start_date to a week from now if not provided
        if not preferences.start_date:
            preferences.start_date = date.today() + timedelta(days=7)
        
        return {"user_preferences": preferences}
    
    # =========================================================================
    # NODE 2: Research
    # =========================================================================
    
    def research_node(self, state:  TravelPlannerState):
        """Fetch attractions, hotels, and weather"""
        prefs = state["user_preferences"]
        
        attractions = self.provider.get_attractions(
            destination=prefs.destination,
            interests=prefs.interests,
            max_results=prefs. num_days * 6
        )
        
        budget_per_night = prefs.budget / prefs.num_days / 3
        hotels = self.provider.get_hotels(
            destination=prefs.destination,
            budget_per_night=budget_per_night,
            max_results=5
        )
        
        weather = {}
        for day in range(prefs.num_days):
            target_date = prefs.start_date + timedelta(days=day)
            weather[day + 1] = self.provider. get_weather_forecast(
                prefs.destination,
                target_date
            )
        
        return {
            "available_attractions": attractions,
            "available_hotels": hotels,
            "weather_forecast": weather
        }
    
    # =========================================================================
    # NODE 3: Select
    # =========================================================================
    
    def select_node(self, state: TravelPlannerState):
        """Select best attractions and hotel"""
        prefs = state["user_preferences"]
        attractions = state["available_attractions"]
        hotels = state["available_hotels"]
        weather = state["weather_forecast"]
        
        pace_map = {"relaxed": 2, "moderate":  3, "packed": 4}
        num_needed = prefs.num_days * pace_map[prefs. pace]
        
        scored = [(a, self._score_attraction(a, prefs. interests, weather)) for a in attractions]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_attractions = [a for a, s in scored[:num_needed]]
        
        selected_hotel = hotels[0] if hotels else None
        
        return {
            "selected_attractions": selected_attractions,
            "selected_hotel": selected_hotel
        }
    
    def _score_attraction(self, attraction: Attraction, interests:  List[str], weather: dict) -> float:
        """Score attraction based on rating, interests, and weather"""
        score = attraction.rating
        
        category_map = {
            "museum": ["art", "culture", "history"],
            "landmark": ["architecture", "culture"],
            "historical": ["history", "culture"],
            "religious": ["culture", "architecture", "history"],
            "nature":  ["nature"],
            "park": ["nature"],
        }
        
        attr_interests = category_map.get(attraction.category, [])
        matches = len(set(attr_interests) & set(interests))
        score += matches * 1.5
        
        if attraction.cost == 0:
            score += 0.5
        
        bad_weather_days = sum(1 for w in weather. values() if not w.suitable_for_outdoor)
        if attraction.weather_dependent and bad_weather_days > len(weather) / 2:
            score -= 1.0
        
        return score
    
    # =========================================================================
    # NODE 4: Optimize Routes
    # =========================================================================
    
    def optimize_routes_node(self, state: TravelPlannerState):
        """Group attractions by day based on proximity"""
        prefs = state["user_preferences"]
        attractions = state["selected_attractions"]
        hotel = state["selected_hotel"]
        
        locations = [a.location for a in attractions]
        if hotel:
            locations.append(hotel.location)
        
        distance_matrix = self.provider.get_distance_matrix(locations)
        
        daily_routes = {}
        attractions_per_day = len(attractions) // prefs.num_days
        
        for day in range(1, prefs.num_days + 1):
            start_idx = (day - 1) * attractions_per_day
            end_idx = start_idx + attractions_per_day if day < prefs.num_days else len(attractions)
            daily_attractions = attractions[start_idx:end_idx]
            daily_routes[day] = [a.id for a in daily_attractions]
        
        return {
            "distance_matrix": distance_matrix,
            "daily_routes": daily_routes
        }
    
    # =========================================================================
    # NODE 5: Build Schedules
    # =========================================================================
    
    def build_schedules_node(self, state: TravelPlannerState):
        """Create detailed hour-by-hour schedules"""
        prefs = state["user_preferences"]
        attractions = state["selected_attractions"]
        hotel = state["selected_hotel"]
        routes = state["daily_routes"]
        
        schedules = []
        
        for day_num in range(1, prefs.num_days + 1):
            day_date = prefs.start_date + timedelta(days=day_num - 1)
            attraction_ids = routes. get(day_num, [])
            day_attractions = [a for a in attractions if a.id in attraction_ids]
            
            activities = []
            current_time = time(9, 0)
            current_cost = 0.0
            
            # Morning attractions
            for attr in day_attractions[: 2]: 
                end_time = self._add_hours(current_time, attr.duration_hours)
                activities.append(Activity(
                    type="attraction",
                    name=attr.name,
                    location=attr.location,
                    start_time=current_time,
                    end_time=end_time,
                    duration_hours=attr.duration_hours,
                    cost=attr.cost
                ))
                current_cost += attr.cost
                current_time = end_time
            
            # Lunch
            lunch_start = max(current_time, time(12, 0))
            lunch_end = self._add_hours(lunch_start, 1.5)
            activities.append(Activity(
                type="meal",
                name="Lunch",
                location=hotel. location if hotel else day_attractions[0].location,
                start_time=lunch_start,
                end_time=lunch_end,
                duration_hours=1.5,
                cost=25.0
            ))
            current_cost += 25.0
            current_time = lunch_end
            
            # Afternoon attractions
            for attr in day_attractions[2:]:
                end_time = self._add_hours(current_time, attr.duration_hours)
                activities.append(Activity(
                    type="attraction",
                    name=attr.name,
                    location=attr.location,
                    start_time=current_time,
                    end_time=end_time,
                    duration_hours=attr. duration_hours,
                    cost=attr.cost
                ))
                current_cost += attr.cost
                current_time = end_time
            
            # Dinner
            activities.append(Activity(
                type="meal",
                name="Dinner",
                location=hotel. location if hotel else day_attractions[0].location,
                start_time=time(19, 0),
                end_time=time(20, 30),
                duration_hours=1.5,
                cost=40.0
            ))
            current_cost += 40.0
            
            schedules.append(DayPlan(
                date=day_date,
                day_number=day_num,
                activities=activities,
                total_cost=round(current_cost, 2),
                total_distance_km=0.0
            ))
        
        return {"daily_schedules": schedules}
    
    def _add_hours(self, t: time, hours: float) -> time:
        """Add hours to a time object"""
        dt = datetime.combine(date.today(), t)
        return (dt + timedelta(hours=hours)).time()
    
    # =========================================================================
    # NODE 6: Check Budget
    # =========================================================================
    
    def check_budget_node(self, state: TravelPlannerState):
        """Calculate total cost and check against budget"""
        prefs = state["user_preferences"]
        schedules = state["daily_schedules"]
        hotel = state["selected_hotel"]
        
        activities_cost = sum(day.total_cost for day in schedules)
        accommodation_cost = hotel.price_per_night * prefs.num_days if hotel else 0
        transport_cost = 50.0 * prefs.num_days
        
        total_cost = activities_cost + accommodation_cost + transport_cost
        
        breakdown = {
            "accommodation": accommodation_cost,
            "food": sum(sum(a.cost for a in day.activities if a.type == "meal") for day in schedules),
            "attractions": sum(sum(a.cost for a in day.activities if a.type == "attraction") for day in schedules),
            "transport": transport_cost
        }
        
        return {
            "total_cost": total_cost,
            "cost_breakdown": breakdown
        }
    
    def _should_continue_after_budget(self, state: TravelPlannerState) -> str:
        """Decide if we should generate itinerary or revise"""
        return "revise" if state["total_cost"] > state["user_preferences"].budget else "generate"
    
    # =========================================================================
    # NODE 7: Generate Itinerary
    # =========================================================================
    
    def generate_itinerary_node(self, state: TravelPlannerState):
        """Create final structured itinerary"""
        prefs = state["user_preferences"]
        
        itinerary = Itinerary(
            destination=prefs.destination,
            start_date=prefs.start_date,
            end_date=prefs.start_date + timedelta(days=prefs. num_days - 1),
            num_days=prefs.num_days,
            days=state["daily_schedules"],
            hotel=state["selected_hotel"],
            total_cost=state["total_cost"],
            cost_breakdown=state["cost_breakdown"]
        )
        
        return {"final_itinerary": itinerary}
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def plan_trip(self, user_request: str) -> Optional[Itinerary]:
        """
        Plan a trip from user's natural language request.
        
        Args:
            user_request: e.g., "Visit Paris for 5 days, love art, budget $2000"
        
        Returns:
            Complete Itinerary object or None if over budget
        """
        thread_id = str(uuid.uuid4())
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_request)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return result.get("final_itinerary")