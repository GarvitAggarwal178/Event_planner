from typing import Optional, List
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph. checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from datetime import date, time, timedelta, datetime
from model import Attraction, Hotel, UserPreferences, DayPlan, Itinerary, Activity
from apis import RealAPIProvider
import uuid
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np


def print_header(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def print_progress(msg: str):
    print(f" {msg}...")


def parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response"""
    try: 
        start, end = text.find('{'), text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return {}


def get_comment(phase: str) -> str:
    """Get user comment/feedback on current phase"""
    print(f"\n{'─'*70}")
    print(f"💬 Any comments or changes for {phase}?")
    print("   (Press Enter to proceed, or type your feedback)")
    return input("➜  ").strip()


def confirm(msg: str = "Proceed") -> bool:
    """Simple yes/no confirmation"""
    response = input(f"{msg}? (y/n): ").strip().lower()
    return response in ['y', 'yes', '']


class TravelPlannerState(MessagesState):
    user_preferences: Optional[UserPreferences] = None
    available_attractions: List[Attraction] = []
    available_hotels: List[Hotel] = []
    weather_forecast: dict = {}
    selected_attractions: List[Attraction] = []
    selected_hotel: Optional[Hotel] = None
    daily_routes: dict = {}
    daily_schedules: List[DayPlan] = []
    total_cost: float = 0.0
    cost_breakdown: dict = {}
    final_itinerary: Optional[Itinerary] = None


class TravelPlanningAgent:
    """AI-powered travel planner with LLM + ML"""
    
    def __init__(self, provider: RealAPIProvider):
        self.provider = provider
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os. getenv("GROQ_API_KEY"),
            temperature=0
        )
        self.structured_llm = self. llm.with_structured_output(UserPreferences)
        
        # Load embedding model
        print_progress("Loading AI models")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   ✅ Models loaded")
        except:
            self.embedding_model = None
            print("   ⚠️  Running without embeddings")
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(TravelPlannerState)
        
        workflow.add_node("extract_preferences", self. extract_preferences_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("select", self.select_node)
        workflow.add_node("optimize_routes", self.optimize_routes_node)
        workflow.add_node("build_schedules", self.build_schedules_node)
        workflow.add_node("check_budget", self.check_budget_node)
        workflow.add_node("generate_itinerary", self.generate_itinerary_node)
        
        workflow.add_edge(START, "extract_preferences")
        workflow.add_edge("extract_preferences", "research")
        workflow.add_edge("research", "select")
        workflow.add_edge("select", "optimize_routes")
        workflow.add_edge("optimize_routes", "build_schedules")
        workflow.add_edge("build_schedules", "check_budget")
        workflow.add_conditional_edges(
            "check_budget",
            lambda s: "generate" if s["total_cost"] <= s["user_preferences"].budget else "revise",
            {"generate": "generate_itinerary", "revise": END}
        )
        workflow.add_edge("generate_itinerary", END)
        
        return workflow. compile(checkpointer=MemorySaver())
    
    def extract_preferences_node(self, state: TravelPlannerState):
        """Parse user request into structured preferences"""
        print_header("PHASE 1: Understanding Request")
        
        request = state["messages"][-1].content if state["messages"] else ""
        print(f"📝 \"{request}\"\n")
        
        prefs = self.structured_llm.invoke([
            SystemMessage(content="""Extract:  destination, num_days, budget, interests, pace. 
Defaults:  num_days=5, budget=2000, interests=["culture"], pace="moderate", start_date=null"""),
            HumanMessage(content=request)
        ])
        
        if not prefs.start_date:
            prefs. start_date = date. today() + timedelta(days=7)
        
        # Display
        print("Understood:\n")
        print(f"   {prefs.destination} | {prefs.num_days} days | $ {prefs.budget:,.0f}")
        print(f"   {', '.join(prefs.interests)} | {prefs.pace} | {prefs.start_date}")
        
        # Get feedback
        comment = get_comment("preferences")
        if comment:
            prefs = self._update_preferences(prefs, comment)
            print(f"\n✅ Updated:  {prefs.destination}, {prefs.num_days} days, {', '.join(prefs.interests)}")
        
        if not confirm(): 
            raise Exception("User cancelled")
        
        return {"user_preferences": prefs}
    
    def research_node(self, state: TravelPlannerState):
        """Fetch attractions, hotels, weather"""
        print_header("PHASE 2: Research")
        
        prefs = state["user_preferences"]
        
        # Fetch data
        print_progress(f"Searching {prefs.destination}")
        attractions = self.provider. get_attractions(prefs.destination, prefs.interests, prefs.num_days * 6)
        
        budget_per_night = prefs.budget / prefs.num_days / 3
        hotels = self.provider.get_hotels(prefs.destination, budget_per_night, 5)
        
        weather = {}
        for day in range(prefs.num_days):
            weather[day + 1] = self.provider. get_weather_forecast(
                prefs.destination, 
                prefs.start_date + timedelta(days=day)
            )
        
        # Display
        print(f"   {len(attractions)} attractions | {len(hotels)} hotels | {len(weather)} days forecast\n")
        
        print("Weather:")
        for d, w in list(weather.items())[:3]:
            print(f"   Day {d}: {w.condition}, {w.temp_low}-{w.temp_high}°C")

        print(f"\nSample attractions:")
        for i, a in enumerate(attractions[:3], 1):
            print(f"   {i}. {a.name} ({a.category}) - {a.rating}, ${a.cost}")
        
        if not confirm():
            raise Exception("User cancelled")
        
        return {
            "available_attractions": attractions,
            "available_hotels": hotels,
            "weather_forecast": weather
        }
    
    def select_node(self, state: TravelPlannerState):
        """AI selects best attractions and hotel"""
        print_header("PHASE 3: AI Selection")
        
        prefs = state["user_preferences"]
        attractions = state["available_attractions"]
        hotels = state["available_hotels"]
        weather = state["weather_forecast"]
        
        pace_map = {"relaxed": 2, "moderate": 3, "packed": 4}
        num_needed = prefs.num_days * pace_map[prefs.pace]
        
        # Semantic filtering
        if self.embedding_model:
            print_progress("AI filtering by semantic relevance")
            attractions = self._semantic_filter(attractions, prefs. interests)
        
        # LLM selection
        print_progress(f"AI selecting {num_needed} attractions")
        selected, reasoning = self._llm_select_attractions(
            attractions, prefs, weather, num_needed
        )
        
        # Display
        print(f"\nSelected {len(selected)} attractions:\n")
        for i, a in enumerate(selected[: 5], 1):
            why = reasoning. get(a.id, {}).get("why", "")
            print(f"   {i}. {a.name} - {why}")
        if len(selected) > 5:
            print(f"   ... and {len(selected) - 5} more")
        
        # Hotel
        print_progress("\nSelecting hotel")
        hotel, hotel_why = self._llm_select_hotel(hotels, prefs)
        if hotel:
            print(f"   {hotel.name} - ${hotel.price_per_night:.0f}/night ({hotel.rating})")
            print(f"   {hotel_why}")
        
        # Get feedback
        comment = get_comment("selections")
        if comment:
            selected, hotel = self._update_selections(comment, selected, hotels, attractions, prefs, num_needed)
        
        if not confirm():
            raise Exception("User cancelled")
        
        return {
            "selected_attractions": selected,
            "selected_hotel": hotel,
            "user_preferences": prefs
        }
    
    def optimize_routes_node(self, state: TravelPlannerState):
        """AI clusters and optimizes routes"""
        print_header("PHASE 4: Route Optimization")
        
        prefs = state["user_preferences"]
        attractions = state["selected_attractions"]
        hotel = state["selected_hotel"]
        weather = state["weather_forecast"]
        
        # K-Means clustering
        print_progress("AI clustering by location")
        daily_routes = self._cluster_by_location(attractions, prefs. num_days)
        
        # LLM optimization
        print_progress("AI optimizing for weather & timing")
        daily_routes, reasoning = self._llm_optimize_routes(
            attractions, daily_routes, prefs, weather
        )
        
        # Display
        print(f"\nDaily Routes:\n")
        for day, ids in daily_routes.items():
            day_attrs = [a for a in attractions if a.id in ids]
            print(f"   Day {day}: {len(day_attrs)} stops")
            for a in day_attrs[:2]:
                print(f"      {a.name}")

        print(f"\nStrategy: {reasoning[: 100]}...")
        
        if not confirm():
            raise Exception("User cancelled")
        
        return {"daily_routes": daily_routes}
    
    def build_schedules_node(self, state:  TravelPlannerState):
        """AI creates hour-by-hour schedules"""
        print_header("PHASE 5: Schedule Building")
        
        prefs = state["user_preferences"]
        attractions = state["selected_attractions"]
        hotel = state["selected_hotel"]
        routes = state["daily_routes"]
        weather = state["weather_forecast"]
        
        schedules = []
        for day in range(1, prefs.num_days + 1):
            day_date = prefs.start_date + timedelta(days=day - 1)
            day_attrs = [a for a in attractions if a.id in routes. get(day, [])]
            
            schedule = self._llm_build_schedule(
                day, day_date, day_attrs, weather[day], hotel, prefs. pace
            )
            schedules.append(schedule)
        
        # Display
        print(f"\n{len(schedules)} days scheduled:\n")
        for s in schedules[: 2]:
            print(f"   Day {s.day_number}: ${s.total_cost:.0f}")
            for act in s.activities[:2]: 
                print(f"      {act.start_time.strftime('%H:%M')} {act.name}")
        
        if not confirm():
            raise Exception("User cancelled")
        
        return {"daily_schedules": schedules}
    
    def check_budget_node(self, state: TravelPlannerState):
        """Calculate costs"""
        prefs = state["user_preferences"]
        schedules = state["daily_schedules"]
        hotel = state["selected_hotel"]
        
        activities_cost = sum(d.total_cost for d in schedules)
        accommodation_cost = hotel.price_per_night * prefs.num_days if hotel else 0
        transport_cost = 50.0 * prefs.num_days
        total = activities_cost + accommodation_cost + transport_cost
        
        breakdown = {
            "accommodation": accommodation_cost,
            "food": sum(sum(a.cost for a in d.activities if a.type == "meal") for d in schedules),
            "attractions": sum(sum(a.cost for a in d.activities if a.type == "attraction") for d in schedules),
            "transport": transport_cost
        }
        
        print(f"\nTotal:  ${total:.0f} / ${prefs.budget:.0f}")
        for k, v in breakdown.items():
            print(f"   {k.title()}: ${v:.0f}")
        
        return {"total_cost": total, "cost_breakdown": breakdown}
    
    def generate_itinerary_node(self, state: TravelPlannerState):
        """Create final itinerary"""
        prefs = state["user_preferences"]
        
        return {"final_itinerary":  Itinerary(
            destination=prefs.destination,
            start_date=prefs.start_date,
            end_date=prefs.start_date + timedelta(days=prefs.num_days - 1),
            num_days=prefs.num_days,
            days=state["daily_schedules"],
            hotel=state["selected_hotel"],
            total_cost=state["total_cost"],
            cost_breakdown=state["cost_breakdown"]
        )}
    
    
    def _update_preferences(self, prefs:  UserPreferences, comment: str) -> UserPreferences:
        """Use LLM to update preferences based on comment"""
        response = self.llm.invoke([
            SystemMessage(f"Update preferences based on user comment.  Current: {prefs}"),
            HumanMessage(comment)
        ])
        
        # Parse updates from LLM response
        text = response.content. lower()
        if "entertainment" in text or "nightlife" in text:
            if "entertainment" not in prefs.interests:
                prefs.interests.append("entertainment")
        if "food" in text or "cuisine" in text:
            if "food" not in prefs.interests:
                prefs.interests. append("food")
        
        # Check for budget/days changes
        import re
        budget_match = re.search(r'\$? (\d+)', text)
        if budget_match and "budget" in text:
            prefs.budget = float(budget_match.group(1))
        
        days_match = re.search(r'(\d+)\s*days?', text)
        if days_match: 
            prefs.num_days = int(days_match.group(1))
        
        return prefs
    
    def _semantic_filter(self, attractions:  List[Attraction], interests: List[str], top_k: int = 40) -> List[Attraction]: 
        """Filter using embeddings"""
        if not self.embedding_model:
            return attractions
        
        user_emb = self.embedding_model. encode(" ".join(interests))
        scored = []
        
        for a in attractions:
            attr_emb = self.embedding_model.encode(f"{a.category} {a.name} {a.description}")
            similarity = np.dot(user_emb, attr_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(attr_emb))
            score = (similarity * 5) + (a.rating * 0.5)
            scored.append((a, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, s in scored[:top_k]]
    
    def _llm_select_attractions(self, attractions, prefs, weather, num_needed) -> tuple:
        """LLM selects attractions"""
        attrs_text = "\n".join([
            f"{i+1}.  {a.name} ({a. category}, {a.rating}★, ${a.cost}, {a.duration_hours}h)"
            for i, a in enumerate(attractions[: 30])
        ])
        
        response = self.llm.invoke([
            SystemMessage("You are a travel expert. Select attractions and respond with valid JSON."),
            HumanMessage(f"""Select {num_needed} best attractions for someone interested in {', '.join(prefs.interests)}. 

            Attractions:
            {attrs_text}

            JSON format:
            {{"selections": [{{"name": ".. .", "why": "..."}}]}}""")
                    ])
        
        result = parse_json(response.content)
        selected, reasoning = [], {}
        
        for item in result.get("selections", [])[:num_needed]:
            name = item.get("name", "")
            matching = [a for a in attractions if name. lower() in a.name.lower()]
            if matching:
                a = matching[0]
                selected.append(a)
                reasoning[a.id] = {"why": item.get("why", "")}
        
        # Fill if needed
        if len(selected) < num_needed:
            remaining = [a for a in attractions if a not in selected]
            selected.extend(remaining[:num_needed - len(selected)])
        
        return selected, reasoning
    
    def _llm_select_hotel(self, hotels, prefs) -> tuple:
        """LLM selects hotel"""
        if not hotels:
            return None, "No hotels available"
        
        hotels_text = "\n".join([f"{h.name} - ${h.price_per_night:.0f}/night, {h.rating}" for h in hotels])
        
        response = self.llm.invoke([
            SystemMessage("Select best hotel.  Respond with JSON."),
            HumanMessage(f"""Budget: ${prefs.budget/prefs.num_days/3:.0f}/night

            Hotels:
            {hotels_text}

            JSON:  {{"hotel": "...", "why": "..."}}""")
        ])
        
        result = parse_json(response.content)
        hotel_name = result.get("hotel", "")
        matching = [h for h in hotels if hotel_name.lower() in h.name.lower()]
        
        return (matching[0], result.get("why", "")) if matching else (hotels[0], "Best value")
    
    def _update_selections(self, comment, selected, hotels, all_attrs, prefs, num_needed):
        """Update selections based on comment"""
        comment_lower = comment.lower()
        
        # Handle exclusions
        if "exclude" in comment_lower or "remove" in comment_lower:
            for a in selected[: ]: 
                if a.name. lower() in comment_lower:
                    selected.remove(a)
        
        # Handle additions
        if "add" in comment_lower:
            for a in all_attrs: 
                if a.name.lower() in comment_lower and a not in selected:
                    selected. append(a)
        
        # Handle hotel change
        hotel = [h for h in hotels if h. name.lower() in comment_lower]
        
        return selected[: num_needed], hotel[0] if hotel else None
    
    def _cluster_by_location(self, attractions, num_days) -> dict:
        """K-Means clustering"""
        if len(attractions) <= num_days:
            return {i+1: [a.id] for i, a in enumerate(attractions)}
        
        coords = np.array([[a.location.lat, a.location.lon] for a in attractions])
        kmeans = KMeans(n_clusters=num_days, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        
        routes = {day: [] for day in range(1, num_days + 1)}
        for i, cluster_id in enumerate(clusters):
            routes[cluster_id + 1]. append(attractions[i].id)
        
        return routes
    
    def _llm_optimize_routes(self, attractions, routes, prefs, weather) -> tuple:
        """LLM optimizes routes"""
        summary = "\n".join([
            f"Day {d}: {len(ids)} stops, {weather[d].condition}"
            for d, ids in routes.items()
        ])
        
        response = self.llm.invoke([
            SystemMessage("Optimize routes.  Respond with JSON."),
            HumanMessage(f"""Current plan: 
            {summary}

            Optimize for weather (indoor on rainy days).
            JSON: {{"routes": {{"1": ["name1", "name2"]}}, "why": "..."}}""")
        ])
        
        result = parse_json(response.content)
        reasoning = result.get("why", "Grouped by location")
        
        return routes, reasoning  # Keep original for now
    
    def _llm_build_schedule(self, day_num, day_date, attractions, weather, hotel, pace) -> DayPlan:
        """LLM creates schedule"""
        attrs_text = "\n".join([f"- {a.name}:  {a.duration_hours}h, ${a.cost}" for a in attractions])
        
        response = self.llm.invoke([
            SystemMessage("Create schedule. Respond with JSON."),
            HumanMessage(f"""Day {day_num}, {weather. condition}, {weather.temp_high}°C
            Attractions:
            {attrs_text}

            Create 9am-9pm schedule with lunch (12: 30, $25) and dinner (19:00, $40).
            JSON: {{"schedule":  [{{"type": "attraction/meal", "name": "...", "start":  "09:00", "duration": 2. 5}}]}}""")
        ])
        
        result = parse_json(response.content)
        activities, cost = [], 0.0
        
        for item in result.get("schedule", []):
            if item["type"] == "attraction": 
                a = next((x for x in attractions if item["name"].lower() in x.name.lower()), None)
                if a:
                    start = datetime.strptime(item["start"], "%H:%M").time()
                    end = (datetime.combine(day_date, start) + timedelta(hours=item["duration"])).time()
                    activities.append(Activity(
                        type="attraction",
                        name=a.name,
                        location=a.location,
                        start_time=start,
                        end_time=end,
                        duration_hours=a.duration_hours,
                        cost=a.cost
                    ))
                    cost += a.cost
            elif item["type"] == "meal":
                start = datetime.strptime(item["start"], "%H:%M").time()
                end = (datetime.combine(day_date, start) + timedelta(hours=item["duration"])).time()
                meal_cost = 25.0 if "lunch" in item["name"].lower() else 40.0
                loc = hotel.location if hotel else attractions[0].location if attractions else None
                activities.append(Activity(
                    type="meal",
                    name=item["name"],
                    location=loc,
                    start_time=start,
                    end_time=end,
                    duration_hours=item["duration"],
                    cost=meal_cost
                ))
                cost += meal_cost
        
        return DayPlan(
            date=day_date,
            day_number=day_num,
            activities=activities,
            total_cost=round(cost, 2),
            total_distance_km=0.0
        )
    
    def plan_trip(self, user_request: str) -> Optional[Itinerary]:
        """Main entry point"""
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_request)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        return result.get("final_itinerary")