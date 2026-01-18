from dotenv import load_dotenv
from apis import RealAPIProvider
from agent import TravelPlanningAgent

# Load environment variables
load_dotenv()


def print_itinerary(itinerary):
    """Pretty print the itinerary"""
    if not itinerary:
        print("\n Failed to generate itinerary (likely over budget)")
        return
    
    print(" TRAVEL ITINERARY GENERATED")

    
    print(f"\n Destination: {itinerary. destination}")
    print(f" Dates: {itinerary.start_date} → {itinerary.end_date} ({itinerary.num_days} days)")
    
    if itinerary.hotel:
        print(f" Hotel: {itinerary.hotel.name}")
        print(f"    ${itinerary.hotel.price_per_night}/night")
        print(f"    {itinerary.hotel.rating} stars")
    
    print(f"\n Budget Breakdown:")
    for category, cost in itinerary.cost_breakdown.items():
        print(f"   {category. title():15} ${cost: 8.2f}")
    print(f"   {'-'*30}")
    print(f"   {'Total':<15} ${itinerary.total_cost:8.2f}")
    
    print(f"\n Daily Schedule:")
    print("="*70)
    
    for day in itinerary.days:
        print(f"\n Day {day.day_number} - {day.date.strftime('%A, %B %d, %Y')}")
        print(f"   Daily cost: ${day.total_cost:.2f}")
        print(f"   {'-'*66}")
        
        for activity in day.activities:
            time_range = f"{activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}"
            print(f"{time_range:13} {activity.name:30} ${activity.cost:6.2f}")
    


def main():
    print(" TRAVEL PLANNING AGENT - REAL API TEST")
    
    # Initialize provider and agent
    print("\n  Initializing APIs...")
    provider = RealAPIProvider()
    agent = TravelPlanningAgent(provider)
    print(" Agent ready!\n")
    
    # Test request
    request = """
    I want to visit Paris for 5 days starting next week.  
    I love art and history. 
    My budget is $3000 and I prefer a moderate pace.
    """
    
    print(" User Request:")
    print(request)
    print("\n Planning your trip.. .\n")
    
    # Plan the trip
    try: 
        itinerary = agent.plan_trip(request)
        print_itinerary(itinerary)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()