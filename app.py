"""
Streamlit Web UI for AI Travel Planner
"""

import streamlit as st
from apis import RealAPIProvider
from agent import TravelPlanningAgent
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# PAGE CONFIG
# ============================================================================

st. set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st. markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight:  bold;
        text-align:  center;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip:  text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    . phase-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius:  0.5rem;
        border-left: 4px solid #8fd3f4;
        margin:  1rem 0;
    }
    .attraction-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'agent' not in st. session_state:
    with st.spinner("🔄 Loading AI models..."):
        provider = RealAPIProvider()
        st.session_state. agent = TravelPlanningAgent(provider)
        st.session_state.itinerary = None


# ============================================================================
# SIDEBAR - INPUT
# ============================================================================

st.sidebar.title("✈️ Plan Your Trip")

destination = st.sidebar.text_input("📍 Destination", "Paris")

col1, col2 = st. sidebar.columns(2)
num_days = col1.number_input("📅 Days", 1, 14, 5)
budget = col2.number_input("💰 Budget ($)", 500, 10000, 3000, 100)

interests = st.sidebar.multiselect(
    "🎨 Interests",
    ["art", "history", "culture", "nature", "food", "architecture", "entertainment", "shopping"],
    default=["art", "history"]
)

pace = st.sidebar.select_slider(
    "⚡ Pace",
    options=["relaxed", "moderate", "packed"],
    value="moderate"
)

start_date = st.sidebar.date_input(
    "🗓️ Start Date",
    date. today() + timedelta(days=7)
)

plan_button = st.sidebar.button("🚀 Generate Itinerary", type="primary", use_container_width=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-header">🌍 AI Travel Planner</h1>', unsafe_allow_html=True)

if not plan_button and st.session_state.itinerary is None:
    # Welcome screen
    st.info("👈 Enter your travel preferences in the sidebar and click **Generate Itinerary**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 AI-Powered")
        st.write("Uses advanced LLMs and ML algorithms for intelligent recommendations")
    
    with col2:
        st.markdown("### 🎯 Personalized")
        st.write("Tailored to your interests, budget, and travel style")
    
    with col3:
        st.markdown("### ⚡ Real-Time")
        st.write("Live data from hotels, weather, and attractions APIs")

elif plan_button:
    # Generate itinerary
    request = f"""I want to visit {destination} for {num_days} days starting {start_date}. 
I love {', '.join(interests)}.
My budget is ${budget} and I prefer a {pace} pace."""
    
    with st.spinner("🔄 AI is planning your perfect trip..."):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Understanding your preferences...")
        progress_bar.progress(20)
        
        status_text.text("Researching destination...")
        progress_bar.progress(40)
        
        status_text.text("AI selecting best attractions...")
        progress_bar. progress(60)
        
        status_text.text("Optimizing routes...")
        progress_bar.progress(80)
        
        status_text.text("Building schedules...")
        progress_bar. progress(90)
        
        # Actually generate (this calls the agent)
        try:
            itinerary = st.session_state.agent.plan_trip(request)
            st.session_state.itinerary = itinerary
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.success("🎉 Your itinerary is ready!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

# Display itinerary if available
if st.session_state. itinerary:
    itinerary = st.session_state.itinerary
    
    # ========================================================================
    # OVERVIEW
    # ========================================================================
    
    st.markdown(f'<div class="phase-header"><h2>📋 Trip Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Destination", itinerary. destination)
    
    with col2:
        st.metric("📅 Duration", f"{itinerary.num_days} days")
    
    with col3:
        st.metric("💰 Total Cost", f"${itinerary. total_cost:.0f}")
        st.caption(f"Budget: ${budget}")
    
    with col4:
        savings = budget - itinerary.total_cost
        st.metric("💵 Savings", f"${savings:.0f}", delta=f"{(savings/budget*100):.0f}%")
    
    # ========================================================================
    # BUDGET BREAKDOWN
    # ========================================================================
    
    st.markdown(f'<div class="phase-header"><h2>💰 Budget Breakdown</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st. columns([1, 1])
    
    with col1:
        # Pie chart
        fig = px.pie(
            values=list(itinerary.cost_breakdown.values()),
            names=[k. title() for k in itinerary.cost_breakdown.keys()],
            title="Cost Distribution",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[k.title() for k in itinerary.cost_breakdown.keys()],
                y=list(itinerary.cost_breakdown. values()),
                marker_color=['#8fd3f4', '#84fab0', '#ffc857', '#ff6b6b']
            )
        ])
        fig.update_layout(title="Cost by Category", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # HOTEL
    # ========================================================================
    
    if itinerary.hotel: 
        st.markdown(f'<div class="phase-header"><h2>🏨 Accommodation</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### {itinerary.hotel.name}")
            st.write(f"⭐ {itinerary. hotel.rating} stars")
        
        with col2:
            st.metric("Per Night", f"${itinerary.hotel.price_per_night:.0f}")
        
        with col3:
            st.metric("Total", f"${itinerary.hotel.price_per_night * itinerary.num_days:.0f}")
    
    # ========================================================================
    # DAILY SCHEDULE
    # ========================================================================
    
    st.markdown(f'<div class="phase-header"><h2>📅 Daily Schedule</h2></div>', unsafe_allow_html=True)
    
    # Day selector
    day_options = [f"Day {d. day_number} - {d.date.strftime('%A, %b %d')}" for d in itinerary. days]
    selected_day_idx = st.selectbox("Select Day", range(len(day_options)), format_func=lambda i: day_options[i])
    
    day_plan = itinerary.days[selected_day_idx]
    
    # Day summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Activities", len(day_plan.activities))
    with col2:
        st.metric("Daily Cost", f"${day_plan.total_cost:.0f}")
    with col3:
        attraction_count = sum(1 for a in day_plan.activities if a.type == "attraction")
        st.metric("Attractions", attraction_count)
    
    # Timeline
    st.markdown("#### 🕐 Timeline")
    
    for activity in day_plan.activities:
        icon = "🎨" if activity.type == "attraction" else "🍽️"
        
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st. write(f"**{activity.start_time.strftime('%H:%M')}**")
                st.caption(f"{activity.duration_hours}h")
            
            with col2:
                st.markdown(f"{icon} **{activity.name}**")
            
            with col3:
                st.write(f"${activity.cost:.0f}")
        
        st.divider()
    
    # ========================================================================
    # ALL ATTRACTIONS MAP
    # ========================================================================
    
    st.markdown(f'<div class="phase-header"><h2>🗺️ Attractions Map</h2></div>', unsafe_allow_html=True)
    
    # Collect all unique attractions
    all_attractions = []
    for day in itinerary.days:
        for activity in day.activities:
            if activity.type == "attraction":
                all_attractions. append({
                    "name": activity. name,
                    "lat": activity.location.lat,
                    "lon": activity.location.lon,
                    "day":  day.day_number
                })
    
    if all_attractions:
        import pandas as pd
        df = pd. DataFrame(all_attractions)
        
        # Create map
        st.map(df[['lat', 'lon']], zoom=12)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    st.markdown(f'<div class="phase-header"><h2>📄 Export</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate text version
        text_itinerary = f"""
{itinerary.destination} - {itinerary.num_days} Day Itinerary
{'='*50}

Total Cost: ${itinerary.total_cost:.2f}
Dates: {itinerary.start_date} to {itinerary.end_date}

Hotel: {itinerary.hotel. name if itinerary.hotel else 'N/A'}
Price: ${itinerary.hotel. price_per_night:.2f}/night

Daily Schedule:
"""
        for day in itinerary.days:
            text_itinerary += f"\nDay {day.day_number} - {day.date}\n"
            text_itinerary += f"Cost: ${day.total_cost:.2f}\n\n"
            for activity in day.activities:
                text_itinerary += f"  {activity.start_time.strftime('%H:%M')} - {activity.name} (${activity.cost:.2f})\n"
        
        st.download_button(
            label="📥 Download as Text",
            data=text_itinerary,
            file_name=f"{itinerary.destination}_itinerary.txt",
            mime="text/plain"
        )
    
    with col2:
        st.button("🔄 Plan New Trip", on_click=lambda:  setattr(st.session_state, 'itinerary', None))


# ============================================================================
# FOOTER
# ============================================================================

st. markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Made with ❤️ using AI • Powered by LLMs + ML"
    "</div>",
    unsafe_allow_html=True
)