from pydantic import BaseModel,Field
from typing import List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END 
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

class Activity(BaseModel):
    name: str = Field(description="Name of the activity")
    category: str = Field(description="Category of the activity")
    hours: float = Field(description="Duration of the activity in hours")
    cost: float = Field(description="Cost of the activity")

class Day(BaseModel):
    date: str = Field(description="Date of the day")
    activities: Optional[List[Activity]] = Field(default_factory=list, description="List of activities planned for the day")

class Trip(BaseModel):
    destination: str = Field(description="Destination of the trip")
    days: List[Day] = Field(description="List of days in the trip")
    cost: float = Field(description="Total cost of the trip")

@tool
def calculate_distance(origin: str, destination: str) -> str:
    "Calculates the driving distance between two cities. Use this to estimate travel time."
    return f"Calculating distance from {origin} to {destination}...\n\nThe driving distance from {origin} to {destination} is approximately 245 km (152 miles). Estimated driving time: 2 hours 45 minutes via highway. Alternative routes available with scenic views may take 3-4 hours."

@tool
def search_places(query:str)->str:
    "Search for places of interest based on a query string."
    return f"Searching for places matching '{query}'...\n\nFound several places matching '{query}':\n\n1) Historic Downtown District - Features 18th century architecture, local museums, and artisan shops. Open daily 9AM-6PM.\n\n2) Riverside Park & Marina - Scenic waterfront with walking trails, boat rentals, and picnic areas.\n\n3) Cultural Arts Center - Hosts rotating exhibitions, live performances, and workshops. Entry fee: $12 adults, $8 students.\n\n4) Local Food Market - Fresh produce, regional specialties, and street food. Peak hours: 10AM-2PM weekends."

@tool
def get_weather(location: str)->str:
    "Get the current weather for a given location."
    return f"Getting weather information for {location}...\n\nWeather in {location}:\nCurrently 22°C (72°F) with partly cloudy skies.\nHumidity: 65%\nWind: 8 km/h from southwest\n\nToday's forecast: High 28°C (82°F), Low 18°C (64°F)\n20% chance of light rain in the evening.\n\nTomorrow: Mostly sunny, 26°C\nUV Index: 6 (High) - sunscreen recommended\n\nGood conditions for outdoor activities until 6 PM."

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("api not found")
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)
tools = [calculate_distance, search_places, get_weather]
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(Trip)

class AgentState(MessagesState):
    final: Optional[Trip]

def router(state: AgentState):
    messages = state["messages"]
    last = messages[-1]

    if last.tool_calls:
        return "tools"
    return "generator"

def generator(state: AgentState):
    message = state["messages"]
    system_msg = SystemMessage(content="You are a travel architect. Review the conversation history and extract the final itinerary details into the structured format. If details are missing, make reasonable estimates based on the context.")
    response = structured_llm.invoke([system_msg]+message)
    return {"final": response}

def call_model(state: AgentState):
    past_messages = state["messages"]
    response = llm_with_tools.invoke(past_messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
tools_node = ToolNode(tools = tools)



workflow.add_node("generator",generator)
workflow.add_node("agent",call_model)
workflow.add_node("tools",tools_node)
workflow.add_edge(START,"agent")
workflow.add_edge("tools","agent")

workflow.add_conditional_edges("agent",router,{"tools": "tools","generator": "generator"})
workflow.add_edge("generator",END)
app = workflow.compile()

initial_input = {"messages": [HumanMessage(content="Plan a trip from New York to DC. Check the distance first.")]}

final_state = app.invoke(initial_input,config={"recursion_limit":10})
print(final_state["final"])