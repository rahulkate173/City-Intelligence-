import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient

# Load Environment Variables
load_dotenv()

# --- Initialize Tools ---
client = TavilyClient()

@tool
def get_news(query: str) -> str:
    """Takes query and return results given by tavily search tool"""
    query = f"Search for latest news in this city/state/country : {query}"
    result = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        topic="news"
    )
    results = result.get("results", [])
    
    if not results:
        return f"No news found for {query}"
    
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    return f"Latest news in {query}:\n\n" + "\n\n".join(news_list)

@tool
def get_weather(location: str) -> str:
    """For provided location it returns current weather of that location"""
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?appid={OPENWEATHER_API_KEY}&q={location}"
    response = requests.get(url)
    x = response.json()
    if x["cod"] != 200:
        return f"Unable to fetch weather for location {location}"
    desc = x["weather"][0]["description"]
    temp = x["main"]["temp"]
    return f"weather in {location}: {desc}, temp :{temp}"

# Setup LangChain & Bindings
llm = ChatMistralAI(model="mistral-large-latest")
llm_with_tools = llm.bind_tools([get_news, get_weather])
tools_dict = {"get_news": get_news, "get_weather": get_weather}

# --- Streamlit Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Holds the conversation history for LangChain
if "pending_tool_calls" not in st.session_state:
    st.session_state.pending_tool_calls = []  # Holds tool calls awaiting human approval

# --- UI Layout ---
st.set_page_config(page_title="City Intelligence Agent", page_icon="🏙️")
st.title("🏙️ City Intelligence")
st.caption("Ask about weather or latest news. Human-In-The-Loop configuration active.")

# Render Conversation History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.status(f"Tool Executed: {msg.tool_call_id[:8]}...", state="complete"):
            st.write(msg.content)

# --- Agent Processing Loop ---
def process_agent_loop():
    """Runs the internal reasoning loop until the model finishes or requires a tool approval."""
    while True:
        # Get AI decision
        result = llm_with_tools.invoke(st.session_state.messages)
        st.session_state.messages.append(result)
        
        if result.tool_calls:
            # Save pending tool calls to state and pause loop execution for human approval
            st.session_state.pending_tool_calls = list(result.tool_calls)
            st.rerun() 
        else:
            # Model final response complete (no tools requested)
            with st.chat_message("assistant"):
                st.write(result.content)
            break

# --- Step 1: User Input ---
# Disable input box if there is a tool waiting for human confirmation
user_input = st.chat_input("Ask something...", disabled=len(st.session_state.pending_tool_calls) > 0)

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.spinner("Agent is thinking..."):
        process_agent_loop()

# --- Step 2: Human-In-The-Loop Interception ---
if st.session_state.pending_tool_calls:
    # Look at the first pending tool call
    current_tool = st.session_state.pending_tool_calls[0]
    tool_name = current_tool["name"]
    tool_args = current_tool["args"]
    
    st.warning(f"🤖 **Agent requesting tool permission:** `{tool_name}` with arguments: `{tool_args}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve", use_container_width=True):
            with st.spinner(f"Executing {tool_name}..."):
                # Invoke the tool
                response = tools_dict[tool_name].invoke(current_tool)
                st.session_state.messages.append(ToolMessage(
                    content=response,
                    tool_call_id=current_tool["id"]
                ))
            # Remove this call from queue
            st.session_state.pending_tool_calls.pop(0)
            
            # Continue the loop if no more tools are blocked
            if not st.session_state.pending_tool_calls:
                with st.spinner("Resuming agent execution..."):
                    process_agent_loop()
            else:
                st.rerun()

    with col2:
        if st.button("❌ Deny", use_container_width=True, type="primary"):
            st.session_state.messages.append(ToolMessage(
                content=f"Error: User denied execution permission for tool '{tool_name}'. Provide an alternative response or ask for clarification.",
                tool_call_id=current_tool["id"]
            ))
            # Remove this call from queue
            st.session_state.pending_tool_calls.pop(0)
            
            if not st.session_state.pending_tool_calls:
                with st.spinner("Resuming agent execution..."):
                    process_agent_loop()
            else:
                st.rerun()