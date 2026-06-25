from dotenv import load_dotenv
load_dotenv() # take all api key out 
from langchain_core.messages import HumanMessage , AIMessage , ToolMessage
from langchain.tools import tool 
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient 
from rich import print # more generalize 
import requests
import os 

## creating customs tools
client = TavilyClient()
@tool
def get_news(query:str)->str:
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
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    return f"Latest news in {query}:\n\n" + "\n\n".join(news_list)

@tool
def get_weather(location:str) -> str:
    """For provided location it returns current weather of that location"""
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") # to pass externally 
    # url = f"http://openweathermap.org{location}&appid={OPENWEATHER_API_KEY}"
    url =f"http://api.openweathermap.org/data/2.5/weather?appid={OPENWEATHER_API_KEY}&q={location}"
    response = requests.get(url)
    x = response.json()
    # print(f"DEBUG:\n{response.json()}")  # test debug 
    if x["cod"] != 200:
        return f"Unable to fetch weather for location {location}"
    desc = x["weather"][0]["description"]
    temp = x["main"]["temp"]
    return f"weather in {location}: {desc}, temp :{temp}"


# print(get_weather.invoke("mumbai")) # test code 
# print(get_news.invoke("pune"))
## Including llm 
llm = ChatMistralAI(model="mistral-large-latest")
llm_with_tools = llm.bind_tools([get_news,get_weather]) # tool binding 
tools_dict = {
    "get_news":get_news,
    "get_weather":get_weather
}
messages = [] # to get the history 
## Agentic loop (human in loop)
print("City Intelligence \n\n to exit type exit !!")
while True: # user and agent loop 
    user = input("You : ")
    if user.lower() == "exit":
        break 
    messages.append(HumanMessage(content=user))
    while True: ## tool_message > llm (loop)
        result = llm_with_tools.invoke(messages) # llm tell weather to use tool or not 
        messages.append(result) # appending ai message 
        if result.tool_calls: # all tools are called 
            for tool_call in result.tool_calls: # if one or more tools exist
                tool_name = tool_call["name"]
                ## human in loop 
                confirm = input(f"Agent want to use this tool {tool_name} confirm with (yes/no) : ")
                if confirm == "no":
                    print(f"Demanded tool {tool_name} permission not accessed!!")
                    messages.append(ToolMessage( # required 
                        content=f"Error: User denied execution permission for tool '{tool_name}'. Provide an alternative response or ask for clarification.",
                        tool_call_id=tool_call["id"]
                    ))
                    break
                response = tools_dict[tool_name].invoke(tool_call) # tool called 
                messages.append(ToolMessage(  # how to understand two tools exist 
                    content = response,
                    tool_call_id = tool_call["id"]
                ))
            continue # go back directly to inner loop to refine message 
        else: # no tools exist 
            print(f"Agent:\n{result.content}") # normal result is printed 
            break