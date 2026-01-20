from mcp_server.tools.web_search import web_search_tool, execute_web_search
from mcp_server.tools.system_info import system_info_tool, execute_system_info

TOOLS = [web_search_tool, system_info_tool]

TOOL_HANDLERS = {
    "web_search": execute_web_search,
    "system_info": execute_system_info,
}
