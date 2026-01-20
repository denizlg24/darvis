import asyncio
from mcp_server.server import run_server
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    asyncio.run(run_server())
