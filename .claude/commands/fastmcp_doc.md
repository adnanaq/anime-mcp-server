TITLE: Defining a Simple FastMCP Tool and Server
DESCRIPTION: This Python code demonstrates how to initialize a FastMCP server, define a basic 'add' tool using a decorator, and run the server. The 'add' tool, which adds two numbers, becomes accessible to LLMs via the Model Context Protocol, showcasing FastMCP's intuitive and high-level approach to building MCP applications.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/welcome.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Install FastMCP Directly with uv pip or pip
DESCRIPTION: These commands demonstrate how to directly install the FastMCP package using either `uv pip` for `uv` users or the standard `pip` package manager. This is suitable for global or virtual environment installations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
uv pip install fastmcp
```

LANGUAGE: bash
CODE:

```
pip install fastmcp
```

---

TITLE: Call FastMCP Server with Gemini Python SDK
DESCRIPTION: This Python script demonstrates how to integrate a FastMCP client session with the Google Gemini SDK. It connects to a local FastMCP server and passes the client session to `genai.types.GenerateContentConfig`'s `tools` parameter, enabling the Gemini model to automatically invoke the exposed FastMCP tools based on the conversation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import Client
from google import genai
import asyncio

mcp_client = Client("server.py")
gemini_client = genai.Client()

async def main():
    async with mcp_client:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Roll 3 dice!",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[mcp_client.session],  # Pass the FastMCP client session
            ),
        )
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: Create a FastMCP Server with Dice Rolling Tool
DESCRIPTION: This Python code defines a FastMCP server named 'Dice Roller' with a single tool, `roll_dice`. The tool simulates rolling a specified number of 6-sided dice and returns the results. The server is configured to run on HTTP transport at port 8000.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

---

TITLE: FastMCP Client: Calling Tools and Reading Resources
DESCRIPTION: This Python snippet illustrates how to initialize the FastMCP `Client` and perform asynchronous operations. It demonstrates calling various tools like `weather_get_forecast`, `assistant_answer_question`, and `calendar_list_events`, as well as reading resources such as `weather/icons/sunny` and `assistant/docs/mcp`. The code highlights the use of server name prefixes for tools and URI paths for resources.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_16

LANGUAGE: Python
CODE:

```
# Create a transport from the config (happens automatically with Client)
client = Client(config)

async def main():
    async with client:
        # Tools are accessible with server name prefixes
        weather = await client.call_tool("weather_get_forecast", {"city": "London"})
        answer = await client.call_tool("assistant_answer_question", {"query": "What is MCP?"})
        events = await client.call_tool("calendar_list_events", {"date": "2023-06-01"})

        # Resources use prefixed URI paths
        icons = await client.read_resource("weather://weather/icons/sunny")
        docs = await client.read_resource("resource://assistant/docs/mcp")

asyncio.run(main())
```

---

TITLE: Add FastMCP as Project Dependency with uv
DESCRIPTION: This command adds FastMCP as a direct dependency to your project using the `uv` package manager. It's recommended for integrating FastMCP into existing projects.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_0

LANGUAGE: bash
CODE:

```
uv add fastmcp
```

---

TITLE: Connect to FastMCP Server and Interact with Client
DESCRIPTION: Demonstrates how to instantiate the `Client` class, connect to a FastMCP server, list available resources, and call a tool. This example uses an asynchronous context manager for connection management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-client.mdx#_snippet_0

LANGUAGE: python
CODE:

```
client = Client("http://localhost:8080")

async with client:
    # List available resources
    resources = await client.list_resources()

    # Call a tool
    result = await client.call_tool("my_tool", {"param": "value"})
```

---

TITLE: FastMCP Streamable HTTP: Default Server and Client Setup
DESCRIPTION: This snippet demonstrates how to initialize and run a FastMCP server using the default Streamable HTTP transport. It also shows how to connect a client to this server and perform a basic `ping` operation, utilizing the default host, port, and path.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

if __name__ == "__main__":
    mcp.run(transport="http")
```

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:8000/mcp/") as client:
        await client.ping()

if __name__ == "__main__":
    asyncio.run(example())
```

---

TITLE: Define Tool Parameters with Standard Python Type Annotations
DESCRIPTION: Standard Python type annotations are crucial for FastMCP tool parameters. They inform the LLM about expected data types, enable input validation, and facilitate accurate JSON schema generation for the MCP protocol.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_3

LANGUAGE: python
CODE:

```
@mcp.tool
def analyze_text(
    text: str,
    max_tokens: int = 100,
    language: str | None = None
) -> dict:
    """Analyze the provided text."""
    # Implementation...
```

---

TITLE: Install FastMCP Library
DESCRIPTION: Installs the FastMCP Python package using pip, a prerequisite for building and interacting with MCP servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_0

LANGUAGE: bash
CODE:

```
pip install fastmcp
```

---

TITLE: Recommended Pattern for Registering Instance Methods with FastMCP
DESCRIPTION: This example demonstrates the correct approach for integrating instance methods with FastMCP. Instead of direct decoration, an instance of the class (`obj`) is created first. By accessing the method through the instance (`obj.add`), Python creates a bound method where `self` is already set. Registering this bound method ensures that the system sees a callable that only expects the appropriate parameters, effectively hiding `self` from the LLM.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class MyClass:
    def add(self, x, y):
        return x + y

# Create an instance first, then register the bound methods
obj = MyClass()
mcp.tool(obj.add)

# Now you can call it without 'self' showing up as a parameter
await mcp._mcp_call_tool('add', {'x': 1, 'y': 2})  # Returns 3
```

---

TITLE: Using the @mcp.tool Decorator in Python
DESCRIPTION: This snippet demonstrates the simplified usage of the `@mcp.tool` decorator in FastMCP, allowing for a more Pythonic way to define tools. It shows how to apply the decorator directly to a function, aligning with standard Python practices.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/changelog.mdx#_snippet_0

LANGUAGE: Python
CODE:

```
@mcp.tool
def my_tool():
    ...
```

---

TITLE: Initialize FastMCP Server in Python
DESCRIPTION: Demonstrates how to create a basic FastMCP server by instantiating the `FastMCP` class. This sets up the core server object.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")
```

---

TITLE: Instantiating a FastMCP Server in Python
DESCRIPTION: Demonstrates how to create a basic FastMCP server instance and how to include initial instructions for client interaction.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Create a basic server instance
mcp = FastMCP(name="MyAssistantServer")

# You can also add instructions for how to interact with the server
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)
```

---

TITLE: FastMCP Deep Research Server Implementation
DESCRIPTION: This Python code provides a reference implementation for a FastMCP server compatible with ChatGPT's Deep Research feature. It defines a `create_server` function that sets up `search` and `fetch` tools to process records from a JSON file. The `search` tool performs a simple keyword search, returning matching record IDs, while the `fetch` tool retrieves full record content by ID. Both tools include detailed docstrings for ChatGPT's understanding.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/chatgpt.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import json
from pathlib import Path
from dataclasses import dataclass
from fastmcp import FastMCP

@dataclass
class Record:
    id: str
    title: str
    text: str
    metadata: dict

def create_server(
    records_path: Path | str,
    name: str | None = None,
    instructions: str | None = None,
) -> FastMCP:
    """Create a FastMCP server that can search and fetch records from a JSON file."""
    records = json.loads(Path(records_path).read_text())

    RECORDS = [Record(**r) for r in records]
    LOOKUP = {r.id: r for r in RECORDS}

    mcp = FastMCP(name=name or "Deep Research MCP", instructions=instructions)

    @mcp.tool()
    async def search(query: str):
        """
        Simple unranked keyword search across title, text, and metadata.
        Searches for any of the query terms in the record content.
        Returns a list of matching record IDs for ChatGPT to fetch.
        """
        toks = query.lower().split()
        ids = []
        for r in RECORDS:
            record_txt = " ".join(
                [r.title, r.text, " ".join(r.metadata.values())]
            ).lower()
            if any(t in record_txt for t in toks):
                ids.append(r.id)

        return {"ids": ids}

    @mcp.tool()
    async def fetch(id: str):
        """
        Fetch a record by ID.
        Returns the complete record data for ChatGPT to analyze and cite.
        """
        if id not in LOOKUP:
            raise ValueError(f"Unknown record ID: {id}")
        return LOOKUP[id]

    return mcp

if __name__ == "__main__":
    mcp = create_server("path/to/records.json")
    mcp.run(transport="http", port=8000)
```

---

TITLE: Create FastMCP Server from OpenAPI Specification
DESCRIPTION: Initializes a FastMCP server by providing an httpx.AsyncClient and an OpenAPI specification. This code snippet demonstrates how FastMCP automatically converts API endpoints into callable tools for LLMs, using a simplified JSONPlaceholder API spec.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_1

LANGUAGE: python
CODE:

```
import httpx
from fastmcp import FastMCP

# Create an HTTP client for the target API
client = httpx.AsyncClient(base_url="https://jsonplaceholder.typicode.com")

# Define a simplified OpenAPI spec for JSONPlaceholder
openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "JSONPlaceholder API", "version": "1.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "Get all users",
                "operationId": "get_users",
                "responses": {"200": {"description": "A list of users."}}
            }
        },
        "/users/{id}": {
            "get": {
                "summary": "Get a user by ID",
                "operationId": "get_user_by_id",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "A single user."}}
            }
        }
    }
}

# Create the MCP server from the OpenAPI spec
mcp = FastMCP.from_openapi(
    openapi_spec=openapi_spec,
    client=client,
    name="JSONPlaceholder MCP Server"
)

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

---

TITLE: Create a Basic FastMCP Server Instance
DESCRIPTION: Initializes a FastMCP application by creating an instance of the FastMCP class. This object serves as the fundamental container for all tools and resources that the MCP server will provide.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Create a server instance with a descriptive name
mcp = FastMCP(name="My First MCP Server")
```

---

TITLE: Initialize FastMCP Server and Define a Tool
DESCRIPTION: This snippet demonstrates how to initialize a FastMCP application instance and define a simple tool using the `@mcp.tool` decorator. The `mcp.run()` call starts the server, defaulting to STDIO transport for local execution.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_11

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()  # Default: uses STDIO transport
```

---

TITLE: In-Memory Testing of FastMCP Servers with Pytest
DESCRIPTION: This snippet demonstrates how to perform efficient in-memory testing of a FastMCP server using a Client and pytest. It shows the setup of a pytest fixture to create a FastMCP server instance with a defined tool, and then how to pass this server directly to the Client constructor to call the tool and assert its output, thereby avoiding the overhead of starting a separate server process.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/testing.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import pytest
from fastmcp import FastMCP, Client

@pytest.fixture
def mcp_server():
    server = FastMCP("TestServer")

@server.tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return server

async def test_tool_functionality(mcp_server):
    # Pass the server directly to the Client constructor
    async with Client(mcp_server) as client:
        result = await client.call_tool("greet", {"name": "World"})
        assert result[0].text == "Hello, World!"
```

---

TITLE: Running FastMCP Server with Python Main Block
DESCRIPTION: Explains how to make the FastMCP server executable directly via Python by adding `mcp.run()` within the `if __name__ == "__main__":` block. This enables running the server using `python my_server.py`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Define a FastMCP Prompt Template
DESCRIPTION: Demonstrates how to create reusable message templates to guide LLM interactions using the `@mcp.prompt` decorator. Functions decorated as prompts can return strings or `Message` objects.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_6

LANGUAGE: python
CODE:

```
@mcp.prompt
def summarize_request(text: str) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text:\n\n{text}"
```

---

TITLE: Run or Connect to FastMCP Server
DESCRIPTION: Executes a local FastMCP server or establishes a connection to a remote one. The server can be specified as a Python module, an importable object (e.g., `file:obj`), or a URL for remote connections. It supports various transport protocols (stdio, streamable-http, sse) and allows configuration of host, port, and log level. Additional server arguments can be passed after a `--` separator.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-cli-cli.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
run(ctx: typer.Context, server_spec: str = typer.Argument(..., help='Python file, object specification (file:obj), or URL'), transport: Annotated[str | None, typer.Option('--transport', '-t', help='Transport protocol to use (stdio, streamable-http, or sse)')] = None, host: Annotated[str | None, typer.Option('--host', help='Host to bind to when using http transport (default: 127.0.0.1)')] = None, port: Annotated[int | None, typer.Option('--port', '-p', help='Port to bind to when using http transport (default: 8000)')] = None, log_level: Annotated[str | None, typer.Option('--log-level', '-l', help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')] = None) -> None
```

---

TITLE: Register Tool with FastMCP Server
DESCRIPTION: Registers a function as a tool within the FastMCP server. Tools can optionally request a Context object for server capabilities like logging and progress reporting. This decorator supports various calling patterns, including direct function calls and decorator usage with or without custom names.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_4

LANGUAGE: APIDOC
CODE:

```
tool(self, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionTool] | FunctionTool

Args:
- name_or_fn: Either a function (when used as @tool), a string name, or None.
- name: Optional name for the tool (keyword-only, alternative to name_or_fn).
- description: Optional description of what the tool does.
- tags: Optional set of tags for categorizing the tool.
- annotations: Optional annotations about the tool's behavior (e.g. {"is_async": True}).
- exclude_args: Optional list of argument names to exclude from the tool schema.
- enabled: Optional boolean to enable or disable the tool.
```

---

TITLE: Create a Basic FastMCP Server with a Tool
DESCRIPTION: This Python snippet demonstrates how to initialize a FastMCP server, define a simple 'add' tool using the @mcp.tool decorator, and run the server locally. The tool takes two integers and returns their sum, making it available for LLM applications.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_0

LANGUAGE: Python
CODE:

```
# server.py
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Define a FastMCP Tool for Multiplication
DESCRIPTION: Shows how to define a Python function as a FastMCP tool using the `@mcp.tool` decorator. Tools allow LLMs to perform actions by executing your Python functions, with FastMCP handling schema generation from type hints and docstrings.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_4

LANGUAGE: python
CODE:

```
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b
```

---

TITLE: Defining a Tool with FastMCP in Python
DESCRIPTION: Illustrates how to define a tool function using the `@mcp.tool` decorator, making it callable by FastMCP clients. Tools perform actions or access external systems.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_2

LANGUAGE: python
CODE:

```
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b
```

---

TITLE: Define a reusable LLM prompt with FastMCP in Python
DESCRIPTION: This Python snippet demonstrates how to define a reusable, parameterized prompt using the `@mcp.prompt` decorator in FastMCP. The `summarize_text` function takes `text_to_summarize` as input and returns a formatted string that instructs an LLM to provide a concise, one-paragraph summary. It showcases how to create consistent instructions for guiding LLM behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/mcp.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.prompt
def summarize_text(text_to_summarize: str) -> str:
    """Creates a prompt asking the LLM to summarize a piece of text."""
    return f"""
        Please provide a concise, one-paragraph summary of the following text:

        {text_to_summarize}
        """
```

---

TITLE: Basic Tool Execution with FastMCP Client
DESCRIPTION: Shows how to execute a server-side tool using `client.call_tool()` with a tool name and arguments, and how to access the text content of the result.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_1

LANGUAGE: python
CODE:

```
async with client:
    # Simple tool call
    result = await client.call_tool("add", {"a": 5, "b": 3})
    # result -> list[mcp.types.TextContent | mcp.types.ImageContent | ...]

    # Access the result content
    print(result[0].text)  # Assuming TextContent, e.g., '8'
```

---

TITLE: Add Parameter Metadata using Pydantic Annotated and Field
DESCRIPTION: Utilize `typing.Annotated` and `pydantic.Field` to attach rich metadata to tool parameters. This modern approach separates type hints from validation rules, allowing for descriptions, range constraints, and other validation features.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from typing import Annotated
from pydantic import Field

@mcp.tool
def process_image(
    image_url: Annotated[str, Field(description="URL of the image to process")],
    resize: Annotated[bool, Field(description="Whether to resize the image")] = False,
    width: Annotated[int, Field(description="Target width in pixels", ge=1, le=2000)] = 800,
    format: Annotated[
        Literal["jpeg", "png", "webp"],
        Field(description="Output image format")
    ] = "jpeg"
) -> dict:
    """Process an image with optional resizing."""
    # Implementation...
```

---

TITLE: Discovering Tools with FastMCP Client
DESCRIPTION: Demonstrates how to use `client.list_tools()` to retrieve and inspect available tools on the MCP server, including their names, descriptions, and input schemas.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_0

LANGUAGE: python
CODE:

```
async with client:
    tools = await client.list_tools()
    # tools -> list[mcp.types.Tool]

    for tool in tools:
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description}")
        if tool.inputSchema:
            print(f"Parameters: {tool.inputSchema}")
```

---

TITLE: Creating and Interacting with FastMCP Client
DESCRIPTION: This snippet demonstrates how to instantiate the `fastmcp.Client` class using various server sources like in-memory, HTTP URL, or a local Python script. It also shows basic asynchronous interactions such as pinging the server, listing tools, resources, and prompts, and executing a tool call within an `async with` context manager for proper connection management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("TestServer")
client = Client(server)

# HTTP server
client = Client("https://example.com/mcp")

# Local Python script
client = Client("my_mcp_server.py")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        # Execute operations
        result = await client.call_tool("example_tool", {"param": "value"})
        print(result)

asyncio.run(main())
```

---

TITLE: Complete FastMCP Server Implementation
DESCRIPTION: This comprehensive Python code provides the full implementation of a FastMCP server, demonstrating the creation of the server instance, definition of a tool, and both static and dynamic resources. It culminates in the `__main__` block for server execution, with inline comments explaining each step.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# 1. Create the server
mcp = FastMCP(name="My First MCP Server")

# 2. Add a tool
@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

# 3. Add a static resource
@mcp.resource("resource://config")
def get_config() -> dict:
    """Provides the application's configuration."""
    return {"version": "1.0", "author": "MyTeam"}

# 4. Add a resource template for dynamic content
@mcp.resource("greetings://{name}")
def personalized_greeting(name: str) -> str:
    """Generates a personalized greeting for the given name."""
    return f"Hello, {name}! Welcome to the MCP server."

# 5. Make the server runnable
if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Defining an MCP Tool with FastMCP
DESCRIPTION: Demonstrates how to define an executable action (Tool) using the @mcp.tool decorator in FastMCP. Tools are like POST requests, used to perform actions or change state. This example shows a 'get_weather' tool that takes a city and returns weather information.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/mcp.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

# This function is now an MCP tool named "get_weather"
@mcp.tool
def get_weather(city: str) -> dict:
    """Gets the current weather for a specific city."""
    # In a real app, this would call a weather API
    return {"city": city, "temperature": "72F", "forecast": "Sunny"}
```

---

TITLE: Define Basic Prompts with FastMCP
DESCRIPTION: Demonstrates how to define basic prompt functions using the `@mcp.prompt` decorator in FastMCP. It shows examples of returning a simple string (auto-converted to a user message) and returning a specific `PromptMessage` object for more control over message role and content. It highlights how the function name becomes the prompt identifier and the docstring becomes the description.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent

mcp = FastMCP(name="PromptServer")

# Basic prompt returning a string (converted to user message automatically)
@mcp.prompt
def ask_about_topic(topic: str) -> str:
    """Generates a user message asking for an explanation of a topic."""
    return f"Can you please explain the concept of '{topic}'?"

# Prompt returning a specific message type
@mcp.prompt
def generate_code_request(language: str, task_description: str) -> PromptMessage:
    """Generates a user message requesting code generation."""
    content = f"Write a {language} function that performs the following task: {task_description}"
    return PromptMessage(role="user", content=TextContent(type="text", text=content))
```

---

TITLE: Use Context object in a fastmcp tool function
DESCRIPTION: Demonstrates how to inject and use the `Context` object within a `fastmcp` tool function. It shows examples of logging messages (info, debug, warning, error), reporting progress, accessing resources, and retrieving request information like `request_id` and `client_id`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-context.mdx#_snippet_1

LANGUAGE: python
CODE:

```
@server.tool
def my_tool(x: int, ctx: Context) -> str:
    # Log messages to the client
    ctx.info(f"Processing {x}")
    ctx.debug("Debug info")
    ctx.warning("Warning message")
    ctx.error("Error message")

    # Report progress
    ctx.report_progress(50, 100, "Processing")

    # Access resources
    data = ctx.read_resource("resource://data")

    # Get request info
    request_id = ctx.request_id
    client_id = ctx.client_id

    return str(x)
```

---

TITLE: Configure Bearer Token Authentication with JWKS URI (Simplified)
DESCRIPTION: A concise example demonstrating the configuration of `BearerAuthProvider` solely using a JWKS URI, which is recommended for production environments due to its support for automatic key rotation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_3

LANGUAGE: python
CODE:

```
provider = BearerAuthProvider(
    jwks_uri="https://idp.example.com/.well-known/jwks.json"
)
```

---

TITLE: Running FastMCP Server with Python `run()` Method
DESCRIPTION: This snippet demonstrates how to run a FastMCP server directly from a Python script by calling the `run()` method on a `FastMCP` instance. It includes a simple tool definition and illustrates the best practice of placing the `run()` call within an `if __name__ == "__main__":` block to ensure the server starts only when the script is executed directly.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Access FastMCP Context in a Tool
DESCRIPTION: Illustrates how to access the `Context` object within a FastMCP tool by adding a `ctx: Context` parameter. The Context provides methods for logging, LLM sampling, HTTP requests, resource access, and progress reporting, enabling interaction with MCP session capabilities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context

mcp = FastMCP("My MCP Server")

@mcp.tool
async def process_data(uri: str, ctx: Context):
    # Log a message to the client
    await ctx.info(f"Processing {uri}...")

    # Read a resource from the server
    data = await ctx.read_resource(uri)

    # Ask client LLM to summarize the data
    summary = await ctx.sample(f"Summarize: {data.content[:500]}")

    # Return the summary
    return summary.text
```

---

TITLE: Run FastMCP Server with Streamable HTTP Transport
DESCRIPTION: This snippet illustrates how to start the FastMCP server using the Streamable HTTP transport. This protocol is recommended for web deployments and allows specifying the host, port, and path for the server endpoint.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_13

LANGUAGE: python
CODE:

```
mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
```

---

TITLE: Define a FastMCP Component with Tools, Resources, and Prompts
DESCRIPTION: Illustrates a complete `MyComponent` class inheriting from `MCPMixin`, showcasing the definition of various FastMCP elements: tools (basic, disabled, excluded arguments, annotated), resources (basic, disabled), and prompts (basic, disabled) using their respective decorators.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/src/fastmcp/contrib/mcp_mixin/README.md#_snippet_0

LANGUAGE: python
CODE:

```
from mcp.types import ToolAnnotations
from fastmcp import FastMCP
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool, mcp_resource, mcp_prompt

class MyComponent(MCPMixin):
    @mcp_tool(name="my_tool", description="Does something cool.")
    def tool_method(self):
        return "Tool executed!"

    # example of disabled tool
    @mcp_tool(name="my_tool", description="Does something cool.", enabled=False)
    def disabled_tool_method(self):
        # This function can't be called by client because it's disabled
        return "You'll never get here!"

    # example of excluded parameter tool
    @mcp_tool(
        name="my_tool", description="Does something cool.",
        enabled=False, exclude_args=['delete_everything'],
    )
    def excluded_param_tool_method(self, delete_everything=False):
        # MCP tool calls can't pass the "delete_everything" argument
        if delete_everything:
            return "Nothing to delete, I bet you're not a tool :)"
        return "You might be a tool if..."

    # example tool w/annotations
    @mcp_tool(
        name="my_tool", description="Does something cool.",
        annotations=ToolAnnotations(
            title="Attn LLM, use this tool first!",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
        )
    )
    def tool_method(self):
        return "Tool executed!"

    # example tool w/everything
    @mcp_tool(
        name="my_tool", description="Does something cool.",
        enabled=True,
        exclude_args=['delete_all'],
        annotations=ToolAnnotations(
            title="Attn LLM, use this tool first!",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
        )
    )
    def tool_method(self, delete_all=False):
        if delete_all:
            return "99 records deleted. I bet you're not a tool :)"
        return "Tool executed, but you might be a tool!"

    @mcp_resource(uri="component://data")
    def resource_method(self):
        return {"data": "some data"}

    # Disabled resource
    @mcp_resource(uri="component://data", enabled=False)
    def resource_method(self):
        return {"data": "some data"}

    # prompt
    @mcp_prompt(name="A prompt")
    def prompt_method(self, name):
        return f"Whats up {name}?"

    # disabled prompt
    @mcp_prompt(name="A prompt", enabled=False)
    def prompt_method(self, name):
        return f"Whats up {name}?"
```

---

TITLE: Connect FastMCP Client to Servers (Stdio, SSE)
DESCRIPTION: Demonstrates how to initialize a `fastmcp.Client` to connect to an MCP server via stdio (local script) and SSE (HTTP endpoint). It shows how to list available tools and call a specific tool, printing the result.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_8

LANGUAGE: python
CODE:

```
from fastmcp import Client

async def main():
    # Connect via stdio to a local script
    async with Client("my_server.py") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        result = await client.call_tool("add", {"a": 5, "b": 3})
        print(f"Result: {result.text}")

    # Connect via SSE
    async with Client("http://localhost:8000/sse") as client:
        # ... use the client
        pass
```

---

TITLE: Defining a Basic FastMCP Tool
DESCRIPTION: Demonstrates how to define a simple tool by decorating a Python function with @mcp.tool. FastMCP automatically infers the tool name, description, and input schema from the function's name, docstring, and type annotations, streamlining the process of exposing Python functions as LLM capabilities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="CalculatorServer")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b
```

---

TITLE: Registering a Tool with FastMCP Server in Python
DESCRIPTION: Shows how to add a simple greeting tool to the FastMCP server. A Python function is decorated with `@mcp.tool` to register it, making it callable by clients.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

---

TITLE: FastMCP Context Sampling Method (`ctx.sample`)
DESCRIPTION: Details the `ctx.sample` method for requesting text generation from the client's LLM. It supports various message formats, optional system prompts, temperature, max tokens, and model preferences, returning either TextContent or ImageContent.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_8

LANGUAGE: APIDOC
CODE:

```
ctx.sample(messages: str | list[str | SamplingMessage], system_prompt: str | None = None, temperature: float | None = None, max_tokens: int | None = None, model_preferences: ModelPreferences | str | list[str] | None = None) -> TextContent | ImageContent
  messages: A string or list of strings/message objects to send to the LLM
  system_prompt: Optional system prompt to guide the LLM's behavior
  temperature: Optional sampling temperature (controls randomness)
  max_tokens: Optional maximum number of tokens to generate (defaults to 512)
  model_preferences: Optional model selection preferences (e.g., a model hint string, list of hints, or a ModelPreferences object)
  Returns the LLM's response as TextContent or ImageContent
```

---

TITLE: Configure Bearer Token Authentication with JWKS URI
DESCRIPTION: This example demonstrates how to configure FastMCP to use Bearer Token authentication by instantiating a `BearerAuthProvider` with a JWKS URI, issuer, and audience. The configured authentication provider is then passed to the `FastMCP` instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="https://my-identity-provider.com/.well-known/jwks.json",
    issuer="https://my-identity-provider.com/",
    audience="my-mcp-server"
)

mcp = FastMCP(name="My MCP Server", auth=auth)
```

---

TITLE: Combine Multiple Built-in Middleware in FastMCP
DESCRIPTION: This example illustrates how to integrate several FastMCP built-in middleware components to create a robust server configuration. It demonstrates adding `ErrorHandlingMiddleware`, `RateLimitingMiddleware`, `TimingMiddleware`, and `LoggingMiddleware` in a logical order to provide comprehensive monitoring, protection, and observability for an MCP server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_20

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

mcp = FastMCP("Production Server")

# Add middleware in logical order
mcp.add_middleware(ErrorHandlingMiddleware())  # Handle errors first
mcp.add_middleware(RateLimitingMiddleware(max_requests_per_second=50))
mcp.add_middleware(TimingMiddleware())  # Time actual execution
mcp.add_middleware(LoggingMiddleware())  # Log everything

@mcp.tool
def my_tool(data: str) -> str:
    return f"Processed: {data}"
```

---

TITLE: Configuring Authentication for FastMCP HTTP Client
DESCRIPTION: Shows how to configure bearer token authentication on an `httpx.AsyncClient` before creating a `FastMCP` server, ensuring all API requests made by the server are authenticated.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_14

LANGUAGE: python
CODE:

```
import httpx
from fastmcp import FastMCP

# Bearer token authentication
api_client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Create MCP server with authenticated client
mcp = FastMCP.from_openapi(..., client=api_client)
```

---

TITLE: Add a Tool to the FastMCP Server
DESCRIPTION: Demonstrates how to define a callable tool for an LLM by decorating a standard Python function with `@mcp.tool`. FastMCP automatically infers the tool's name, description from the docstring, and input schema from type hints, simplifying protocol boilerplate.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="My First MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b
```

---

TITLE: Supporting Asynchronous Prompts in FastMCP with Python
DESCRIPTION: Demonstrates how FastMCP seamlessly supports both synchronous ('def') and asynchronous ('async def') functions as prompts. Asynchronous prompts are recommended for I/O-bound operations like network requests or database queries.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_9

LANGUAGE: python
CODE:

```
# Synchronous prompt
@mcp.prompt
def simple_question(question: str) -> str:
    """Generates a simple question to ask the LLM."""
    return f"Question: {question}"

# Asynchronous prompt
@mcp.prompt
async def data_based_prompt(data_id: str) -> str:
    """Generates a prompt based on data that needs to be fetched."""
    # In a real scenario, you might fetch data from a database or API
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/data/{data_id}") as response:
            data = await response.json()
            return f"Analyze this data: {data['content']}"
```

---

TITLE: Define Flexible Parameters with Union and Optional Types in FastMCP
DESCRIPTION: This example demonstrates how to use Python's union (`|`) and optional (`| None`) types to create flexible FastMCP tool parameters. This allows parameters to accept multiple data types or be entirely omitted, providing adaptability for various client inputs while maintaining clear type hints.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_22

LANGUAGE: python
CODE:

```
@mcp.tool
def flexible_search(
    query: str | int,              # Can be either string or integer
    filters: dict[str, str] | None = None,  # Optional dictionary
    sort_field: str | None = None  # Optional string
):
    """Search with flexible parameter types."""
    # Implementation...
```

---

TITLE: Convert OpenAPI Spec to MCP Server (Python)
DESCRIPTION: Demonstrates how to create an MCP server from an OpenAPI specification using `FastMCP.from_openapi`. It involves setting up an HTTP client, loading the OpenAPI spec, and then initializing and running the MCP server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import httpx
from fastmcp import FastMCP

# Create an HTTP client for your API
client = httpx.AsyncClient(base_url="https://api.example.com")

# Load your OpenAPI spec
openapi_spec = httpx.get("https://api.example.com/openapi.json").json()

# Create the MCP server
mcp = FastMCP.from_openapi(
    openapi_spec=openapi_spec,
    client=client,
    name="My API Server"
)

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Connect to In-Memory FastMCP Server (Python)
DESCRIPTION: Shows how to use `fastmcp.client.transports.FastMCPTransport` to connect directly to a `FastMCP` server instance within the same Python process. This transport is automatically inferred when a `FastMCP` server instance is provided to the client and is extremely useful for testing and unit testing due to its efficient in-memory communication.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_14

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Client
import asyncio

# 1. Create your FastMCP server instance
server = FastMCP(name="InMemoryServer")

@server.tool
def ping():
    return "pong"

# 2. Create a client pointing directly to the server instance
client = Client(server)  # Transport is automatically inferred

async def main():
    async with client:
        result = await client.call_tool("ping")
        print(f"In-memory call result: {result}")

asyncio.run(main())
```

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.FastMCPTransport
Inferred From: An instance of fastmcp.server.FastMCP or a FastMCP 1.0 server (mcp.server.fastmcp.FastMCP)
Use Case: Connecting directly to a FastMCP server instance in the same Python process
```

---

TITLE: FastMCP Client Constructor Configuration Options
DESCRIPTION: This section details the various configuration options available when initializing a FastMCP `Client` instance. It covers parameters for transport, callback handlers, local context, and request timeouts.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_10

LANGUAGE: APIDOC
CODE:

```
Client Constructor Options:
  transport: Transport instance or source for automatic inference
  log_handler: Handle server log messages
  progress_handler: Monitor long-running operations
  sampling_handler: Respond to server LLM requests
  roots: Provide local context to servers
  timeout: Default timeout for requests (in seconds)
```

---

TITLE: Run FastMCP Server with STDIO or HTTP Transport
DESCRIPTION: Illustrates how to start a FastMCP server using the mcp.run() method. It shows the default STDIO transport and an example of configuring Streamable HTTP transport with a specific host and port. This pattern ensures compatibility with various MCP clients.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_9

LANGUAGE: python
CODE:

```
# my_server.py
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    # This runs the server, defaulting to STDIO transport
    mcp.run()

    # To use a different transport, e.g., Streamable HTTP:
    # mcp.run(transport="http", host="127.0.0.1", port=9000)
```

---

TITLE: Renaming Tool Arguments with ArgTransform
DESCRIPTION: Illustrates how to rename a tool argument using `ArgTransform` to make it more intuitive for an LLM. The example renames a generic `q` argument to `search_query` for a search function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform

mcp = FastMCP()

@mcp.tool
def search(q: str):
    """Searches for items in the database."""
    return database.search(q)

new_tool = Tool.from_tool(
    search,
    transform_args={
        "q": ArgTransform(name="search_query")
    }
)
```

---

TITLE: Using Pydantic Models for Structured Data in FastMCP
DESCRIPTION: Details how to use Pydantic models for complex, structured, and validated input data in FastMCP tools. FastMCP automatically handles data validation against the model schema and converts inputs from JSON strings or dictionaries into Pydantic model instances.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_29

LANGUAGE: python
CODE:

```
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    username: str
    email: str = Field(description="User's email address")
    age: int | None = None
    is_active: bool = True

@mcp.tool
def create_user(user: User):
    """Create a new user in the system."""
    # The input is automatically validated against the User model
    # Even if provided as a JSON string or dict
    # Implementation...
```

---

TITLE: FastMCP Supported Parameter Types
DESCRIPTION: This section outlines the various type annotations supported by FastMCP for defining tool parameters, including basic scalar types, collection types, date/time objects, and complex Pydantic models.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_6

LANGUAGE: APIDOC
CODE:

```
FastMCP Supported Types:
- Basic types: int, float, str, bool (Simple scalar values)
- Binary data: bytes (Binary content)
- Date and Time: datetime, date, timedelta (Date and time objects)
- Collection types: list[str], dict[str, int], set[int] (Collections of items)
- Optional types: float | None, Optional[float] (Parameters that may be null/omitted)
- Union types: str | int, Union[str, int] (Parameters accepting multiple types)
- Constrained types: Literal["A", "B"], Enum (Parameters with specific allowed values)
- Paths: Path (File system paths)
- UUIDs: UUID (Universally unique identifiers)
- Pydantic models: UserData (Complex structured data)
```

---

TITLE: Defining a Resource with FastMCP in Python
DESCRIPTION: Shows how to expose a data source as a resource using the `@mcp.resource` decorator, allowing clients to read specific data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_3

LANGUAGE: python
CODE:

```
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}
```

---

TITLE: Defining a Parameterized Resource Template with FastMCP in Python
DESCRIPTION: Demonstrates how to create a parameterized resource template using `@mcp.resource` with a URI pattern, enabling clients to request specific data based on parameters extracted from the URI.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_4

LANGUAGE: python
CODE:

```
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    # The {user_id} in the URI is extracted and passed to this function
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}
```

---

TITLE: Execute FastMCP Server Tools in Python
DESCRIPTION: This example illustrates how to list available tools on a FastMCP server and execute a specific tool with arguments. It demonstrates retrieving and printing the result of a tool call.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_5

LANGUAGE: python
CODE:

```
async with client:
    # List available tools
    tools = await client.list_tools()

    # Execute a tool
    result = await client.call_tool("multiply", {"a": 5, "b": 3})
    print(result[0].text)  # "15"
```

---

TITLE: Initialize FastMCP Server Instance
DESCRIPTION: Demonstrates how to create the central FastMCP application instance. This object holds your tools, resources, prompts, and manages connections and configurations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Create a server instance
mcp = FastMCP(name="MyAssistantServer")
```

---

TITLE: FastMCP Prompt with Typed Arguments and Generated Schema
DESCRIPTION: This snippet demonstrates a FastMCP prompt function utilizing complex Python type annotations (e.g., `list[int]`, `dict[str, str]`). It also shows the corresponding JSON structure that FastMCP generates for the MCP client, which includes JSON schema descriptions to guide clients on the expected string format for these complex types.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_3

LANGUAGE: python
CODE:

```
@mcp.prompt
def analyze_data(
    numbers: list[int],
    metadata: dict[str, str],
    threshold: float
) -> str:
    """Analyze numerical data."""
    avg = sum(numbers) / len(numbers)
    return f"Average: {avg}, above threshold: {avg > threshold}"
```

LANGUAGE: json
CODE:

```
{
  "name": "analyze_data",
  "description": "Analyze numerical data.",
  "arguments": [
    {
      "name": "numbers",
      "description": "Provide as a JSON string matching the following schema: {\"items\":{\"type\":\"integer\"},\"type\":\"array\"}",
      "required": true
    },
    {
      "name": "metadata",
      "description": "Provide as a JSON string matching the following schema: {\"additionalProperties\":{\"type\":\"string\"},\"type\":\"object\"}",
      "required": true
    },
    {
      "name": "threshold",
      "description": "Provide as a JSON string matching the following schema: {\"type\":\"number\"}",
      "required": true
    }
  ]
}
```

---

TITLE: Integrating FastMCP with FastAPI
DESCRIPTION: Illustrates how to mount a FastMCP server into a FastAPI application. Similar to Starlette, it emphasizes the necessity of passing the FastMCP app's lifespan context to the FastAPI app to correctly initialize the session manager for Streamable HTTP transport.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_8

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastapi import FastAPI
from starlette.routing import Mount

# Create your FastMCP server as well as any tools, resources, etc.
mcp = FastMCP("MyServer")

# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Create a FastAPI app and mount the MCP server
app = FastAPI(lifespan=mcp_app.lifespan)
app.mount("/mcp-server", mcp_app)
```

---

TITLE: Connect to Remote/Authenticated FastMCP Server (Python)
DESCRIPTION: This Python code snippet shows how to configure a `fastmcp.Client` to connect to a remote FastMCP server using a URL and authenticate with a `BearerAuth` token. This flexibility allows the Gemini SDK to interact with various FastMCP server deployments, local or remote, secured or unsecured.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

mcp_client = Client(
    "https://my-server.com/mcp/",
    auth=BearerAuth("<your-token>"),
)
```

---

TITLE: Access MCP Context in FastMCP Tools
DESCRIPTION: Demonstrates how to use the 'Context' object within a FastMCP tool to perform actions like logging, reading resources, reporting progress, and interacting with the client's LLM for sampling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_17

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context

mcp = FastMCP(name="ContextDemo")

@mcp.tool
async def process_data(data_uri: str, ctx: Context) -> dict:
    """Process data from a resource with progress reporting."""
    await ctx.info(f"Processing data from {data_uri}")

    # Read a resource
    resource = await ctx.read_resource(data_uri)
    data = resource[0].content if resource else ""

    # Report progress
    await ctx.report_progress(progress=50, total=100)

    # Example request to the client's LLM for help
    summary = await ctx.sample(f"Summarize this in 10 words: {data[:200]}")

    await ctx.report_progress(progress=100, total=100)
    return {
        "length": len(data),
        "summary": summary.text
    }
```

---

TITLE: Python: ArgTransform Usage Examples
DESCRIPTION: Illustrates various practical applications of the `ArgTransform` class for modifying tool arguments, including renaming, updating descriptions, setting default values, using default factories, changing types, hiding arguments, making arguments required, and combining multiple transformations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-tools-tool_transform.mdx#_snippet_1

LANGUAGE: python
CODE:

```
# Rename argument 'old_name' to 'new_name'
ArgTransform(name="new_name")
```

LANGUAGE: python
CODE:

```
# Change description only
ArgTransform(description="Updated description")
```

LANGUAGE: python
CODE:

```
# Add a default value (makes argument optional)
ArgTransform(default=42)
```

LANGUAGE: python
CODE:

```
# Add a default factory (makes argument optional)
ArgTransform(default_factory=lambda: time.time())
```

LANGUAGE: python
CODE:

```
# Change the type
ArgTransform(type=str)
```

LANGUAGE: python
CODE:

```
# Hide the argument entirely from clients
ArgTransform(hide=True)
```

LANGUAGE: python
CODE:

```
# Hide argument but pass a constant value to parent
ArgTransform(hide=True, default="constant_value")
```

LANGUAGE: python
CODE:

```
# Hide argument but pass a factory-generated value to parent
ArgTransform(hide=True, default_factory=lambda: uuid.uuid4().hex)
```

LANGUAGE: python
CODE:

```
# Make an optional parameter required (removes any default)
ArgTransform(required=True)
```

LANGUAGE: python
CODE:

```
# Combine multiple transformations
ArgTransform(name="new_name", description="New desc", default=None, type=int)
```

---

TITLE: Define FastMCP Tools with Naked Decorators in Python
DESCRIPTION: This snippet demonstrates the simplified 'naked' decorator usage introduced in FastMCP 2.7, allowing for more Pythonic and direct registration of functions as MCP tools. Decorators now also return the created objects, enhancing usability.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/updates.mdx#_snippet_0

LANGUAGE: python
CODE:

```
mcp = FastMCP()

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b
```

---

TITLE: Implement Basic Error Handling with FastMCP Middleware
DESCRIPTION: This Python snippet demonstrates how to create a custom `Middleware` to catch exceptions during message processing, log them, and track error statistics. It shows a fundamental approach to centralizing error management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_17

LANGUAGE: python
CODE:

```
import logging
from fastmcp.server.middleware import Middleware, MiddlewareContext

class SimpleErrorHandlingMiddleware(Middleware):
    def __init__(self):
        self.logger = logging.getLogger("errors")
        self.error_counts = {}

    async def on_message(self, context: MiddlewareContext, call_next):
        try:
            return await call_next(context)
        except Exception as error:
            # Log the error and track statistics
            error_key = f"{type(error).__name__}:{context.method}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            self.logger.error(f"Error in {context.method}: {type(error).__name__}: {error}")
            raise
```

---

TITLE: Implement Error Handling in FastMCP Tools
DESCRIPTION: Shows how to raise 'ToolError' for client-facing messages (always sent) and standard 'TypeError' for internal validation, illustrating how 'mask_error_details' affects their visibility to clients.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_14

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""

    if b == 0:
        # Error messages from ToolError are always sent to clients,
        # regardless of mask_error_details setting
        raise ToolError("Division by zero is not allowed.")

    # If mask_error_details=True, this message would be masked
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers.")

    return a / b
```

---

TITLE: Run FastMCP Server from Command Line
DESCRIPTION: This snippet shows the command-line instruction to start the FastMCP server. It executes the Python script directly, using the default STDIO transport for client communication.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_6

LANGUAGE: bash
CODE:

```
python my_mcp_server.py
```

---

TITLE: Defining an MCP Resource with FastMCP
DESCRIPTION: Illustrates how to define a read-only data source (Resource) using the @mcp.resource decorator in FastMCP. Resources are like GET requests, used to retrieve information idempotently. This example provides a 'system://status' resource.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/mcp.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

# This function provides a resource at the URI "system://status"
@mcp.resource("system://status")
def get_system_status() -> dict:
    """Returns the current operational status of the service."""
    return {"status": "all systems normal"}
```

---

TITLE: Complete FastMCP Server Example with Bearer Auth
DESCRIPTION: This comprehensive example provides a full FastMCP server setup with bearer token authentication. It includes key pair generation, `BearerAuthProvider` configuration, and a `roll_dice` tool. For demonstration, it prints the access token to the console, but this practice is strongly discouraged in production environments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_8

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
import random

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")

auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    print(f"\n---\n\n🔑 Dice Roller access token:\n\n{access_token}\n\n---\n")
    mcp.run(transport="http", port=8000)
```

---

TITLE: FastMCP Middleware Hook Parameters and Control Flow
DESCRIPTION: Details the parameters received by every middleware hook ('MiddlewareContext' and 'call_next') and explains how to control the request flow within the middleware chain, including options for continuing, modifying, stopping, or handling errors.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_6

LANGUAGE: APIDOC
CODE:

```
Middleware Hook Parameters:
- context: MiddlewareContext
  - Description: Contains information about the current request.
  - Properties:
    - method: string (e.g., "tools/call") - The MCP method name.
    - source: string ("client" or "server") - Where the request came from.
    - type: string ("request" or "notification") - Message type.
    - message: object - The MCP message data.
    - timestamp: datetime - When the request was received.
    - fastmcp_context: object (optional) - FastMCP Context object.

- call_next: function
  - Description: A function that continues the middleware chain. Must be called to proceed unless stopping processing entirely.

Control Flow Options:
- Continue processing: Call `await call_next(context)` to proceed.
- Modify the request: Change the context before calling `call_next`.
- Modify the response: Change the result after calling `call_next`.
- Stop the chain: Do not call `call_next` (rarely needed).
- Handle errors: Wrap `call_next` in try/catch blocks.
```

---

TITLE: Basic LLM Sampling Handler Implementation in Python
DESCRIPTION: This Python example provides a basic implementation of the `sampling_handler`. It demonstrates how to extract content from `SamplingMessage`s, utilize an optional `systemPrompt` from `SamplingParams`, and construct a placeholder response, illustrating the core logic for integrating with an LLM service.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/sampling.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext

async def basic_sampling_handler(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext
) -> str:
    # Extract message content
    conversation = []
    for message in messages:
        content = message.content.text if hasattr(message.content, 'text') else str(message.content)
        conversation.append(f"{message.role}: {content}")

    # Use the system prompt if provided
    system_prompt = params.systemPrompt or "You are a helpful assistant."

    # Here you would integrate with your preferred LLM service
    # This is just a placeholder response
    return f"Response based on conversation: {' | '.join(conversation)}"

client = Client(
    "my_mcp_server.py",
    sampling_handler=basic_sampling_handler
)
```

---

TITLE: Using In-Memory Transport for FastMCP Server Testing
DESCRIPTION: This snippet demonstrates how to create a FastMCP server and pass it directly to a Client, utilizing the in-memory transport. This method is preferred for testing as it eliminates network complexity and separate processes, making debugging and local development more efficient.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/CLAUDE.md#_snippet_0

LANGUAGE: python
CODE:

```
# Create your FastMCP server
mcp = FastMCP("TestServer")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Pass server directly to client - uses in-memory transport
async with Client(mcp) as client:
    result = await client.call_tool("greet", {"name": "World"})
```

---

TITLE: Running FastMCP Server using CLI Command
DESCRIPTION: Shows how to start a FastMCP server using the `fastmcp run` command-line interface. This method automatically handles server execution and ignores the `__main__` block.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_5

LANGUAGE: bash
CODE:

```
fastmcp run my_server.py:mcp
```

---

TITLE: FastMCP Client In-Memory Testing
DESCRIPTION: Illustrates how to connect a `fastmcp.Client` directly to a `FastMCP` server instance using the in-memory transport. This method is ideal for efficient testing, eliminating the need for process management or network calls during development.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Client

mcp = FastMCP("My MCP Server")

async def main():
    # Connect via in-memory transport
    async with Client(mcp) as client:
        # ... use the client
```

---

TITLE: FastMCP Client Transport Inference Examples
DESCRIPTION: This snippet illustrates how the `fastmcp.Client` automatically infers the appropriate transport mechanism based on the input provided during instantiation. Examples include creating clients for in-memory servers, local Python script servers, and remote HTTP servers, showcasing the client's flexibility in connection management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import Client, FastMCP

# Examples of transport inference
client_memory = Client(FastMCP("TestServer"))
client_script = Client("./server.py")
client_http = Client("https://api.example.com/mcp")
```

---

TITLE: Define Asynchronous FastMCP Resources
DESCRIPTION: Provides an example of creating an asynchronous FastMCP resource using `async def`. This approach is crucial for resource functions that perform I/O operations, ensuring the server remains non-blocking and responsive.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_6

LANGUAGE: python
CODE:

```
import aiofiles
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

@mcp.resource("file:///app/data/important_log.txt", mime_type="text/plain")
async def read_important_log() -> str:
    """Reads content from a specific log file asynchronously."""
    try:
        async with aiofiles.open("/app/data/important_log.txt", mode="r") as f:
            content = await f.read()
        return content
    except FileNotFoundError:
        return "Log file not found."
```

---

TITLE: Complete FastMCP Server with Bearer Authentication and Dice Roller Tool
DESCRIPTION: This comprehensive example demonstrates a FastMCP server configured with bearer token authentication. It includes RSA key pair generation, `BearerAuthProvider` setup, and a simple `roll_dice` tool, showing how to run an authenticated server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
import random

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")

auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    print(f"\n---\n\n🔑 Dice Roller access token:\n\n{access_token}\n\n---\n")
    mcp.run(transport="http", port=8000)
```

---

TITLE: Constrain Tool Parameters with Literal Types in FastMCP
DESCRIPTION: This snippet illustrates the use of `Literal` types to restrict FastMCP tool parameters to a predefined set of exact values. Literals enhance LLM understanding, provide robust input validation, and generate clear schemas, ensuring that only acceptable options are provided for parameters like sorting order or algorithm.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_23

LANGUAGE: python
CODE:

```
from typing import Literal

@mcp.tool
def sort_data(
    data: list[float],
    order: Literal["ascending", "descending"] = "ascending",
    algorithm: Literal["quicksort", "mergesort", "heapsort"] = "quicksort"
):
    """Sort data using specific options."""
    # Implementation...
```

---

TITLE: Update Python Import for FastMCP 2.0
DESCRIPTION: This Python code snippet shows the necessary change to import statements when migrating from the official MCP SDK's FastMCP 1.0 to FastMCP 2.0. The core server API remains highly compatible, often requiring only this import path adjustment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_4

LANGUAGE: python
CODE:

```
# Before
# from mcp.server.fastmcp import FastMCP

# After
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")
```

---

TITLE: Creating FastMCP Server from OpenAPI Specification
DESCRIPTION: This class method constructs a FastMCP server instance directly from a given OpenAPI specification dictionary. It requires an `httpx.AsyncClient` for internal communication and allows for custom route mapping and component handling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_16

LANGUAGE: python
CODE:

```
from_openapi(cls, openapi_spec: dict[str, Any], client: httpx.AsyncClient, route_maps: list[RouteMap] | None = None, route_map_fn: OpenAPIRouteMapFn | None = None, mcp_component_fn: OpenAPIComponentFn | None = None, mcp_names: dict[str, str] | None = None, tags: set[str] | None = None, **settings: Any) -> FastMCPOpenAPI
```

---

TITLE: FastMCP Pydantic Field Validation with Annotated
DESCRIPTION: Demonstrates how to use Pydantic's `Field` with `typing.Annotated` to define robust validation constraints for parameters in FastMCP tools. It covers numeric ranges, string patterns, and length constraints.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_30

LANGUAGE: python
CODE:

```
from typing import Annotated
from pydantic import Field

@mcp.tool
def analyze_metrics(
    # Numbers with range constraints
    count: Annotated[int, Field(ge=0, le=100)],         # 0 <= count <= 100
    ratio: Annotated[float, Field(gt=0, lt=1.0)],       # 0 < ratio < 1.0

    # String with pattern and length constraints
    user_id: Annotated[str, Field(
        pattern=r"^[A-Z]{2}\d{4}$",                     # Must match regex pattern
        description="User ID in format XX0000"
    )],

    # String with length constraints
    comment: Annotated[str, Field(min_length=3, max_length=500)] = "",

    # Numeric constraints
    factor: Annotated[int, Field(multiple_of=5)] = 10  # Must be multiple of 5
):
    """Analyze metrics with validated parameters."""
    # Implementation...
```

---

TITLE: Configure FastMCP Instance with Server-Specific Settings
DESCRIPTION: Illustrates how to configure a FastMCP instance with various server-specific settings during initialization. This includes defining optional dependencies, tag-based component exposure (include/exclude), and policies for handling duplicate registrations of tools, resources, and prompts.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_12

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Configure server-specific settings
mcp = FastMCP(
    name="ConfiguredServer",
    dependencies=["requests", "pandas>=2.0.0"],  # Optional server dependencies
    include_tags={"public", "api"},              # Only expose these tagged components
    exclude_tags={"internal", "deprecated"},     # Hide these tagged components
    on_duplicate_tools="error",                  # Handle duplicate registrations
    on_duplicate_resources="warn",
    on_duplicate_prompts="replace"
)
```

---

TITLE: Configuring FastMCP Transport Settings in Python
DESCRIPTION: Shows how to configure transport-specific settings like host, port, and log level when running a FastMCP server, both synchronously and asynchronously. These settings override global defaults for network behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_16

LANGUAGE: python
CODE:

```
mcp.run(
    transport="http",
    host="0.0.0.0",
    port=9000,
    log_level="DEBUG"
)

# Or for async usage
await mcp.run_async(
    transport="http",
    host="127.0.0.1",
    port=8080
)
```

---

TITLE: Configuring Tag-Based Filtering for FastMCP Server in Python
DESCRIPTION: Demonstrates how to configure tag-based filtering when instantiating a FastMCP server, using `include_tags` to expose only specific components or `exclude_tags` to hide others.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_7

LANGUAGE: python
CODE:

```
# Only expose components tagged with "public"
mcp = FastMCP(include_tags={"public"})

# Hide components tagged as "internal" or "deprecated"
mcp = FastMCP(exclude_tags={"internal", "deprecated"})
```

---

TITLE: FastMCP Server with Main Execution Block
DESCRIPTION: This Python snippet demonstrates how to make a FastMCP server executable by adding a `__main__` block that calls `mcp.run()`. It includes examples of defining a tool (`add`) and resources (`get_config`, `personalized_greeting`) within the server, showcasing core FastMCP functionalities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="My First MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@mcp.resource("resource://config")
def get_config() -> dict:
    """Provides the application's configuration."""
    return {"version": "1.0", "author": "MyTeam"}

@mcp.resource("greetings://{name}")
def personalized_greeting(name: str) -> str:
    """Generates a personalized greeting for the given name."""
    return f"Hello, {name}! Welcome to the MCP server."

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Mounting FastMCP App in Starlette Application
DESCRIPTION: Illustrates how to mount a FastMCP ASGI application as a sub-application within an existing Starlette application using `starlette.routing.Mount`. It also highlights the importance of passing the FastMCP app's lifespan context to the main Starlette app.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

# Create your FastMCP server as well as any tools, resources, etc.
mcp = FastMCP("MyServer")

# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Create a Starlette app and mount the MCP server
app = Starlette(
    routes=[
        Mount("/mcp-server", app=mcp_app),
        # Add other routes as needed
    ],
    lifespan=mcp_app.lifespan,
)
```

---

TITLE: Run FastMCP Server with Inspector
DESCRIPTION: Runs a FastMCP server and integrates it with the MCP Inspector. This command allows specifying the Python file containing the server, installing dependencies in editable mode, adding extra packages, and configuring the Inspector's version and UI/server ports for development and debugging.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-cli-cli.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
dev(server_spec: str = typer.Argument(..., help='Python file to run, optionally with :object suffix'), with_editable: Annotated[Path | None, typer.Option('--with-editable', '-e', help='Directory containing pyproject.toml to install in editable mode', exists=True, file_okay=False, resolve_path=True)] = None, with_packages: Annotated[list[str], typer.Option('--with', help='Additional packages to install')] = [], inspector_version: Annotated[str | None, typer.Option('--inspector-version', help='Version of the MCP Inspector to use')] = None, ui_port: Annotated[int | None, typer.Option('--ui-port', help='Port for the MCP Inspector UI')] = None, server_port: Annotated[int | None, typer.Option('--server-port', help='Port for the MCP Inspector Proxy server')] = None) -> None
```

---

TITLE: Exclude Arguments from LLM Schema in FastMCP Python
DESCRIPTION: This section demonstrates how to prevent specific arguments from being exposed to the Large Language Model (LLM) or client in FastMCP. It's useful for injecting runtime values like user IDs or credentials. Only arguments with default values can be excluded.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_8

LANGUAGE: python
CODE:

```
@mcp.tool(
    name="get_user_details",
    exclude_args=["user_id"]
)
def get_user_details(user_id: str = None) -> str:
    # user_id will be injected by the server, not provided by the LLM
    ...
```

---

TITLE: Registering Prompts with Python Decorator
DESCRIPTION: This section details the `prompt` decorator for registering functions as prompts within the FastMCP framework. It explains how prompts can access a `Context` object for logging and session information, and demonstrates various decorator usage patterns including direct function calls, named prompts, and context-aware prompts.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_11

LANGUAGE: APIDOC
CODE:

```
prompt(self, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionPrompt] | FunctionPrompt
	name_or_fn: Either a function (when used as @prompt), a string name, or None
	name: Optional name for the prompt (keyword-only, alternative to name_or_fn)
	description: Optional description of what the prompt does
	tags: Optional set of tags for categorizing the prompt
	enabled: Optional boolean to enable or disable the prompt
```

LANGUAGE: python
CODE:

```
@server.prompt
def analyze_table(table_name: str) -> list\[Message]:
    schema = read_table_schema(table_name)
    return [
        {
            "role": "user",
            "content": f"Analyze this schema:\n{schema}"
        }
    ]
```

LANGUAGE: python
CODE:

```
@server.prompt()
def analyze_with_context(table_name: str, ctx: Context) -> list\[Message]:
    ctx.info(f"Analyzing table {table_name}")
    schema = read_table_schema(table_name)
    return [
        {
            "role": "user",
            "content": f"Analyze this schema:\n{schema}"
        }
    ]
```

LANGUAGE: python
CODE:

```
@server.prompt("custom_name")
def analyze_file(path: str) -> list\[Message]:
    content = await read_file(path)
    return [
        {
            "role": "user",
            "content": {
                "type": "resource",
                "resource": {
                    "uri": f"file://{path}",
                    "text": content
                }
            }
        }
    ]
```

LANGUAGE: python
CODE:

```
@server.prompt(name="custom_name")
def another_prompt(data: str) -> list\[Message]:
    return [{"role": "user", "content": data}]
```

LANGUAGE: python
CODE:

```
# Direct function call
server.prompt(my_function, name="custom_name")
```

---

TITLE: Mounting FastMCP in a Nested Starlette Application
DESCRIPTION: Demonstrates how to embed a FastMCP server within a nested Starlette application structure. It highlights the critical step of passing the FastMCP app's lifespan context to the outermost Starlette app to ensure proper session manager initialization for Streamable HTTP transport.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_7

LANGUAGE: python
CODE:

```
inner_app = Starlette(routes=[Mount("/inner", app=mcp_app)])
app = Starlette(
    routes=[Mount("/outer", app=inner_app)],
    lifespan=mcp_app.lifespan,
)
```

---

TITLE: Adding Custom Middleware to FastMCP ASGI App
DESCRIPTION: Demonstrates how to integrate custom Starlette middleware, such as `CORSMiddleware`, into a FastMCP ASGI application by passing a list of middleware instances during app creation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Create your FastMCP server
mcp = FastMCP("MyServer")

# Define custom middleware
custom_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["https://example.com", "https://app.example.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    ),
]

# Create ASGI app with custom middleware
http_app = mcp.http_app(middleware=custom_middleware)
```

---

TITLE: Deploy FastMCP Server Locally with ngrok
DESCRIPTION: These commands demonstrate how to deploy the FastMCP server locally and expose it to the internet using ngrok. The first command starts the Python server, and the second command creates a public tunnel to the server's port, making it accessible to Anthropic's MCP connector.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
python server.py
```

LANGUAGE: bash
CODE:

```
ngrok http 8000
```

---

TITLE: Install FastMCP Library
DESCRIPTION: Installs the FastMCP library using pip, which is a prerequisite for building an MCP server and utilizing its features.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_0

LANGUAGE: bash
CODE:

```
pip install fastmcp
```

---

TITLE: Registering Instance Methods During Class Initialization with FastMCP
DESCRIPTION: This pattern demonstrates how to automatically register instance methods when creating an object. The `__init__` method of `ComponentProvider` uses `mcp_instance.tool` and `mcp_instance.resource` to register its instance methods, encapsulating registration logic within the class.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class ComponentProvider:
    def __init__(self, mcp_instance):
        # Register methods
        mcp_instance.tool(self.tool_method)
        mcp_instance.resource("resource://data")(self.resource_method)

    def tool_method(self, x):
        return x * 2

    def resource_method(self):
        return "Resource data"

# The methods are automatically registered when creating the instance
provider = ComponentProvider(mcp)
```

---

TITLE: Define Optional and Required Tool Parameters
DESCRIPTION: FastMCP adheres to Python's standard conventions for function parameters. Parameters without a default value are considered required, while those with a default value (including `None` for `Optional` types) are optional.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_7

LANGUAGE: python
CODE:

```
@mcp.tool
def search_products(
    query: str,                   # Required - no default value
    max_results: int = 10,        # Optional - has default value
    sort_by: str = "relevance",   # Optional - has default value
    category: str | None = None   # Optional - can be None
) -> list[dict]:
    """Search the product catalog."""
    # Implementation...
```

---

TITLE: Creating FastMCP Server from FastAPI Application
DESCRIPTION: This class method facilitates the creation of a FastMCP server by integrating with an existing FastAPI application. It allows for custom naming, route mapping, and component handling, along with specific `httpx` client configurations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_17

LANGUAGE: python
CODE:

```
from_fastapi(cls, app: Any, name: str | None = None, route_maps: list[RouteMap] | None = None, route_map_fn: OpenAPIRouteMapFn | None = None, mcp_component_fn: OpenAPIComponentFn | None = None, mcp_names: dict[str, str] | None = None, httpx_client_kwargs: dict[str, Any] | None = None, tags: set[str] | None = None, **settings: Any) -> FastMCPOpenAPI
```

---

TITLE: Implement Custom Rate Limiting with FastMCP Middleware
DESCRIPTION: This Python snippet provides a basic implementation of a rate-limiting middleware using `defaultdict` to track client requests per minute. It demonstrates how to intercept requests and raise an `McpError` if the rate limit is exceeded.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_15

LANGUAGE: python
CODE:

```
import time
from collections import defaultdict
from fastmcp.server.middleware import Middleware, MiddlewareContext
from mcp import McpError
from mcp.types import ErrorData

class SimpleRateLimitMiddleware(Middleware):
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.client_requests = defaultdict(list)

    async def on_request(self, context: MiddlewareContext, call_next):
        current_time = time.time()
        client_id = "default"  # In practice, extract from headers or context

        # Clean old requests and check limit
        cutoff_time = current_time - 60
        self.client_requests[client_id] = [
            req_time for req_time in self.client_requests[client_id]
            if req_time > cutoff_time
        ]

        if len(self.client_requests[client_id]) >= self.requests_per_minute:
            raise McpError(ErrorData(code=-32000, message="Rate limit exceeded"))

        self.client_requests[client_id].append(current_time)
        return await call_next(context)
```

---

TITLE: Install FastMCP with uv
DESCRIPTION: Instructions for installing the FastMCP library using the uv package manager, which is the recommended method.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_2

LANGUAGE: bash
CODE:

```
uv pip install fastmcp
```

---

TITLE: Interact with FastMCP Server Tools via Claude Code
DESCRIPTION: An example demonstrating how Claude Code automatically discovers and uses tools provided by the connected FastMCP server based on user prompts. It also notes that resources can be referenced using `@` mentions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-code.mdx#_snippet_2

LANGUAGE: text
CODE:

```
Roll some dice for me
```

---

TITLE: Define LLM Sampling Handler in Python
DESCRIPTION: This Python snippet demonstrates how to define an asynchronous `sampling_handler` function and register it with the `fastmcp.Client`. This handler is responsible for processing LLM completion requests from MCP servers, taking `SamplingMessage`s, `SamplingParams`, and `RequestContext` as input and returning a generated string response.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/sampling.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.sampling import (
    SamplingMessage,
    SamplingParams,
    RequestContext,
)

async def sampling_handler(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext
) -> str:
    # Your LLM integration logic here
    # Extract text from messages and generate a response
    return "Generated response based on the messages"

client = Client(
    "my_mcp_server.py",
    sampling_handler=sampling_handler,
)
```

---

TITLE: Accessing FastMCP Tool Metadata in Middleware
DESCRIPTION: Demonstrates how to access tool metadata (like tags and enabled status) during the 'on_call_tool' hook using 'context.fastmcp_context.fastmcp.get_tool' to implement access control and prevent execution of private or disabled tools.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class TagBasedMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Access the tool object to check its metadata
        if context.fastmcp_context:
            try:
                tool = await context.fastmcp_context.fastmcp.get_tool(context.message.name)

                # Check if this tool has a "private" tag
                if "private" in tool.tags:
                    raise ToolError("Access denied: private tool")

                # Check if tool is enabled
                if not tool.enabled:
                    raise ToolError("Tool is currently disabled")

            except Exception:
                # Tool not found or other error - let execution continue
                # and handle the error naturally
                pass

        return await call_next(context)
```

---

TITLE: Transforming Tool Metadata with Tool.from_tool()
DESCRIPTION: Demonstrates how to create a new tool with a modified name and description from an existing generic tool using `Tool.from_tool()`, making it more domain-specific for an LLM client.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool

mcp = FastMCP()

# The original, generic tool
@mcp.tool
def search(query: str, category: str = "all") -> list[dict]:
    """Searches for items in the database."""
    return database.search(query, category)

# Create a more domain-specific version by changing its metadata
product_search_tool = Tool.from_tool(
    search,
    name="find_products",
    description="""
        Search for products in the e-commerce catalog.
        Use this when customers ask about finding specific items,
        checking availability, or browsing product categories.
        """,
)

mcp.add_tool(product_search_tool)
```

---

TITLE: Configure Advanced Rate Limiting Middleware in FastMCP
DESCRIPTION: This Python example shows how to apply FastMCP's built-in `RateLimitingMiddleware` (token bucket) and `SlidingWindowRateLimitingMiddleware` to control request rates, allowing for burst capacity or precise time-based limits.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_16

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware.rate_limiting import (
    RateLimitingMiddleware,
    SlidingWindowRateLimitingMiddleware
)

# Token bucket rate limiting (allows controlled bursts)
mcp.add_middleware(RateLimitingMiddleware(
    max_requests_per_second=10.0,
    burst_capacity=20
))

# Sliding window rate limiting (precise time-based control)
mcp.add_middleware(SlidingWindowRateLimitingMiddleware(
    max_requests=100,
    window_minutes=1
))
```

---

TITLE: Adding Descriptions to Tool Arguments with ArgTransform
DESCRIPTION: Demonstrates how to use `ArgTransform` to add a helpful description to a tool argument, improving LLM understanding. The example shows modifying the `user_id` argument of a `find_user` tool with a specific format description.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform

mcp = FastMCP()

@mcp.tool
def find_user(user_id: str):
    """Finds a user by their ID."""
    ...

new_tool = Tool.from_tool(
    find_user,
    transform_args={
        "user_id": ArgTransform(
            description=(
                "The unique identifier for the user, "
                "usually in the format 'usr-xxxxxxxx'."
            )
        )
    }
)
```

---

TITLE: Create a FastMCP Server with a Dice Rolling Tool (Python)
DESCRIPTION: This Python script initializes a FastMCP server named "Dice Roller" and defines a `roll_dice` tool that simulates rolling N 6-sided dice. The server is configured to run on HTTP transport at port 8000, making it ready for deployment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

---

TITLE: Hiding Tool Arguments with Constant Default Values using ArgTransform
DESCRIPTION: Explains how to hide tool arguments from the LLM using `hide=True` in `ArgTransform`, supplying a constant default value. This is useful for sensitive or internal parameters like API keys. The example hides an `api_key` argument, providing its value from an environment variable.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_7

LANGUAGE: python
CODE:

```
import os
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform

mcp = FastMCP()

@mcp.tool
def send_email(to: str, subject: str, body: str, api_key: str):
    """Sends an email."""
    ...

# Create a simplified version that hides the API key
new_tool = Tool.from_tool(
    send_email,
    name="send_notification",
    transform_args={
        "api_key": ArgTransform(
            hide=True,
            default=os.environ.get("EMAIL_API_KEY")
        )
    }
)
```

---

TITLE: Control Error Details with FastMCP ResourceError
DESCRIPTION: This example demonstrates how to use ResourceError to explicitly control error messages sent to clients, regardless of the mask_error_details setting. It also shows how other exceptions like ValueError are handled and potentially masked, and applies to template resources.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_16

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError

mcp = FastMCP(name="DataServer")

@mcp.resource("resource://safe-error")
def fail_with_details() -> str:
    """This resource provides detailed error information."""
    # ResourceError contents are always sent back to clients,
    # regardless of mask_error_details setting
    raise ResourceError("Unable to retrieve data: file not found")

@mcp.resource("resource://masked-error")
def fail_with_masked_details() -> str:
    """This resource masks internal error details when mask_error_details=True."""
    # This message would be masked if mask_error_details=True
    raise ValueError("Sensitive internal file path: /etc/secrets.conf")

@mcp.resource("data://{id}")
def get_data_by_id(id: str) -> dict:
    """Template resources also support the same error handling pattern."""
    if id == "secure":
        raise ValueError("Cannot access secure data")
    elif id == "missing":
        raise ResourceError("Data ID 'missing' not found in database")
    return {"id": id, "value": "data"}
```

---

TITLE: Utilize Python Collection Types for Complex Data in FastMCP
DESCRIPTION: This snippet showcases how FastMCP supports standard Python collection types like `list`, `dict`, `set`, and `tuple` for tool parameters. These types allow for the definition of complex data structures, including nested collections, with automatic parsing of matching JSON strings into Python objects.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_21

LANGUAGE: python
CODE:

```
@mcp.tool
def analyze_data(
    values: list[float],           # List of numbers
    properties: dict[str, str],    # Dictionary with string keys and values
    unique_ids: set[int],          # Set of unique integers
    coordinates: tuple[float, float],  # Tuple with fixed structure
    mixed_data: dict[str, list[int]] # Nested collections
):
    """Analyze collections of data."""
    # Implementation...
```

---

TITLE: Install OpenAI Python SDK (Bash)
DESCRIPTION: This command installs the necessary OpenAI Python SDK, which is required to interact with the OpenAI API and call the deployed FastMCP server. It ensures all dependencies are met for the subsequent Python code examples.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
pip install openai
```

---

TITLE: Exception-Based Error Handling for FastMCP Tools
DESCRIPTION: Demonstrates how to catch `ToolError` exceptions raised by `client.call_tool()` when a server-side tool execution fails, providing a robust error handling mechanism.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp.exceptions import ToolError

async with client:
    try:
        result = await client.call_tool("potentially_failing_tool", {"param": "value"})
        print("Tool succeeded:", result)
    except ToolError as e:
        print(f"Tool failed: {e}")
```

---

TITLE: Call FastMCP Server using OpenAI Responses API (Python)
DESCRIPTION: This Python code demonstrates how to use the OpenAI Python SDK to call the deployed FastMCP server. It configures the `responses.create` method with the server's URL and a prompt, then prints the AI's output, showcasing the integration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from openai import OpenAI

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "dice_server",
            "server_url": f"{url}/mcp/",
            "require_approval": "never",
        },
    ],
    input="Roll a few dice!",
)

print(resp.output_text)
```

---

TITLE: Running FastMCP ASGI App with Uvicorn (CLI)
DESCRIPTION: Provides the command-line instruction to run a FastMCP ASGI application using `uvicorn`, specifying the module path and host/port.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
uvicorn path.to.your.app:http_app --host 0.0.0.0 --port 8000
```

---

TITLE: Call FastMCP Server using Anthropic Messages API
DESCRIPTION: This Python script demonstrates how to interact with your deployed FastMCP server via the Anthropic Messages API. It uses the Anthropic SDK to create a message, specifying the FastMCP server URL and including the necessary `anthropic-beta` header for MCP client support. The response content, which includes the tool's output, is then printed.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_4

LANGUAGE: python
CODE:

```
import anthropic
from rich import print

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Roll a few dice!"}],
    mcp_servers=[
        {
            "type": "url",
            "url": "f\"{url}/mcp/\"",
            "name": "dice-server"
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)
```

---

TITLE: Configure FastMCP Client with Callback Handlers in Python
DESCRIPTION: This example demonstrates how to initialize a FastMCP client with custom callback handlers for logging, progress updates, and LLM sampling. It shows how to integrate advanced server interactions into the client configuration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.logging import LogMessage

async def log_handler(message: LogMessage):
    print(f"Server log: {message.data}")

async def progress_handler(progress: float, total: float | None, message: str | None):
    print(f"Progress: {progress}/{total} - {message}")

async def sampling_handler(messages, params, context):
    # Integrate with your LLM service here
    return "Generated response"

client = Client(
    "my_mcp_server.py",
    log_handler=log_handler,
    progress_handler=progress_handler,
    sampling_handler=sampling_handler,
    timeout=30.0
)
```

---

TITLE: Run a Local FastMCP Server
DESCRIPTION: Executes a FastMCP server defined in a local Python file directly using the `run` command. This command runs the server in your current Python environment, requiring you to manage dependencies.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp run server.py
```

---

TITLE: Running FastMCP ASGI App with Uvicorn (Python)
DESCRIPTION: Shows how to programmatically run a FastMCP ASGI application using the `uvicorn` server, binding it to a host and port.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
import uvicorn

mcp = FastMCP("MyServer")

http_app = mcp.http_app()

if __name__ == "__main__":
    uvicorn.run(http_app, host="0.0.0.0", port=8000)
```

---

TITLE: Create a Simple FastMCP Dice Roller Server in Python
DESCRIPTION: This Python code defines a basic FastMCP server named 'Dice Roller' with a `roll_dice` tool. It demonstrates initializing FastMCP and exposing a function to simulate rolling 6-sided dice, returning the results.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Testing FastMCP Server with an Asynchronous Client in Python
DESCRIPTION: Illustrates how to create a FastMCP client, connect it to the server object, and asynchronously call a registered tool. It highlights the use of `asyncio.run` and client context management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_2

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import FastMCP, Client

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

client = Client(mcp)

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))
```

---

TITLE: Retrieve and Render FastMCP Server Prompts in Python
DESCRIPTION: This example demonstrates how to list available prompts on a FastMCP server and retrieve a rendered message template. It shows how to pass arguments to prompts and access the generated messages.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_7

LANGUAGE: python
CODE:

```
async with client:
    # List available prompts
    prompts = await client.list_prompts()

    # Get a rendered prompt
    messages = await client.get_prompt("analyze_data", {"data": [1, 2, 3]})
    print(messages.messages)
```

---

TITLE: Configure FastMCP to Mask Error Details
DESCRIPTION: This snippet shows how to initialize a FastMCP instance with mask_error_details=True to prevent sensitive internal error details from being sent to client LLMs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_15

LANGUAGE: python
CODE:

```
mcp = FastMCP(name="SecureServer", mask_error_details=True)
```

---

TITLE: FastMCP Client with Multi-Server Configuration
DESCRIPTION: Shows how to configure a `fastmcp.Client` to connect to multiple MCP servers (e.g., a remote URL and a local script) using a standard MCP configuration dictionary. This allows the client to access tools and resources from different servers with server prefixes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_10

LANGUAGE: python
CODE:

```
from fastmcp import Client

# Standard MCP configuration with multiple servers
config = {
    "mcpServers": {
        "weather": {"url": "https://weather-api.example.com/mcp"},
        "assistant": {"command": "python", "args": ["./assistant_server.py"]}
    }
}

# Create a client that connects to all servers
client = Client(config)

async def main():
    async with client:
        # Access tools and resources with server prefixes
        forecast = await client.call_tool("weather_get_forecast", {"city": "London"})
        answer = await client.call_tool("assistant_answer_question", {"query": "What is MCP?"})
```

---

TITLE: Progress Reporting in FastMCP Tools with Python
DESCRIPTION: This section explains how to provide real-time progress updates from long-running FastMCP tool functions to the client. It illustrates the use of `ctx.report_progress` to notify clients about task completion status, improving user experience for asynchronous operations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_5

LANGUAGE: python
CODE:

```
@mcp.tool
async def process_items(items: list[str], ctx: Context) -> dict:
    """Process a list of items with progress updates."""
    total = len(items)
    results = []

    for i, item in enumerate(items):
        # Report progress as percentage
        await ctx.report_progress(progress=i, total=total)

        # Process the item (simulated with a sleep)
        await asyncio.sleep(0.1)
        results.append(item.upper())

    # Report 100% completion
    await ctx.report_progress(progress=total, total=total)

    return {"processed": len(results), "results": results}
```

LANGUAGE: APIDOC
CODE:

```
Method signature:
- ctx.report_progress(progress: float, total: float | None = None)
  - progress: Current progress value (e.g., 24)
  - total: Optional total value (e.g., 100). If provided, clients may interpret this as a percentage.
```

---

TITLE: FastMCP Client Multi-Server Interaction Example
DESCRIPTION: This snippet demonstrates how to use a FastMCP client configured with multiple servers, such as a "weather" API and an "assistant" server. It shows how to call tools and read resources, emphasizing that tool names and resource URIs are prefixed with the server names (e.g., `weather_get_forecast`, `resource://assistant/templates/list`) when interacting with a multi-server client.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_3

LANGUAGE: python
CODE:

```
config = {
    "mcpServers": {
        "weather": {"url": "https://weather-api.example.com/mcp"},
        "assistant": {"command": "python", "args": ["./assistant_server.py"]}
    }
}

client = Client(config)

async with client:
    # Tools are prefixed with server names
    weather_data = await client.call_tool("weather_get_forecast", {"city": "London"})
    response = await client.call_tool("assistant_answer_question", {"question": "What's the capital of France?"})

    # Resources use prefixed URIs
    icons = await client.read_resource("weather://weather/icons/sunny")
    templates = await client.read_resource("resource://assistant/templates/list")
```

---

TITLE: Accessing Multiple MCP Servers with MCPConfigTransport
DESCRIPTION: This example demonstrates how to initialize a `fastmcp.Client` using an `MCPConfig` that defines multiple MCP servers. It shows how to access tools using the `{server_name}_{tool_name}` prefix and read resources using `protocol://{server_name}/path/to/resource` patterns, simplifying interaction with diverse services through a single client interface.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-transports.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.utilities.mcp_config import MCPConfig

# Create a config with multiple servers
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "http"
        },
        "calendar": {
            "url": "https://calendar-api.example.com/mcp",
            "transport": "http"
        }
    }
}

# Create a client with the config
client = Client(config)

async with client:
    # Access tools with prefixes
    weather = await client.call_tool("weather_get_forecast", {"city": "London"})
    events = await client.call_tool("calendar_list_events", {"date": "2023-06-01"})

    # Access resources with prefixed URIs
    icons = await client.read_resource("weather://weather/icons/sunny")
```

---

TITLE: Expose Read-Only Data as a Resource
DESCRIPTION: Illustrates how to expose read-only data to an LLM by decorating a function with `@mcp.resource` and providing a unique URI. The function is executed only when the resource is requested, enabling lazy-loading of data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="My First MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@mcp.resource("resource://config")
def get_config() -> dict:
    """Provides the application's configuration."""
    return {"version": "1.0", "author": "MyTeam"}
```

---

TITLE: Streamable HTTP Transport: Authentication with Headers
DESCRIPTION: Illustrates how to include custom HTTP headers, such as an `Authorization` header, when instantiating `StreamableHttpTransport` to authenticate requests to a FastMCP server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Create transport with authentication headers
transport = StreamableHttpTransport(
    url="https://example.com/mcp",
    headers={"Authorization": "Bearer your-token-here"}
)

client = Client(transport)
```

---

TITLE: Recommended Pattern for Registering Class Methods with FastMCP
DESCRIPTION: This example demonstrates the correct method for registering class methods with FastMCP. The `@classmethod` decorator is applied normally during class definition. After the class is defined, the class method is registered using `mcp.tool(MyClass.from_string)`. This approach ensures that Python properly handles the `cls` parameter internally, exposing only the relevant parameters to the LLM and preventing unexpected behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class MyClass:
    @classmethod
    def from_string(cls, s):
        return cls(s)

# Register the class method after the class is defined
mcp.tool(MyClass.from_string)
```

---

TITLE: FastMCP Streamable HTTP: Custom Server and Client Configuration
DESCRIPTION: This example illustrates how to customize the host, port, path, and log level when running a FastMCP server with Streamable HTTP. It includes the corresponding client setup to connect to the server using the specified custom URL.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=4200,
        path="/my-custom-path",
        log_level="debug",
    )
```

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:4200/my-custom-path") as client:
        await client.ping()

if __name__ == "__main__":
    asyncio.run(example())
```

---

TITLE: Authenticate FastMCP Client with String Bearer Token
DESCRIPTION: Shows the most straightforward way to use a pre-existing Bearer token by providing it as a string to the `auth` parameter of the `fastmcp.Client` instance. FastMCP automatically formats it correctly for the `Authorization` header.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/bearer.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import Client

async with Client(
    "https://fastmcp.cloud/mcp",
    auth="<your-token>",
) as client:
    await client.ping()
```

---

TITLE: Defining a Reusable Prompt with FastMCP in Python
DESCRIPTION: Explains how to define a reusable message template for guiding LLMs using the `@mcp.prompt` decorator, allowing for dynamic content generation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_5

LANGUAGE: python
CODE:

```
@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"
```

---

TITLE: Converting FastAPI Application to FastMCP Server
DESCRIPTION: Illustrates how to convert an existing FastAPI application into a FastMCP server using `FastMCP.from_fastapi()`. This allows reusing FastAPI endpoints and their definitions directly.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_16

LANGUAGE: python
CODE:

```
from fastapi import FastAPI
from fastmcp import FastMCP

# Your FastAPI app
app = FastAPI(title="My API", version="1.0.0")

@app.get("/items", tags=["items"], operation_id="list_items")
def list_items():
    return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

@app.get("/items/{item_id}", tags=["items", "detail"], operation_id="get_item")
def get_item(item_id: int):
    return {"id": item_id, "name": f"Item {item_id}"}

@app.post("/items", tags=["items", "create"], operation_id="create_item")
def create_item(name: str):
    return {"id": 3, "name": name}

# Convert FastAPI app to MCP server
mcp = FastMCP.from_fastapi(app=app)

if __name__ == "__main__":
    mcp.run()  # Run as MCP server
```

---

TITLE: Generate Python Code Example with LLM
DESCRIPTION: Illustrates generating a Python code example using the client's LLM with a system prompt and user message. It sets temperature and max tokens for controlled generation, returning the generated code formatted as a Markdown code block.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_9

LANGUAGE: python
CODE:

````
@mcp.tool
async def generate_example(concept: str, ctx: Context) -> str:
    """Generate a Python code example for a given concept."""
    # Using a system prompt and a user message
    response = await ctx.sample(
        messages=f"Write a simple Python code example demonstrating '{concept}'.",
        system_prompt="You are an expert Python programmer. Provide concise, working code examples without explanations.",
        temperature=0.7,
        max_tokens=300
    )

    code_example = response.text
    return f"```python\n{code_example}\n```"
````

---

TITLE: Analyze Sentiment with Client LLM
DESCRIPTION: Demonstrates how to use the client's LLM to analyze text sentiment. It constructs a prompt, sends a sampling request via `ctx.sample`, and processes the LLM's response to categorize sentiment as positive, negative, or neutral.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_7

LANGUAGE: python
CODE:

```
@mcp.tool
async def analyze_sentiment(text: str, ctx: Context) -> dict:
    """Analyze the sentiment of a text using the client's LLM."""
    # Create a sampling prompt asking for sentiment analysis
    prompt = f"Analyze the sentiment of the following text as positive, negative, or neutral. Just output a single word - 'positive', 'negative', or 'neutral'. Text to analyze: {text}"

    # Send the sampling request to the client's LLM (provide a hint for the model you want to use)
    response = await ctx.sample(prompt, model_preferences="claude-3-sonnet")

    # Process the LLM's response
    sentiment = response.text.strip().lower()

    # Map to standard sentiment values
    if "positive" in sentiment:
        sentiment = "positive"
    elif "negative" in sentiment:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"text": text, "sentiment": sentiment}
```

---

TITLE: Create a Dynamic Resource Template
DESCRIPTION: Shows how to generate dynamic content by defining a resource template using `@mcp.resource` with placeholders in the URI. FastMCP automatically maps URI segments to function parameters, allowing clients to request personalized or parameterized resources.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/create-mcp-server.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="My First MCP Server")

@mcp.tool
def add(a: int, b: int) -> int:
    """Adds two integer numbers together."""
    return a + b

@mcp.resource("resource://config")
def get_config() -> dict:
    """Provides the application's configuration."""
    return {"version": "1.0", "author": "MyTeam"}

@mcp.resource("greetings://{name}")
def personalized_greeting(name: str) -> str:
    """Generates a personalized greeting for the given name."""
    return f"Hello, {name}! Welcome to the MCP server."
```

---

TITLE: List available prompt templates (Python)
DESCRIPTION: Demonstrates how to retrieve all prompt templates from the FastMCP server using `client.list_prompts()`. It iterates through the results to print prompt names, descriptions, and arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_0

LANGUAGE: python
CODE:

```
async with client:
    prompts = await client.list_prompts()
    # prompts -> list[mcp.types.Prompt]

    for prompt in prompts:
        print(f"Prompt: {prompt.name}")
        print(f"Description: {prompt.description}")
        if prompt.arguments:
            print(f"Arguments: {[arg.name for arg in prompt.arguments]}")
```

---

TITLE: Access Token Claims in FastMCP Tools (Python)
DESCRIPTION: This snippet demonstrates how to access token information within a FastMCP tool using the `get_access_token()` dependency function. It shows how to retrieve the authenticated principal identifier (client_id) and granted scopes, and how to perform scope-based authorization checks.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context, ToolError
from fastmcp.server.dependencies import get_access_token, AccessToken

@mcp.tool
async def get_my_data(ctx: Context) -> dict:
    access_token: AccessToken = get_access_token()

    user_id = access_token.client_id  # From JWT 'sub' or 'client_id' claim
    user_scopes = access_token.scopes

    if "data:read_sensitive" not in user_scopes:
        raise ToolError("Insufficient permissions: 'data:read_sensitive' scope required.")

    return {
        "user": user_id,
        "sensitive_data": f"Private data for {user_id}",
        "granted_scopes": user_scopes
    }
```

---

TITLE: Resource Access in FastMCP Tools with Python
DESCRIPTION: This section describes how FastMCP tool functions can read data from resources registered with the FastMCP server. It demonstrates using `ctx.read_resource` to fetch content based on a URI, enabling functions to access files or configuration dynamically.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_6

LANGUAGE: python
CODE:

```
@mcp.tool
async def summarize_document(document_uri: str, ctx: Context) -> str:
    """Summarize a document by its resource URI."""
    # Read the document content
    content_list = await ctx.read_resource(document_uri)

    if not content_list:
        return "Document is empty"

    document_text = content_list[0].content

    # Example: Generate a simple summary (length-based)
    words = document_text.split()
    total_words = len(words)

    await ctx.info(f"Document has {total_words} words")

    # Return a simple summary
    if total_words > 100:
        summary = " ".join(words[:100]) + "..."
        return f"Summary ({total_words} words total): {summary}"
    else:
        return f"Full document ({total_words} words): {document_text}"
```

LANGUAGE: APIDOC
CODE:

```
Method signature:
- ctx.read_resource(uri: str | AnyUrl) -> list[ReadResourceContents]
  - uri: The resource URI to read
  - Returns a list of resource content parts (usually containing just one item)
```

---

TITLE: FastMCP Prompt Returning a List of Messages
DESCRIPTION: This example showcases a FastMCP prompt function designed to return a `list[Message]`. This pattern is useful for setting up multi-turn conversations or providing a sequence of initial responses, leveraging the `Message` class from `fastmcp.prompts.prompt`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp.prompts.prompt import Message

@mcp.prompt
def roleplay_scenario(character: str, situation: str) -> list[Message]:
    """Sets up a roleplaying scenario with initial messages."""
    return [
        Message(f"Let's roleplay. You are {character}. The situation is: {situation}"),
        Message("Okay, I understand. I am ready. What happens next?", role="assistant")
    ]
```

---

TITLE: Passing Arguments to FastMCP Tools
DESCRIPTION: Illustrates how to pass simple and complex dictionary arguments to server-side tools using `client.call_tool()`, supporting nested structures for configuration and data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_7

LANGUAGE: python
CODE:

```
async with client:
    # Simple arguments
    result = await client.call_tool("greet", {"name": "World"})

    # Complex arguments
    result = await client.call_tool("process_data", {
        "config": {"format": "json", "validate": True},
        "items": [1, 2, 3, 4, 5],
        "metadata": {"source": "api", "version": "1.0"}
    })
```

---

TITLE: Running FastMCP Server via CLI with Transport Options
DESCRIPTION: This command demonstrates how to specify transport options and other configurations, such as the port, when running a FastMCP server using the `fastmcp` CLI. This allows overriding transport settings that might be specified within the server's Python code.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp run server.py --transport sse --port 9000
```

---

TITLE: Define Static and Dynamic FastMCP Resources
DESCRIPTION: Illustrates how to create read-only data sources using `@mcp.resource`. The first example shows a static resource, while the second demonstrates a dynamic resource template with placeholders for parameters, allowing clients to request specific data subsets.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_5

LANGUAGE: python
CODE:

```
# Static resource
@mcp.resource("config://version")
def get_version():
    return "2.0.1"
```

LANGUAGE: python
CODE:

```
# Dynamic resource template
@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    # Fetch profile for user_id...
    return {"name": f"User {user_id}", "status": "active"}
```

---

TITLE: Configure FastMCP for Masked Error Details
DESCRIPTION: Demonstrates how to initialize a FastMCP instance with 'mask_error_details=True' to prevent internal error details from being sent to client LLMs, enhancing security.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_13

LANGUAGE: python
CODE:

```
mcp = FastMCP(name="SecureServer", mask_error_details=True)
```

---

TITLE: Defining Basic Dynamic Resources with @mcp.resource
DESCRIPTION: Demonstrates how to use the `@mcp.resource` decorator to define simple dynamic resources in FastMCP. It shows examples of returning a plain string and a dictionary (which is auto-serialized to JSON), illustrating the basic setup for exposing data via URIs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import json
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

# Basic dynamic resource returning a string
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    """Provides a simple greeting message."""
    return "Hello from FastMCP Resources!"

# Resource returning JSON data (dict is auto-serialized)
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides application configuration as JSON."""
    return {
        "theme": "dark",
        "version": "1.2.0",
        "features": ["tools", "resources"],
    }
```

---

TITLE: Get a prompt with arguments (Python)
DESCRIPTION: Illustrates how to pass a dictionary of arguments to `client.get_prompt()` to customize the prompt's output. The example retrieves and prints personalized messages based on the provided arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_2

LANGUAGE: python
CODE:

```
async with client:
    # Prompt with simple arguments
    result = await client.get_prompt("user_greeting", {
        "name": "Alice",
        "role": "administrator"
    })

    # Access the personalized messages
    for message in result.messages:
        print(f"Generated message: {message.content}")
```

---

TITLE: Registering Static Methods with FastMCP (Recommended Pattern)
DESCRIPTION: This preferred pattern shows how to register a static method with FastMCP by first defining it as a static method and then explicitly registering it using `mcp.tool(MyClass.utility)`. This ensures proper method binding and is the recommended approach.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class MyClass:
    @staticmethod
    def utility(x, y):
        return x + y

# This also works
mcp.tool(MyClass.utility)
```

---

TITLE: Configure FastMCP Client with Default OAuth
DESCRIPTION: Demonstrates the simplest way to enable OAuth authentication for a FastMCP client by passing the string 'oauth' to the `auth` parameter. This uses FastMCP's default OAuth settings, simplifying initial setup for common use cases.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import Client

# Uses default OAuth settings
async with Client("https://fastmcp.cloud/mcp", auth="oauth") as client:
    await client.ping()
```

---

TITLE: Install FastMCP Server in Claude Desktop App
DESCRIPTION: Installs a FastMCP server for integration with the Claude desktop application. This command allows for custom server naming, installing dependencies in editable mode, including additional packages, and managing environment variables either directly or by loading them from a `.env` file. Environment variables are preserved across installations unless explicitly updated.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-cli-cli.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
install(server_spec: str = typer.Argument(..., help='Python file to run, optionally with :object suffix'), server_name: Annotated[str | None, typer.Option('--name', '-n', help="Custom name for the server (defaults to server's name attribute or file name)")] = None, with_editable: Annotated[Path | None, typer.Option('--with-editable', '-e', help='Directory containing pyproject.toml to install in editable mode', exists=True, file_okay=False, resolve_path=True)] = None, with_packages: Annotated[list[str], typer.Option('--with', help='Additional packages to install')] = [], env_vars: Annotated[list[str], typer.Option('--env-var', '-v', help='Environment variables in KEY=VALUE format')] = [], env_file: Annotated[Path | None, typer.Option('--env-file', '-f', help='Load environment variables from a .env file', exists=True, file_okay=True, dir_okay=False, resolve_path=True)] = None) -> None
```

---

TITLE: Mandatory Development Workflow for FastMCP
DESCRIPTION: Outlines the essential commands for installing dependencies, running pre-commit hooks for linting and type-checking, and executing the full test suite. All tests must pass, and linting/typing must be clean before committing.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/AGENTS.md#_snippet_0

LANGUAGE: bash
CODE:

```
uv sync                              # install dependencies
uv run pre-commit run --all-files    # Ruff + Prettier + Pyright
uv run pytest                        # run full test suite
```

---

TITLE: Manage FastMCP Client Connection Lifecycle with Python Context Manager
DESCRIPTION: This snippet demonstrates how to establish and automatically close a connection to a FastMCP server using an asynchronous context manager. It shows making multiple calls within a single session and verifying connection status.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_4

LANGUAGE: python
CODE:

```
async def example():
    client = Client("my_mcp_server.py")

    # Connection established here
    async with client:
        print(f"Connected: {client.is_connected()}")

        # Make multiple calls within the same session
        tools = await client.list_tools()
        result = await client.call_tool("greet", {"name": "World"})

    # Connection closed automatically here
    print(f"Connected: {client.is_connected()}")
```

---

TITLE: FastMCP Pydantic Field Validation as Default Values
DESCRIPTION: Illustrates an alternative method for applying Pydantic `Field` validation by using `Field` directly as a default value for parameters in FastMCP tools. This approach supports value, string, and collection constraints.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_31

LANGUAGE: python
CODE:

```
@mcp.tool
def validate_data(
    # Value constraints
    age: int = Field(ge=0, lt=120),                     # 0 <= age < 120

    # String constraints
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"),  # Email pattern

    # Collection constraints
    tags: list[str] = Field(min_length=1, max_length=10)  # 1-10 tags
):
    """Process data with field validations."""
    # Implementation...
```

---

TITLE: Mounting FastMCP Server Instances
DESCRIPTION: The `mount` method dynamically connects another FastMCP server to the current server. It forwards client requests in real-time, ensuring immediate reflection of changes in the mounted server. It supports optional prefixing for tools, resources, templates, and prompts, and operates in two modes: direct in-memory access (default for servers without custom lifespans) or proxy-based communication (default for servers with custom lifespans) to preserve client-facing behaviors.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_15

LANGUAGE: python
CODE:

```
mount(self, server: FastMCP[LifespanResultT], prefix: str | None = None, as_proxy: bool | None = None) -> None
```

LANGUAGE: APIDOC
CODE:

```
mount(self, server: FastMCP[LifespanResultT], prefix: str | None = None, as_proxy: bool | None = None) -> None
  Args:
    server: The FastMCP server to mount.
    prefix: Optional prefix to use for the mounted server's objects. If None, the server's objects are accessible with their original names.
    as_proxy: Whether to treat the mounted server as a proxy. If None (default), automatically determined based on whether the server has a custom lifespan (True if it has a custom lifespan, False otherwise).
    tool_separator: Deprecated. Separator character for tool names.
    resource_separator: Deprecated. Separator character for resource URIs.
    prompt_separator: Deprecated. Separator character for prompt names.
```

---

TITLE: Configuring Direct vs. Proxy Mounting in FastMCP
DESCRIPTION: This Python snippet demonstrates the syntax for configuring direct and proxy mounting modes in FastMCP. It shows how to use the `as_proxy=True` parameter for proxy mounting and how to mount without a prefix, explaining the implications for client lifecycle and communication.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_3

LANGUAGE: python
CODE:

```
# Direct mounting (default when no custom lifespan)
main_mcp.mount(api_server, prefix="api")

# Proxy mounting (preserves full client lifecycle)
main_mcp.mount(api_server, prefix="api", as_proxy=True)

# Mounting without a prefix (components accessible without prefixing)
main_mcp.mount(api_server)
```

---

TITLE: Read Static Resources with Python MCP Client
DESCRIPTION: Illustrates how to read content from a static resource using its URI with `client.read_resource()`. It demonstrates accessing both text and binary content from the returned `TextResourceContents` or `BlobResourceContents` objects.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_2

LANGUAGE: python
CODE:

```
async with client:
    # Read a static resource
    content = await client.read_resource("file:///path/to/README.md")
    # content -> list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]

    # Access text content
    if hasattr(content[0], 'text'):
        print(content[0].text)

    # Access binary content
    if hasattr(content[0], 'blob'):
        print(f"Binary data: {len(content[0].blob)} bytes")
```

---

TITLE: Setting up FastMCP Development Environment on Windows
DESCRIPTION: This snippet provides the necessary `uv` commands to set up a FastMCP development environment specifically for Windows. It covers creating a virtual environment, activating it, and installing the FastMCP package in editable mode with development dependencies.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/Windows_Notes.md#_snippet_0

LANGUAGE: Bash
CODE:

```
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

---

TITLE: Deploy FastMCP Server and Expose with ngrok (Bash)
DESCRIPTION: These commands demonstrate how to run the FastMCP server locally and then expose it to the internet using `ngrok`. The first command starts the Python server, and the second creates a public tunnel to port 8000, making the local server accessible to OpenAI.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
python server.py
```

LANGUAGE: bash
CODE:

```
ngrok http 8000
```

---

TITLE: Define Synchronous and Asynchronous FastMCP Tools in Python
DESCRIPTION: This section demonstrates how FastMCP supports both standard synchronous (`def`) and asynchronous (`async def`) Python functions as tools. Asynchronous tools are ideal for I/O-bound operations, preventing server blocking during external calls.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_11

LANGUAGE: python
CODE:

```
# Synchronous tool (suitable for CPU-bound or quick tasks)
@mcp.tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two coordinates."""
    # Implementation...
    return 42.5

# Asynchronous tool (ideal for I/O-bound operations)
@mcp.tool
async def fetch_weather(city: str) -> dict:
    """Retrieve current weather conditions for a city."""
    # Use 'async def' for operations involving network calls, file I/O, etc.
    # This prevents blocking the server while waiting for external operations.
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/weather/{city}") as response:
            # Check response status before returning
            response.raise_for_status()
            return await response.json()
```

---

TITLE: Request Context Middleware (Python)
DESCRIPTION: A middleware component designed to store the incoming HTTP request within a `ContextVar`, making it easily accessible throughout the request's lifecycle within the application.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-http.mdx#_snippet_6

LANGUAGE: APIDOC
CODE:

```
RequestContextMiddleware
  Middleware that stores each request in a ContextVar.
```

---

TITLE: Incorrect Direct Decoration of Class Methods: Decorator Order Matters
DESCRIPTION: This snippet highlights the pitfalls of directly decorating class methods with FastMCP, emphasizing that decorator order is crucial. If `@classmethod` is applied before `@mcp.tool`, no error is raised, but the method won't function correctly. Conversely, if `@mcp.tool` is applied first, FastMCP will detect the issue and raise a helpful `ValueError`, preventing silent failures. Both scenarios expose `cls` to the LLM, leading to incorrect behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class MyClass:
    @classmethod
    @mcp.tool  # This won't work but won't raise an error
    def from_string_v1(cls, s):
        return cls(s)

    @mcp.tool
    @classmethod  # This will raise a helpful ValueError
    def from_string_v2(cls, s):
        return cls(s)
```

---

TITLE: Create a FastMCP Server with Python
DESCRIPTION: This example demonstrates how to create a local HTTP FastMCP server using Python for development purposes. It defines a 'Dice Roller' server with a `roll_dice` tool that simulates rolling 6-sided dice.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-code.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

---

TITLE: Specify Dependencies for FastMCP Server Installation
DESCRIPTION: Demonstrates two ways to include dependencies for a FastMCP server. Dependencies can be specified via the `--with` flag during CLI installation or directly within the `FastMCP` constructor in Python code.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp install server.py --with pandas --with requests
```

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(
    name="Dice Roller",
    dependencies=["pandas", "requests"]
)
```

---

TITLE: Setup Authentication Middleware and Routes (Python)
DESCRIPTION: Configures authentication middleware and defines routes for an application based on the provided OAuth provider. This function is crucial for securing API endpoints by integrating an authorization server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-http.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
setup_auth_middleware_and_routes(auth: OAuthProvider) -> tuple[list[Middleware], list[BaseRoute], list[str]]
  auth: The OAuthProvider authorization server provider.
  Returns: A tuple containing a list of middleware, a list of authentication-related routes, and a list of required scopes.
```

---

TITLE: Run FastMCP Server Asynchronously with run_async()
DESCRIPTION: This snippet demonstrates how to run a FastMCP server in an asynchronous context using the `run_async()` method. It highlights the importance of using `run_async()` within async functions to avoid event loop errors, contrasting it with the synchronous `run()` method. Both methods accept the same transport arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_10

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
import asyncio

mcp = FastMCP(name="MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

async def main():
    # Use run_async() in async contexts
    await mcp.run_async(transport="http")

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: Python: Run FastMCP Server in Separate Process
DESCRIPTION: This context manager executes a FastMCP server in an isolated process and provides its URL. Upon exiting the context, the server process is automatically terminated. It requires a function to create and run the server, as FastMCP servers are not pickleable. It can also accept arguments to pass to the server function, including an option to provide host and port as keyword arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-utilities-tests.mdx#_snippet_1

LANGUAGE: python
CODE:

```
run_server_in_process(server_fn: Callable[..., None], *args, **kwargs) -> Generator[str, None, None]
```

LANGUAGE: APIDOC
CODE:

```
Context manager that runs a FastMCP server in a separate process and
returns the server URL. When the context manager is exited, the server process is killed.

Args:
- server_fn: The function that runs a FastMCP server. FastMCP servers are
not pickleable, so we need a function that creates and runs one.
- *args: Arguments to pass to the server function.
- provide_host_and_port: Whether to provide the host and port to the server function as kwargs.
- **kwargs: Keyword arguments to pass to the server function.

Returns:
- The server URL.
```

---

TITLE: Importing FastMCP Servers for Static Composition
DESCRIPTION: This Python example demonstrates how to use `FastMCP.import_server()` to statically compose a `WeatherService` subserver into a `MainApp` server. It illustrates the definition of tools and resources within a subserver and their subsequent import into a main server, showing how components are copied and optionally prefixed (e.g., 'weather_get_forecast', 'data://weather/cities/supported'). This method performs a one-time copy, meaning changes to the subserver after import are not reflected.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
import asyncio

# Define subservers
weather_mcp = FastMCP(name="WeatherService")

@weather_mcp.tool
def get_forecast(city: str) -> dict:
    """Get weather forecast."""
    return {"city": city, "forecast": "Sunny"}

@weather_mcp.resource("data://cities/supported")
def list_supported_cities() -> list[str]:
    """List cities with weather support."""
    return ["London", "Paris", "Tokyo"]

# Define main server
main_mcp = FastMCP(name="MainApp")

# Import subserver
async def setup():
    await main_mcp.import_server(weather_mcp, prefix="weather")

# Result: main_mcp now contains prefixed components:
# - Tool: "weather_get_forecast"
# - Resource: "data://weather/cities/supported"

if __name__ == "__main__":
    asyncio.run(setup())
    main_mcp.run()
```

---

TITLE: Pydantic Field Common Validation Options
DESCRIPTION: Lists common validation options available with Pydantic's `Field` class, including numeric constraints (`ge`, `gt`, `le`, `lt`, `multiple_of`), string/collection length constraints (`min_length`, `max_length`), pattern matching (`pattern`), and adding descriptions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_32

LANGUAGE: APIDOC
CODE:

```
Validation Options for Pydantic Field:
- ge (Number): Greater than or equal constraint.
- gt (Number): Greater than constraint.
- le (Number): Less than or equal constraint.
- lt (Number): Less than constraint.
- multiple_of (Number): Value must be a multiple of this number.
- min_length (String, List, etc.): Minimum length constraint.
- max_length (String, List, etc.): Maximum length constraint.
- pattern (String): Regular expression pattern constraint.
- description (Any): Human-readable description (appears in schema).
```

---

TITLE: Filtering FastMCP Listing Results in Middleware
DESCRIPTION: Demonstrates how to inspect and modify listing results (e.g., 'on_list_tools') to filter components based on their properties before they are converted to MCP format and returned to the client. This example filters out tools with a 'private' tag.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext, ListToolsResult

class ListingFilterMiddleware(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        result = await call_next(context)

        # Filter out tools with "private" tag
        filtered_tools = {
            name: tool for name, tool in result.tools.items()
            if "private" not in tool.tags
        }

        # Return modified result
        return ListToolsResult(tools=filtered_tools)
```

---

TITLE: MCP Client Call Example with String Arguments
DESCRIPTION: This JSON snippet illustrates how an MCP client would invoke the `analyze_data` prompt. As per the MCP specification, all arguments are passed as JSON-formatted strings, even for complex types like lists and dictionaries.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_4

LANGUAGE: json
CODE:

```
{
  "numbers": "[1, 2, 3, 4, 5]",
  "metadata": "{\"source\": \"api\", \"version\": \"1.0\"}",
  "threshold": "2.5"
}
```

---

TITLE: Accessing FastMCP Context via Dependency Function
DESCRIPTION: This Python code shows how to retrieve the active `Context` object from anywhere within a FastMCP server request's execution flow using the `get_context()` dependency function. This is useful for utility functions that don't directly receive `Context` as a parameter but need to interact with MCP capabilities. Note that `get_context` only works within a server request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context

mcp = FastMCP(name="DependencyDemo")

# Utility function that needs context but doesn't receive it as a parameter
async def process_data(data: list[float]) -> dict:
    # Get the active context - only works when called within a request
    ctx = get_context()
    await ctx.info(f"Processing {len(data)} data points")

@mcp.tool
async def analyze_dataset(dataset_name: str) -> dict:
    # Call utility function that uses context internally
    data = load_data(dataset_name)
    await process_data(data)
```

---

TITLE: Implement Basic Request Logging with FastMCP Middleware
DESCRIPTION: This Python snippet demonstrates how to create a custom `Middleware` class to log incoming messages and their processing status (success or failure) within a FastMCP server. It shows the basic structure for intercepting requests and responses.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_13

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext

class SimpleLoggingMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        print(f"Processing {context.method} from {context.source}")

        try:
            result = await call_next(context)
            print(f"Completed {context.method}")
            return result
        except Exception as e:
            print(f"Failed {context.method}: {e}")
            raise
```

---

TITLE: FastMCP `run` Command Server Specification
DESCRIPTION: Describes the three ways to specify a server when using the `fastmcp run` command: by local Python file, local file with a custom object name, or a remote HTTP/HTTPS URL.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_4

LANGUAGE: APIDOC
CODE:

```
Server Specification:
1. server.py - imports the module and looks for a FastMCP object named mcp, server, or app. Errors if no such object is found.
2. server.py:custom_name - imports and uses the specified server object
3. http://server-url/path or https://server-url/path - connects to a remote server and creates a proxy
```

---

TITLE: Access and process prompt results (Python)
DESCRIPTION: Explains how to work with the `GetPromptResult` object returned by `get_prompt()`. It shows how to iterate through the `messages` list and access individual message roles and content, handling potential content variations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_5

LANGUAGE: python
CODE:

```
async with client:
    result = await client.get_prompt("conversation_starter", {"topic": "climate"})

    # Access individual messages
    for i, message in enumerate(result.messages):
        print(f"Message {i + 1}:")
        print(f"  Role: {message.role}")
        print(f"  Content: {message.content.text if hasattr(message.content, 'text') else message.content}")
```

---

TITLE: Configure FastMCP Client with OAuth Helper
DESCRIPTION: Shows how to use the `fastmcp.client.auth.OAuth` helper to fully configure the OAuth flow. This helper manages the Authorization Code Grant with PKCE and implements the `httpx.Auth` interface, providing enhanced security and detailed control over the authentication process.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.auth import OAuth

oauth = OAuth(mcp_url="https://fastmcp.cloud/mcp")

async with Client("https://fastmcp.cloud/mcp", auth=oauth) as client:
    await client.ping()
```

---

TITLE: FastMCP: Defining Resource Templates with URI Parameters
DESCRIPTION: Illustrates how to create dynamic resource templates in FastMCP using the `@mcp.resource` decorator. It shows how to embed placeholders like `{parameter_name}` in the URI to map to function arguments, enabling on-demand resource generation based on client requests for varying parameters.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_10

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

# Template URI includes {city} placeholder
@mcp.resource("weather://{city}/current")
def get_weather(city: str) -> dict:
    """Provides weather information for a specific city."""
    # In a real implementation, this would call a weather API
    # Here we're using simplified logic for example purposes
    return {
        "city": city.capitalize(),
        "temperature": 22,
        "condition": "Sunny",
        "unit": "celsius"
    }

# Template with multiple parameters
@mcp.resource("repos://{owner}/{repo}/info")
def get_repo_info(owner: str, repo: str) -> dict:
    """Retrieves information about a GitHub repository."""
    # In a real implementation, this would call the GitHub API
    return {
        "owner": owner,
        "name": repo,
        "full_name": f"{owner}/{repo}",
        "stars": 120,
        "forks": 48
    }
```

---

TITLE: FastMCP Constructor Parameters Reference
DESCRIPTION: Detailed API documentation for the FastMCP constructor parameters, including their types, descriptions, and default values. These parameters control server behavior such as dependencies, tag-based filtering, and duplicate registration policies.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_13

LANGUAGE: APIDOC
CODE:

```
FastMCP Constructor Parameters:
  dependencies: list[str] | None
    Optional server dependencies list with package specifications.
  include_tags: set[str] | None
    Only expose components with at least one matching tag.
  exclude_tags: set[str] | None
    Hide components with any matching tag.
  on_duplicate_tools: Literal["error", "warn", "replace"] (default: "error")
    How to handle duplicate tool registrations.
  on_duplicate_resources: Literal["error", "warn", "replace"] (default: "warn")
    How to handle duplicate resource registrations.
  on_duplicate_prompts: Literal["error", "warn", "replace"] (default: "replace")
    How to handle duplicate prompt registrations.
```

---

TITLE: Mounting a FastMCP Proxy Server in Python
DESCRIPTION: This Python snippet demonstrates how to create a proxy for a remote server using FastMCP.as_proxy() and subsequently mount it onto a main FastMCP server. It highlights that this process inherently uses proxy mounting, enabling seamless interaction with external services.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_4

LANGUAGE: python
CODE:

```
# Create a proxy for a remote server
remote_proxy = FastMCP.as_proxy(Client("http://example.com/mcp"))

# Mount the proxy (always uses proxy mounting)
main_server.mount(remote_proxy, prefix="remote")
```

---

TITLE: Obtaining FastMCP Starlette App Instance
DESCRIPTION: Demonstrates how to get a Starlette ASGI application instance from a FastMCP server using `http_app()` for Streamable HTTP transport (recommended) and `http_app(transport="sse")` for legacy SSE transport. It also shows how to define a tool on the FastMCP instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

# Get a Starlette app instance for Streamable HTTP transport (recommended)
http_app = mcp.http_app()

# For legacy SSE transport (deprecated)
sse_app = mcp.http_app(transport="sse")
```

---

TITLE: Run a FastMCP Server from the Command Line
DESCRIPTION: This command-line instruction shows how to execute a FastMCP server defined in a Python file (e.g., server.py) using the fastmcp run command. This starts the server, making its defined tools and resources accessible.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_1

LANGUAGE: Bash
CODE:

```
fastmcp run server.py
```

---

TITLE: FastMCP Install Command Usage Examples
DESCRIPTION: These examples illustrate various ways to use the `fastmcp install` command. It demonstrates auto-detection of the server object, specifying a custom server object (e.g., `server.py:my_server`), and combining with a custom name (`-n`) and additional dependencies (`--with pandas`). The command supports `file.py:object` notation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_13

LANGUAGE: bash
CODE:

```
# Auto-detects server object (looks for 'mcp', 'server', or 'app')
fastmcp install server.py

# Uses specific server object
fastmcp install server.py:my_server

# With custom name and dependencies
fastmcp install server.py:my_server -n "My Analysis Server" --with pandas
```

---

TITLE: Streamable HTTP Transport: Client Inference from URL
DESCRIPTION: Demonstrates how the `Client` automatically infers and uses `StreamableHttpTransport` when provided with an HTTP or HTTPS URL, simplifying connection to FastMCP servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import Client
import asyncio

# The Client automatically uses StreamableHttpTransport for HTTP URLs
client = Client("https://example.com/mcp")

async def main():
    async with client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

asyncio.run(main())
```

---

TITLE: Passing Command Line Arguments to FastMCP Servers via CLI
DESCRIPTION: These examples illustrate how to pass custom command-line arguments to your FastMCP server scripts when running them via the `fastmcp` CLI. Arguments intended for the server script itself must be passed after a `--` separator.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_4

LANGUAGE: bash
CODE:

```
fastmcp run config_server.py -- --config config.json
fastmcp run database_server.py -- --database-path /tmp/db.sqlite --debug
```

---

TITLE: Pass complex arguments for automatic serialization (Python)
DESCRIPTION: Demonstrates FastMCP's automatic serialization of complex Python objects (dataclasses, dicts, lists) into JSON strings when passed as prompt arguments. Simple strings are passed unchanged, simplifying data handling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from dataclasses import dataclass

@dataclass
class UserData:
    name: str
    age: int

async with client:
    # Complex arguments are automatically serialized
    result = await client.get_prompt("analyze_user", {
        "user": UserData(name="Alice", age=30),     # Automatically serialized to JSON
        "preferences": {"theme": "dark"},           # Dict serialized to JSON string
        "scores": [85, 92, 78],                     # List serialized to JSON string
        "simple_name": "Bob"                        # Strings passed through unchanged
    })
```

---

TITLE: Accessing FastMCP Context in Python Prompts
DESCRIPTION: Shows how prompts can access additional FastMCP information and features through the 'Context' object. This is achieved by adding a parameter with a 'Context' type annotation to the prompt function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_10

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context

mcp = FastMCP(name="PromptServer")

@mcp.prompt
async def generate_report_request(report_type: str, ctx: Context) -> str:
    """Generates a request for a report."""
    return f"Please create a {report_type} report. Request ID: {ctx.request_id}"
```

---

TITLE: Configure Automatic Retry with Exponential Backoff in FastMCP
DESCRIPTION: This snippet demonstrates how to add `RetryMiddleware` to a FastMCP instance. It configures the middleware to automatically retry failed requests up to 3 times when `ConnectionError` or `TimeoutError` exceptions occur, providing resilience against transient network issues.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_19

LANGUAGE: python
CODE:

```
mcp.add_middleware(RetryMiddleware(
    max_retries=3,
    retry_exceptions=(ConnectionError, TimeoutError)
))
```

---

TITLE: Generate Multi-Turn Conversation Templates (Python)
DESCRIPTION: This Python snippet illustrates fetching an 'interview_template' prompt with 'candidate_name' and 'position' arguments. It then iterates through the resulting messages, printing each message's role and content to display the multi-turn conversation flow.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_9

LANGUAGE: python
CODE:

```
async with client:
    result = await client.get_prompt("interview_template", {
        "candidate_name": "Alice",
        "position": "Senior Developer"
    })

    # Multiple messages for a conversation flow
    for message in result.messages:
        print(f"{message.role}: {message.content}")
```

---

TITLE: Importing FastMCP Servers Without Prefixes
DESCRIPTION: This Python example demonstrates how to import components from a subserver into a main server using `main_mcp.import_server(weather_mcp)` without specifying a prefix. Components retain their original names and URIs, and the example includes defining tools and resources, then running the setup asynchronously.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
import asyncio

# Define subservers
weather_mcp = FastMCP(name="WeatherService")

@weather_mcp.tool
def get_forecast(city: str) -> dict:
    """Get weather forecast."""
    return {"city": city, "forecast": "Sunny"}

@weather_mcp.resource("data://cities/supported")
def list_supported_cities() -> list[str]:
    """List cities with weather support."""
    return ["London", "Paris", "Tokyo"]

# Define main server
main_mcp = FastMCP(name="MainApp")

# Import subserver
async def setup():
    # Import without prefix - components keep original names
    await main_mcp.import_server(weather_mcp)

# Result: main_mcp now contains:
# - Tool: "get_forecast" (original name preserved)
# - Resource: "data://cities/supported" (original URI preserved)

if __name__ == "__main__":
    asyncio.run(setup())
    main_mcp.run()
```

---

TITLE: Handle Text Resources with Python MCP Client
DESCRIPTION: Shows how to process text content returned by `client.read_resource()`. It iterates through the content items, checks for the `text` attribute, and prints the text content along with its MIME type.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_4

LANGUAGE: python
CODE:

```
async with client:
    content = await client.read_resource("resource://config/settings.json")

    for item in content:
        if hasattr(item, 'text'):
            print(f"Text content: {item.text}")
            print(f"MIME type: {item.mimeType}")
```

---

TITLE: Handle Various Return Types with FastMCP Tools in Python
DESCRIPTION: This example showcases how FastMCP automatically converts different Python return types into appropriate content formats for the client, including text, images, and handling `None` for no response. It demonstrates returning an `Image` object and a `None` value.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_12

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
import io

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Please install the `pillow` library to run this example.")

mcp = FastMCP("Image Demo")

@mcp.tool
def generate_image(width: int, height: int, color: str) -> Image:
    """Generates a solid color image."""
    # Create image using Pillow
    img = PILImage.new("RGB", (width, height), color=color)

    # Save to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    # Return using FastMCP's Image helper
    return Image(data=img_bytes, format="png")

@mcp.tool
def do_nothing() -> None:
    """This tool performs an action but returns no data."""
    print("Performing a side effect...")
    return None
```

---

TITLE: Inspect FastMCP Server and Generate JSON Report
DESCRIPTION: The `inspect` command generates a detailed JSON report about a FastMCP server. This report includes comprehensive information regarding the server's tools, prompts, resources, and overall capabilities. It's useful for understanding the server's configuration and exposed functionalities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_14

LANGUAGE: bash
CODE:

```
fastmcp inspect server.py
```

---

TITLE: Bridging Transports with FastMCP Proxy
DESCRIPTION: This example illustrates how to use FastMCP to bridge different transports, specifically making a remote SSE server accessible locally via a different transport (e.g., Stdio). The proxy targets the remote server directly by its URL.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Target a remote SSE server directly by URL
proxy = FastMCP.as_proxy("http://example.com/mcp/sse", name="SSE to Stdio Proxy")

# The proxy can now be used with any transport
# No special handling needed - it works like any FastMCP server
```

---

TITLE: Direct Python Call Example with Native Types
DESCRIPTION: This Python code demonstrates that while MCP clients require string arguments, the `analyze_data` prompt can still be called directly within a Python application using native Python types (e.g., `list`, `dict`, `float`) for improved developer convenience.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_5

LANGUAGE: python
CODE:

```
# This also works for direct calls
result = await prompt.render({
    "numbers": [1, 2, 3, 4, 5],
    "metadata": {"source": "api", "version": "1.0"},
    "threshold": 2.5
})
```

---

TITLE: Connect to MCP Server via UVX Stdio Transport (Python)
DESCRIPTION: Demonstrates how to use `fastmcp.client.transports.UvxStdioTransport` to run an MCP server packaged as a Python tool using `uvx`. This transport is useful for executing servers distributed as command-line tools or packages without installing them into your environment. The example shows initializing the transport, creating a client, and asynchronously calling a tool.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_12

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import UvxStdioTransport

# Run a hypothetical 'cloud-analyzer-mcp' tool via uvx
transport = UvxStdioTransport(
    tool_name="cloud-analyzer-mcp",
    # from_package="cloud-analyzer-cli", # Optional: specify package if tool name differs
    # with_packages=["boto3", "requests"] # Optional: add dependencies
)
client = Client(transport)

async def main():
    async with client:
        result = await client.call_tool("analyze_bucket", {"name": "my-data"})
        print(f"Analysis result: {result}")

asyncio.run(main())
```

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.UvxStdioTransport
Inferred From: Not automatically inferred
Use Case: Running an MCP server packaged as a Python tool using uvx
```

---

TITLE: Access FastMCP Context in Resource Functions
DESCRIPTION: Illustrates how to access the `Context` object within FastMCP resource functions. By type-hinting a parameter as `Context`, resources can retrieve additional information like `request_id` or other MCP-specific features.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context

mcp = FastMCP(name="DataServer")

@mcp.resource("resource://system-status")
async def get_system_status(ctx: Context) -> dict:
    """Provides system status information."""
    return {
        "status": "operational",
        "request_id": ctx.request_id
    }

@mcp.resource("resource://{name}/details")
async def get_details(name: str, ctx: Context) -> dict:
    """Get details for a specific name."""
    return {
        "name": name,
        "accessed_at": ctx.request_id
    }
```

---

TITLE: Configure FastMCP Server Environment Variables
DESCRIPTION: This JSON configuration illustrates how to pass environment variables to a FastMCP server. It defines a `weather-server` that runs a Python script and sets `API_KEY` and `DEBUG` variables, emphasizing the need for explicit variable passing in Claude Desktop's isolated environment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_6

LANGUAGE: json
CODE:

```
{
  "mcpServers": {
    "weather-server": {
      "command": "python",
      "args": ["path/to/weather_server.py"],
      "env": {
        "API_KEY": "your-api-key",
        "DEBUG": "true"
      }
    }
  }
}
```

---

TITLE: Configure Advanced Error Handling Middleware in FastMCP
DESCRIPTION: This Python example illustrates how to integrate FastMCP's built-in `ErrorHandlingMiddleware` and `RetryMiddleware` for comprehensive error logging, transformation, and potential retry mechanisms, enhancing server resilience.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_18

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware.error_handling import (
    ErrorHandlingMiddleware,
    RetryMiddleware
)

# Comprehensive error logging and transformation
mcp.add_middleware(ErrorHandlingMiddleware(
    include_traceback=True,
    transform_errors=True,
    error_callback=my_error_callback
))
```

---

TITLE: Running FastMCP Server via CLI
DESCRIPTION: This command shows how to run a FastMCP server using the `fastmcp` command-line interface, specifying the server's Python file. The CLI automatically looks for a FastMCP object (e.g., `mcp`, `server`, or `app`) and calls its `run()` method directly.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
fastmcp run server.py
```

---

TITLE: Python: TransformedTool.from_tool Usage Examples
DESCRIPTION: Provides Python code examples demonstrating how to use `TransformedTool.from_tool` for various transformation scenarios, including simple argument renames, custom functions with partial argument mapping, and handling all arguments with `**kwargs`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-tools-tool_transform.mdx#_snippet_4

LANGUAGE: python
CODE:

```
# Transform specific arguments only
Tool.from_tool(parent, transform_args={"old": "new"})  # Others unchanged
```

LANGUAGE: python
CODE:

```
# Custom function with partial transforms
async def custom(x: int, y: int) -> str:
    result = await forward(x=x, y=y)
    return f"Custom: {result}"
Tool.from_tool(parent, transform_fn=custom, transform_args={"a": "x", "b": "y"})
```

LANGUAGE: python
CODE:

```
# Using **kwargs (gets all args, transformed and untransformed)
async def flexible(**kwargs) -> str:
    result = await forward(**kwargs)
    return f"Got: {kwargs}"
Tool.from_tool(parent, transform_fn=flexible, transform_args={"a": "x"})
```

---

TITLE: FastMCP Prompt with Required and Optional Parameters
DESCRIPTION: This snippet illustrates how FastMCP determines whether a prompt parameter is required or optional. Parameters without a default value (e.g., `data_uri`) are considered required, while those with a default value (e.g., `analysis_type`, `include_charts`) are optional and will use their defaults if not provided by the client.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_7

LANGUAGE: python
CODE:

```
@mcp.prompt
def data_analysis_prompt(
    data_uri: str,                        # Required - no default value
    analysis_type: str = "summary",       # Optional - has default value
    include_charts: bool = False          # Optional - has default value
) -> str:
    """Creates a request to analyze data with specific parameters."""
    prompt = f"Please perform a '{analysis_type}' analysis on the data found at {data_uri}."
    if include_charts:
        prompt += " Include relevant charts and visualizations."
    return prompt
```

---

TITLE: Inspect FastMCP Server and Generate JSON Report
DESCRIPTION: Analyzes a FastMCP server (compatible with v1.x or v2.x) and produces a comprehensive JSON report. The report includes detailed information about the server's name, instructions, version, tools, prompts, resources, templates, and capabilities. The output file path for the JSON report can be customized.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-cli-cli.mdx#_snippet_4

LANGUAGE: APIDOC
CODE:

```
inspect(server_spec: str = typer.Argument(..., help='Python file to inspect, optionally with :object suffix'), output: Annotated[Path, typer.Option('--output', '-o', help='Output file path for the JSON report (default: server-info.json)')] = Path('server-info.json')) -> None
```

LANGUAGE: cli
CODE:

```
fastmcp inspect server.py
fastmcp inspect server.py -o report.json
fastmcp inspect server.py:mcp -o analysis.json
fastmcp inspect path/to/server.py:app -o /tmp/server-info.json
```

---

TITLE: Create and Add Custom Middleware to FastMCP
DESCRIPTION: This snippet provides an example of creating a custom middleware by extending the `Middleware` base class in FastMCP. The `CustomHeaderMiddleware` demonstrates how to intercept requests (`on_request` method), perform custom logic before and after the request is processed by the next middleware, and then add it to the FastMCP instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_21

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext

class CustomHeaderMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Add custom logic here
        print(f"Processing {context.method}")

        result = await call_next(context)

        print(f"Completed {context.method}")
        return result

mcp.add_middleware(CustomHeaderMiddleware())
```

---

TITLE: APIDOC: FastMCP Function Resource for Lazy Loading
DESCRIPTION: A resource that defers data loading by wrapping a function. The function is only called when the resource is read, allowing for lazy loading of potentially expensive data. This is particularly useful when listing resources, as the function won't be called until the resource is actually accessed. The function can return str, bytes, or other types which will be converted to JSON.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-resources-resource.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
FunctionResource:
  from_function(cls, fn: Callable[[], Any], uri: str | AnyUrl, name: str | None = None, description: str | None = None, mime_type: str | None = None, tags: set[str] | None = None, enabled: bool | None = None) -> FunctionResource
    Create a FunctionResource from a function.
```

---

TITLE: Set OpenAI API Key Environment Variable (Bash)
DESCRIPTION: This command sets the `OPENAI_API_KEY` environment variable, which is used by the OpenAI Python SDK for authentication. Users should replace "your-api-key" with their actual OpenAI API key to enable API calls.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
export OPENAI_API_KEY="your-api-key"
```

---

TITLE: Configure and Connect to Multiple MCP Servers via MCPConfig Transport (Python)
DESCRIPTION: Demonstrates using `fastmcp.client.transports.MCPConfigTransport` to connect to one or more MCP servers defined in a configuration object. This transport follows an emerging standard for MCP server configuration and supports both local servers (running via stdio) and remote servers (accessed via HTTP).
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_15

LANGUAGE: python
CODE:

```
from fastmcp import Client

# Configuration for multiple MCP servers (both local and remote)
config = {
    "mcpServers": {
        # Remote HTTP server
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "http"
        },
        # Local stdio server
        "assistant": {
            "command": "python",
            "args": ["./assistant_server.py"],
            "env": {"DEBUG": "true"}
        },
        # Another remote server
        "calendar": {
            "url": "https://calendar-api.example.com/mcp",
            "transport": "http"
        }
    }
}
```

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.MCPConfigTransport
Inferred From: An instance of MCPConfig or a dictionary matching the MCPConfig schema
Use Case: Connecting to one or more MCP servers defined in a configuration object
```

---

TITLE: Configure Custom Python Log Handler for MCP Client
DESCRIPTION: This Python snippet demonstrates how to integrate a custom asynchronous `log_handler` function when initializing a `fastmcp.Client`. The handler receives `LogMessage` objects, allowing for custom processing and display of server-emitted log data, including level, logger name, and content.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/logging.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.logging import LogMessage

async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger = message.logger or 'server'
    data = message.data
    print(f"[{level}] {logger}: {data}")

client = Client(
    "my_mcp_server.py",
    log_handler=log_handler,
)
```

---

TITLE: Accessing FastMCP Resource and Prompt Metadata in Middleware
DESCRIPTION: Illustrates how to apply similar metadata access patterns for resources and prompts using 'on_read_resource' and 'on_get_prompt' hooks. This allows for enforcing restrictions based on resource tags or checking the enabled status of prompts.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ResourceError, PromptError

class ComponentAccessMiddleware(Middleware):
    async def on_read_resource(self, context: MiddlewareContext, call_next):
        if context.fastmcp_context:
            try:
                resource = await context.fastmcp_context.fastmcp.get_resource(context.message.uri)
                if "restricted" in resource.tags:
                    raise ResourceError("Access denied: restricted resource")
            except Exception:
                pass
        return await call_next(context)

    async def on_get_prompt(self, context: MiddlewareContext, call_next):
        if context.fastmcp_context:
            try:
                prompt = await context.fastmcp_context.fastmcp.get_prompt(context.message.name)
                if not prompt.enabled:
                    raise PromptError("Prompt is currently disabled")
            except Exception:
                pass
        return await call_next(context)
```

---

TITLE: Register Function as Resource with FastMCP Server
DESCRIPTION: Decorator to register a Python function as a resource or resource template. The function will be called when the resource is read to generate its content, supporting various return types (str, bytes, or JSON-convertible). Resources can optionally request a Context object for server capabilities. If the URI or function has parameters, it will be registered as a template resource.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_8

LANGUAGE: APIDOC
CODE:

```
resource(self, uri: str) -> Callable[[AnyFunction], Resource | ResourceTemplate]

Args:
- uri: URI for the resource (e.g. "resource://my-resource" or "resource://{param}").
- name: Optional name for the resource.
- description: Optional description of the resource.
- mime_type: Optional MIME type for the resource.
- tags: Optional set of tags for categorizing the resource.
- enabled: Optional boolean to enable or disable the resource.
```

---

TITLE: Handling Tool Execution Results in FastMCP Client
DESCRIPTION: Explains how to process the list of content objects returned by tool execution, distinguishing between `TextContent` and binary data based on available attributes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_4

LANGUAGE: python
CODE:

```
async with client:
    result = await client.call_tool("get_weather", {"city": "London"})

    for content in result:
        if hasattr(content, 'text'):
            print(f"Text result: {content.text}")
        elif hasattr(content, 'data'):
            print(f"Binary data: {len(content.data)} bytes")
```

---

TITLE: Authenticate OpenAI Client for FastMCP Server
DESCRIPTION: This code demonstrates how to authenticate an OpenAI client when calling a FastMCP server secured with bearer tokens. The access token is passed within the `Authorization` header using the `Bearer` scheme in the `headers` dictionary of the MCP tool configuration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_10

LANGUAGE: python
CODE:

```
from openai import OpenAI

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

# Your access token (replace with your actual token)
access_token = 'your-access-token'

client = OpenAI()

resp = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "dice_server",
            "server_url": f"{url}/mcp/",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {access_token}"
            }
        },
    ],
    input="Roll a few dice!",
)

print(resp.output_text)
```

---

TITLE: FastMCP Tool with Context Dependency Injection
DESCRIPTION: This Python code defines a FastMCP tool `process_file` that demonstrates how to access the `Context` object via dependency injection. By type-hinting a parameter as `Context`, FastMCP automatically provides the context instance, enabling the tool to interact with MCP session capabilities like logging or resource access.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Context

mcp = FastMCP(name="ContextDemo")

@mcp.tool
async def process_file(file_uri: str, ctx: Context) -> str:
    """Processes a file, using context for logging and resource access."""
    # Context is available as the ctx parameter
    return "Processed file"
```

---

TITLE: Creating FastMCP Proxy Server
DESCRIPTION: This class method generates a FastMCP proxy server for a specified backend. The `backend` argument is flexible, accepting either an existing `Client` instance or any value compatible with the `transport` argument of the `Client` constructor, mirroring its convenience.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_18

LANGUAGE: python
CODE:

```
as_proxy(cls, backend: Client[ClientTransportT] | ClientTransport | FastMCP[Any] | AnyUrl | Path | MCPConfig | dict[str, Any] | str, **settings: Any) -> FastMCPProxy
```

---

TITLE: Implementing Custom Tool Serialization in FastMCP with Python
DESCRIPTION: Demonstrates how to provide a custom `tool_serializer` function to FastMCP, allowing non-string tool return values to be formatted differently (e.g., YAML instead of JSON). This offers control over output formatting for client consumption.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_18

LANGUAGE: python
CODE:

```
import yaml
from fastmcp import FastMCP

# Define a custom serializer that formats dictionaries as YAML
def yaml_serializer(data):
    return yaml.dump(data, sort_keys=False)

# Create a server with the custom serializer
mcp = FastMCP(name="MyServer", tool_serializer=yaml_serializer)

@mcp.tool
def get_config():
    """Returns configuration in YAML format."""
    return {"api_key": "abc123", "debug": True, "rate_limit": 100}
```

---

TITLE: Generate System Messages for LLM Configuration (Python)
DESCRIPTION: This Python snippet demonstrates how to fetch a system configuration prompt using a client. It shows how to pass 'role' and 'expertise' as arguments to the prompt and then access the generated system message content, which typically has the 'system' role.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_8

LANGUAGE: python
CODE:

```
async with client:
    result = await client.get_prompt("system_configuration", {
        "role": "helpful assistant",
        "expertise": "python programming"
    })

    # Typically returns messages with role="system"
    system_message = result.messages[0]
    print(f"System prompt: {system_message.content}")
```

---

TITLE: Customizing Resource Properties with @mcp.resource Decorator Arguments
DESCRIPTION: Illustrates how to customize a FastMCP resource's properties such as its URI, name, description, MIME type, and tags using arguments within the `@mcp.resource` decorator. This allows for more explicit control over resource metadata.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

# Example specifying metadata
@mcp.resource(
    uri="data://app-status",      # Explicit URI (required)
    name="ApplicationStatus",     # Custom name
    description="Provides the current status of the application.", # Custom description
    mime_type="application/json", # Explicit MIME type
    tags={"monitoring", "status"} # Categorization tags
)
def get_application_status() -> dict:
    """Internal function description (ignored if description is provided above)."""
    return {"status": "ok", "uptime": 12345, "version": mcp.settings.version} # Example usage
```

---

TITLE: Register Function as Prompt with FastMCP Server
DESCRIPTION: Decorator to register a Python function as a prompt within the FastMCP server. This method supports different calling patterns for registering prompts, either directly with a function or as a decorator.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_10

LANGUAGE: APIDOC
CODE:

```
prompt(self, name_or_fn: AnyFunction) -> FunctionPrompt
prompt(self, name_or_fn: str | None = None) -> Callable[[AnyFunction], FunctionPrompt]
```

---

TITLE: FastMCP Run Command Examples
DESCRIPTION: Illustrates various ways to use the `fastmcp run` command, including running a local server with Streamable HTTP transport on a custom port, proxying a remote server, and connecting to a remote server with a specified log level.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_7

LANGUAGE: bash
CODE:

```
# Run a local server with Streamable HTTP transport on a custom port
fastmcp run server.py --transport http --port 8000
```

LANGUAGE: bash
CODE:

```
# Connect to a remote server and proxy as a stdio server
fastmcp run https://example.com/mcp-server
```

LANGUAGE: bash
CODE:

```
# Connect to a remote server with specified log level
fastmcp run https://example.com/mcp-server --log-level DEBUG
```

---

TITLE: Get a basic prompt without arguments (Python)
DESCRIPTION: Shows how to request a rendered prompt using `client.get_prompt()` with just the prompt name. It then accesses and prints the generated messages from the `GetPromptResult` object.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_1

LANGUAGE: python
CODE:

```
async with client:
    # Simple prompt without arguments
    result = await client.get_prompt("welcome_message")
    # result -> mcp.types.GetPromptResult

    # Access the generated messages
    for message in result.messages:
        print(f"Role: {message.role}")
        print(f"Content: {message.content}")
```

---

TITLE: Run FastMCP Unit Tests with Coverage Report
DESCRIPTION: This command runs the FastMCP unit tests using `pytest` via `uv`, generating a code coverage report. The report covers the `src` and `examples` directories and outputs an HTML report for detailed analysis of test coverage.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_18

LANGUAGE: bash
CODE:

```
uv run pytest --cov=src --cov=examples --cov-report=html
```

---

TITLE: Run FastMCP Unit Tests with pytest
DESCRIPTION: Execute the comprehensive unit test suite for FastMCP using `pytest`. All pull requests are required to introduce and pass appropriate tests to ensure code quality and functionality.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_6

LANGUAGE: bash
CODE:

```
pytest
```

---

TITLE: Python Examples for FastMCP Post and Thread Tools
DESCRIPTION: This comprehensive snippet demonstrates various ways to use the `post` and `create_thread` tools via the FastMCP client. Examples include simple posts, posts with images, replies, quotes, rich text with links and mentions, and multi-part threads.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/examples/atproto_mcp/README.md#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import Client
from atproto_mcp.server import atproto_mcp

async def demo():
    async with Client(atproto_mcp) as client:
        # Simple post
        await client.call_tool("post", {
            "text": "Hello from FastMCP!"
        })

        # Post with image
        await client.call_tool("post", {
            "text": "Beautiful sunset! 🌅",
            "images": ["https://example.com/sunset.jpg"],
            "image_alts": ["Sunset over the ocean"]
        })

        # Reply to a post
        await client.call_tool("post", {
            "text": "Great point!",
            "reply_to": "at://did:plc:xxx/app.bsky.feed.post/yyy"
        })

        # Quote post
        await client.call_tool("post", {
            "text": "This is important:",
            "quote": "at://did:plc:xxx/app.bsky.feed.post/yyy"
        })

        # Rich text with links and mentions
        await client.call_tool("post", {
            "text": "Check out FastMCP by @alternatebuild.dev",
            "links": [{"text": "FastMCP", "url": "https://github.com/jlowin/fastmcp"}],
            "mentions": [{"handle": "alternatebuild.dev", "display_text": "@alternatebuild.dev"}]
        })

        # Advanced: Quote with image
        await client.call_tool("post", {
            "text": "Adding visual context:",
            "quote": "at://did:plc:xxx/app.bsky.feed.post/yyy",
            "images": ["https://example.com/chart.png"]
        })

        # Advanced: Reply with rich text
        await client.call_tool("post", {
            "text": "I agree! See this article for more info",
            "reply_to": "at://did:plc:xxx/app.bsky.feed.post/yyy",
            "links": [{"text": "this article", "url": "https://example.com/article"}]
        })

        # Create a thread
        await client.call_tool("create_thread", {
            "posts": [
                {"text": "Starting a thread about Python 🧵"},
                {"text": "Python is great for rapid prototyping"},
                {"text": "And the ecosystem is amazing!", "images": ["https://example.com/python.jpg"]}
            ]
        })
```

---

TITLE: Proxy an Existing MCP Server with FastMCP
DESCRIPTION: Demonstrates how to use FastMCP.as_proxy to create a proxy for an existing MCP server, whether local or remote. This allows bridging different transport mechanisms or adding a frontend to an existing server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_11

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Client

backend = Client("http://example.com/mcp/sse")
proxy = FastMCP.as_proxy(backend, name="ProxyServer")
# Now use the proxy like any FastMCP server
```

---

TITLE: Create Authenticated FastMCP Proxy for Remote Server
DESCRIPTION: This Python code shows how to set up an authenticated proxy server using `FastMCP`. It involves creating a `Client` with `BearerAuth` for an API token and then passing this authenticated client to `FastMCP.as_proxy` to secure communication with the remote server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_8

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Client
from fastmcp.client.auth import BearerAuth

# Create authenticated client
client = Client(
    "https://api.example.com/mcp/sse",
    auth=BearerAuth(token="your-access-token")
)

# Create proxy using the authenticated client
proxy = FastMCP.as_proxy(client, name="Authenticated Proxy")

if __name__ == "__main__":
    proxy.run()
```

---

TITLE: Configure Bearer Token Authentication with Static Public Key
DESCRIPTION: This example shows how to configure the `BearerAuthProvider` using a static RSA public key provided directly in PEM format. This method is suitable when a JWKS endpoint is not available or preferred.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp.server.auth import BearerAuthProvider
import inspect

public_key_pem = inspect.cleandoc(
    """
    -----BEGIN PUBLIC KEY-----
    MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAy...
    -----END PUBLIC KEY-----
    """
)

auth = BearerAuthProvider(public_key=public_key_pem)
```

---

TITLE: Handle Date and Time Parameters in FastMCP Tools
DESCRIPTION: This example illustrates the use of `datetime`, `date`, and `timedelta` types from Python's `datetime` module as FastMCP tool parameters. FastMCP automatically converts ISO-formatted strings or appropriate objects into the corresponding Python date/time types, ensuring correct data handling for temporal information.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_20

LANGUAGE: python
CODE:

```
from datetime import datetime, date, timedelta

@mcp.tool
def process_date_time(
    event_date: date,             # ISO format date string or date object
    event_time: datetime,         # ISO format datetime string or datetime object
    duration: timedelta = timedelta(hours=1)  # Integer seconds or timedelta
) -> str:
    """Process date and time information."""
    # Types are automatically converted from strings
    assert isinstance(event_date, date)
    assert isinstance(event_time, datetime)
    assert isinstance(duration, timedelta)

    return f"Event on {event_date} at {event_time} for {duration}"
```

---

TITLE: Implementing Raw FastMCP Middleware with **call**
DESCRIPTION: Demonstrates the most fundamental way to create FastMCP middleware by overriding the `__call__` method of the `Middleware` base class. This method intercepts all incoming JSON-RPC messages, providing complete control over request and response processing regardless of message type.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_0

LANGUAGE: Python
CODE:

```
from fastmcp.server.middleware import Middleware, MiddlewareContext

class RawMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        # This method receives ALL messages regardless of type
        print(f"Raw middleware processing: {context.method}")
        result = await call_next(context)
        print(f"Raw middleware completed: {context.method}")
        return result
```

---

TITLE: FastMCP: Advanced Route Mapping with `route_map_fn`
DESCRIPTION: This snippet demonstrates advanced route mapping using a custom `route_map_fn` callable. The function `custom_route_mapper` inspects `HTTPRoute` details (path, tags, method) to programmatically override or refine the `MCPType` assignment. It converts admin routes to `TOOL`s, excludes internal routes, and maps specific POST requests to `RESOURCE_TEMPLATE`s, providing highly flexible control over component typing.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType, HTTPRoute

def custom_route_mapper(route: HTTPRoute, mcp_type: MCPType) -> MCPType | None:
    """Advanced route type mapping."""
    # Convert all admin routes to tools regardless of HTTP method
    if "/admin/" in route.path:
        return MCPType.TOOL

    elif "internal" in route.tags:
        return MCPType.EXCLUDE

    # Convert user detail routes to templates even if they're POST
    elif route.path.startswith("/users/") and route.method == "POST":
        return MCPType.RESOURCE_TEMPLATE

    # Use defaults for all other routes
    return None

mcp = FastMCP.from_openapi(
    ...,
    route_map_fn=custom_route_mapper,
)
```

---

TITLE: Test FastMCP Server with Client
DESCRIPTION: Connects to a running FastMCP server using fastmcp.Client, lists the automatically generated tools, and demonstrates calling a specific tool (get_user_by_id) to fetch data from the underlying REST API. This verifies the server's functionality.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_2

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

async def main():
    # Connect to the MCP server we just created
    async with Client("http://127.0.0.1:8000/mcp/") as client:

        # List the tools that were automatically generated
        tools = await client.list_tools()
        print("Generated Tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # Call one of the generated tools
        print("\n\nCalling tool 'get_user_by_id'...")
        user = await client.call_tool("get_user_by_id", {"id": 1})
        print(f"Result:\n{user[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: FastMCP Prompt with Context Dependency Injection
DESCRIPTION: This Python code demonstrates how to use the `Context` object within a FastMCP prompt function. The `data_analysis_request` prompt receives `ctx: Context` as a parameter, allowing it to leverage MCP session capabilities while generating text based on provided messages.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_2

LANGUAGE: python
CODE:

```
@mcp.prompt
async def data_analysis_request(dataset: str, ctx: Context) -> str:
    """Generate a request to analyze data with contextual information."""
    # Context is available as the ctx parameter
    return f"Please analyze the following dataset: {dataset}"
```

---

TITLE: Incorrect Direct Decoration of Instance Methods: Unbound Method Issue
DESCRIPTION: This snippet illustrates a common mistake: directly decorating an instance method (`add`) with `@mcp.tool`. When the decorator is applied this way, it captures the unbound method. Consequently, when an LLM attempts to use this component, it will incorrectly perceive `self` as a required parameter, leading to errors or unexpected behavior because it cannot provide a value for `self`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

class MyClass:
    @mcp.tool  # This won't work correctly
    def add(self, x, y):
        return x + y
```

---

TITLE: Replacing Tool Logic with a Custom Transform Function in fastmcp
DESCRIPTION: This example shows how to completely replace a parent tool's logic using the `transform_fn` parameter in `Tool.from_tool()`. The asynchronous `transform_fn` defines the new tool's execution, and its arguments determine the final schema, allowing for custom validation or post-processing.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_10

LANGUAGE: python
CODE:

```
async def my_custom_logic(user_input: str, max_length: int = 100) -> str:
    # Your custom logic here - this completely replaces the parent tool
    return f"Custom result for: {user_input[:max_length]}"

Tool.from_tool(transform_fn=my_custom_logic)
```

---

TITLE: Running FastMCP Server in Development Mode with CLI
DESCRIPTION: This command uses the `fastmcp dev` command to run the server, which also launches the MCP Inspector. This mode is particularly useful for development and testing purposes, providing an interactive interface for server interaction.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
fastmcp dev server.py
```

---

TITLE: Logging in FastMCP Tools with Python
DESCRIPTION: This section demonstrates how to send log messages from a FastMCP tool function back to the client using the `ctx` object. It covers different log levels (debug, info, warning, error) and shows how to integrate logging into a data analysis function for visibility and error handling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_4

LANGUAGE: python
CODE:

```
@mcp.tool
async def analyze_data(data: list[float], ctx: Context) -> dict:
    """Analyze numerical data with logging."""
    await ctx.debug("Starting analysis of numerical data")
    await ctx.info(f"Analyzing {len(data)} data points")

    try:
        result = sum(data) / len(data)
        await ctx.info(f"Analysis complete, average: {result}")
        return {"average": result, "count": len(data)}
    except ZeroDivisionError:
        await ctx.warning("Empty data list provided")
        return {"error": "Empty data list"}
    except Exception as e:
        await ctx.error(f"Analysis failed: {str(e)}")
        raise
```

LANGUAGE: APIDOC
CODE:

```
Available Logging Methods:
- ctx.debug(message: str): Low-level details useful for debugging
- ctx.info(message: str): General information about execution
- ctx.warning(message: str): Potential issues that didn't prevent execution
- ctx.error(message: str): Errors that occurred during execution
- ctx.log(level: Literal["debug", "info", "warning", "error"], message: str, logger_name: str | None = None): Generic log method supporting custom logger names
```

---

TITLE: Configure FastMCP Server with Bearer Token Authentication
DESCRIPTION: This code shows how to initialize a `BearerAuthProvider` using the previously generated public key and audience. The configured authentication provider is then passed to the `FastMCP` instance to secure the server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)
```

---

TITLE: FastMCP Inspect Command Usage Examples
DESCRIPTION: These examples showcase the flexibility of the `fastmcp inspect` command. It supports auto-detection of the server object, explicit specification of a server object (e.g., `server.py:my_server`), and the ability to direct the output JSON report to a custom file using the `--output` flag.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_15

LANGUAGE: bash
CODE:

```
# Auto-detect server object
fastmcp inspect server.py

# Specify server object
fastmcp inspect server.py:my_server

# Custom output location
fastmcp inspect server.py --output analysis.json
```

---

TITLE: Configure FastMCP with Custom Route Maps for GET Requests
DESCRIPTION: This Python snippet demonstrates how to configure a FastMCP server to use custom route maps. It shows how to convert GET requests with path parameters (e.g., /users/{id}) into ResourceTemplates and other GET requests (e.g., /users) into Resources, overriding the default Tool conversion. It uses a simplified OpenAPI spec for JSONPlaceholder as an example.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_5

LANGUAGE: python
CODE:

```
import httpx
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType


# Create an HTTP client for the target API
client = httpx.AsyncClient(base_url="https://jsonplaceholder.typicode.com")

# Define a simplified OpenAPI spec for JSONPlaceholder
openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "JSONPlaceholder API", "version": "1.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "Get all users",
                "operationId": "get_users",
                "responses": {"200": {"description": "A list of users."}}
            }
        },
        "/users/{id}": {
            "get": {
                "summary": "Get a user by ID",
                "operationId": "get_user_by_id",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "A single user."}}
            }
        }
    }
}

# Create the MCP server with custom route mapping
mcp = FastMCP.from_openapi(
    openapi_spec=openapi_spec,
    client=client,
    name="JSONPlaceholder MCP Server",
    route_maps=[
        # Map GET requests with path parameters (e.g., /users/{id}) to ResourceTemplate
        RouteMap(methods=["GET"], pattern=r".*\{.*\}.*", mcp_type=MCPType.RESOURCE_TEMPLATE),
        # Map all other GET requests to Resource
        RouteMap(methods=["GET"], mcp_type=MCPType.RESOURCE),
    ]
)

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

---

TITLE: Using Pathlib Paths as Parameters in FastMCP
DESCRIPTION: Explains how to use `pathlib.Path` objects for file system paths in FastMCP tool parameters. When a client sends a string path, FastMCP automatically converts it into a `Path` object, allowing for convenient file system operations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_27

LANGUAGE: python
CODE:

```
from pathlib import Path

@mcp.tool
def process_file(path: Path) -> str:
    """Process a file at the given path."""
    assert isinstance(path, Path)  # Path is properly converted
    return f"Processing file at {path}"
```

---

TITLE: Add Multiple Middleware to FastMCP Server
DESCRIPTION: Demonstrates how to add multiple middleware instances to a FastMCP server. This example highlights that middleware executes in the order it's added, affecting both the pre-processing (on the way in) and post-processing (on the way out) phases of a request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_9

LANGUAGE: python
CODE:

```
mcp = FastMCP("MyServer")

mcp.add_middleware(AuthenticationMiddleware("secret-token"))
mcp.add_middleware(PerformanceMiddleware())
mcp.add_middleware(LoggingMiddleware())
```

---

TITLE: Handle Binary Resources with Python MCP Client
DESCRIPTION: Illustrates how to process binary content returned by `client.read_resource()`. It checks for the `blob` attribute, prints the size of the binary data, its MIME type, and demonstrates saving the binary content to a file.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_5

LANGUAGE: python
CODE:

```
async with client:
    content = await client.read_resource("resource://images/logo.png")

    for item in content:
        if hasattr(item, 'blob'):
            print(f"Binary content: {len(item.blob)} bytes")
            print(f"MIME type: {item.mimeType}")

            # Save to file
            with open("downloaded_logo.png", "wb") as f:
                f.write(item.blob)
```

---

TITLE: Authenticate FastMCP Client with Custom Headers
DESCRIPTION: Explains how to manually set custom headers on the transport instance if the MCP server expects a non-standard header or token scheme, bypassing the `auth` parameter.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/bearer.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async with Client(
    transport=StreamableHttpTransport(
        "https://fastmcp.cloud/mcp",
        headers={"X-API-Key": "<your-token>"},
    ),
) as client:
    await client.ping()
```

---

TITLE: FastMCP Client `call_tool()` Method Parameters
DESCRIPTION: Detailed documentation for the parameters accepted by the `call_tool()` method, including their types and descriptions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
call_tool(name: string, arguments: dict = None, timeout: float = None, progress_handler: function = None)
  name: The tool name (string)
  arguments: Dictionary of arguments to pass to the tool (optional)
  timeout: Maximum execution time in seconds (optional, overrides client-level timeout)
  progress_handler: Progress callback function (optional, overrides client-level handler)
```

---

TITLE: Inferring Client Transport Type with infer_transport
DESCRIPTION: The infer_transport function intelligently determines and returns the appropriate ClientTransport subclass based on the provided input. It supports various input types including ClientTransport instances, FastMCP objects, file paths (for Python or Node stdio), URLs (for HTTP or SSE), and MCPConfig dictionaries (for single or multiple server connections). For multi-server configurations, it creates a composite client allowing access to resources via prefixed names.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-transports.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
infer_transport(transport: ClientTransport | FastMCP | FastMCP1Server | AnyUrl | Path | MCPConfig | dict[str, Any] | str) -> ClientTransport

Description: Infer the appropriate transport type from the given transport argument.
             This function attempts to infer the correct transport type from the provided
             argument, handling various input types and converting them to the appropriate
             ClientTransport subclass.

             The function supports these input types:
             - ClientTransport: Used directly without modification
             - FastMCP or FastMCP1Server: Creates an in-memory FastMCPTransport
             - Path or str (file path): Creates PythonStdioTransport (.py) or NodeStdioTransport (.js)
             - AnyUrl or str (URL): Creates StreamableHttpTransport (default) or SSETransport (for /sse endpoints)
             - MCPConfig or dict: Creates MCPConfigTransport, potentially connecting to multiple servers

             For HTTP URLs, they are assumed to be Streamable HTTP URLs unless they end in `/sse`.

             For MCPConfig with multiple servers, a composite client is created where each server
             is mounted with its name as prefix. This allows accessing tools and resources from multiple
             servers through a single unified client interface, using naming patterns like
             `servername_toolname` for tools and `protocol://servername/path` for resources.
             If the MCPConfig contains only one server, a direct connection is established without prefixing.
```

LANGUAGE: python
CODE:

```
# Connect to a local Python script
transport = infer_transport("my_script.py")

# Connect to a remote server via HTTP
transport = infer_transport("http://example.com/mcp")

# Connect to multiple servers using MCPConfig
config = {
    "mcpServers": {
        "weather": {"url": "http://weather.example.com/mcp"},
        "calendar": {"url": "http://calendar.example.com/mcp"}
    }
}
transport = infer_transport(config)
```

---

TITLE: Generate Test Tokens with FastMCP RSAKeyPair (Python)
DESCRIPTION: This snippet demonstrates how to use the `RSAKeyPair` utility class to generate JWT tokens for development and testing purposes. It includes setting up a `BearerAuthProvider` with the public key and creating a `FastMCP` instance, then generating a token with specified claims and scopes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair

# Generate a new key pair
key_pair = RSAKeyPair.generate()

# Configure the auth provider with the public key
auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    issuer="https://dev.example.com",
    audience="my-dev-server"
)

mcp = FastMCP(name="Development Server", auth=auth)

# Generate a token for testing
token = key_pair.create_token(
    subject="dev-user",
    issuer="https://dev.example.com",
    audience="my-dev-server",
    scopes=["read", "write"]
)

print(f"Test token: {token}")
```

---

TITLE: Compose FastMCP Servers Using Mount
DESCRIPTION: Shows an example of composing two FastMCP servers, 'main' and 'sub', by mounting 'sub' onto 'main' with a specified prefix. This allows organizing large applications into modular components or reusing existing servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_10

LANGUAGE: python
CODE:

```
# Example: Importing a subserver
from fastmcp import FastMCP
import asyncio

main = FastMCP(name="Main")
sub = FastMCP(name="Sub")

@sub.tool
def hello():
    return "hi"

# Mount directly
main.mount(sub, prefix="sub")
```

---

TITLE: FastMCP Resource with Optional Parameters and Defaults
DESCRIPTION: Illustrates how FastMCP handles function parameters with default values. The `search_resources` function defines `max_results` and `include_archived` as optional, allowing clients to request resources using only the required `query` parameter, with FastMCP applying the defaults for the others.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_13

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

@mcp.resource("search://{query}")
def search_resources(query: str, max_results: int = 10, include_archived: bool = False) -> dict:
    """Search for resources matching the query string."""
    # Only 'query' is required in the URI, the other parameters use their defaults
    results = perform_search(query, limit=max_results, archived=include_archived)
    return {
        "query": query,
        "max_results": max_results,
        "include_archived": include_archived,
        "results": results
    }
```

---

TITLE: FastMCP Resources and Templates with Context Injection
DESCRIPTION: This Python code illustrates how to inject the `Context` object into FastMCP resource functions. It shows two examples: a static resource `get_user_data` and a templated resource `get_user_profile`, both receiving `ctx: Context` as a parameter to access MCP capabilities within their logic.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_1

LANGUAGE: python
CODE:

```
@mcp.resource("resource://user-data")
async def get_user_data(ctx: Context) -> dict:
    """Fetch personalized user data based on the request context."""
    # Context is available as the ctx parameter
    return {"user_id": "example"}

@mcp.resource("resource://users/{user_id}/profile")
async def get_user_profile(user_id: str, ctx: Context) -> dict:
    """Fetch user profile with context-aware logging."""
    # Context is available as the ctx parameter
    return {"id": user_id}
```

---

TITLE: Demonstrating Incorrect Direct Decoration of Instance Methods
DESCRIPTION: This example shows why directly applying a FastMCP decorator like `@mcp.tool` to an instance method is problematic. The decorator returns a `Tool` object, replacing the original method, which makes it uncallable by Python code and prevents LLMs from properly interacting with it due to the exposed `self` parameter.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/decorating-methods.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
mcp = FastMCP()

class MyClass:
    @mcp.tool
    def my_method(self, x: int) -> int:
        return x * 2

obj = MyClass()
obj.my_method(5)  # Fails - my_method is a Tool, not a function
```

---

TITLE: FastMCP Client Authentication Error Response
DESCRIPTION: This snippet shows the error response received when an unauthenticated client attempts to access a FastMCP server configured with bearer authentication. The error indicates that an `authorization_token` is required.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_8

LANGUAGE: json
CODE:

```
Error code: 400 - {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "MCP server 'dice-server' requires authentication. Please provide an authorization_token.",
    },
}
```

---

TITLE: Pass Environment Variables to FastMCP Server via CLI
DESCRIPTION: Illustrates how to provide necessary environment variables to a FastMCP server during installation. Variables can be passed individually using `--env-var` or loaded from a `.env` file using `--env-file`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
fastmcp install server.py --name "Weather Server" \
  --env-var API_KEY=your-api-key \
  --env-var DEBUG=true

fastmcp install server.py --name "Weather Server" --env-file .env
```

---

TITLE: Connect a FastMCP Server to Claude Code
DESCRIPTION: Instructions to start your FastMCP server and then add it to Claude Code using the `claude mcp add` command. This enables Claude Code to discover and utilize the server's tools.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-code.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
# Start your server first
python server.py

claude mcp add dice --transport http http://localhost:8000/mcp/
```

---

TITLE: Install FastMCP for Development
DESCRIPTION: To contribute to FastMCP, clone the repository and use `uv sync` to install all dependencies, including development-specific ones. This sets up a virtual environment ready for development work.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_5

LANGUAGE: bash
CODE:

```
git clone https://github.com/jlowin/fastmcp.git
cd fastmcp
uv sync
```

---

TITLE: Defining an MCP Resource Template with FastMCP
DESCRIPTION: Shows how to create a dynamic Resource Template using the @mcp.resource decorator with a URI pattern. Resource Templates allow clients to request dynamic data based on parameters, similar to parameterized GET requests. This example defines a template for fetching user profiles by 'user_id'.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/mcp.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

# This template provides user data for any given user ID
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """Returns the profile for a specific user."""
    # Fetch user from a database...
    return {"id": user_id, "name": "Zaphod Beeblebrox"}
```

---

TITLE: Setting Default Values for Tool Arguments with ArgTransform
DESCRIPTION: Shows how to update the default value for a tool argument using the `default` parameter of `ArgTransform`. The example changes the default value of the `y` argument to 10 for an addition function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform

mcp = FastMCP()

@mcp.tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

new_tool = Tool.from_tool(
    add,
    transform_args={
        "y": ArgTransform(default=10)
    }
)
```

---

TITLE: Customizing FastMCP Tool Metadata with Decorator Arguments
DESCRIPTION: Illustrates how to override the inferred tool name and description, and add optional tags using arguments to the @mcp.tool decorator. This provides precise control over how the tool is presented to the LLM client and allows for better organization and filtering of available tools.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_1

LANGUAGE: python
CODE:

```
@mcp.tool(
    name="find_products",           # Custom tool name for the LLM
    description="Search the product catalog with optional category filtering.", # Custom description
    tags={"catalog", "search"},      # Optional tags for organization/filtering
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    """Internal function description (ignored if description is provided above)."""
    # Implementation...
    print(f"Searching for '{query}' in category '{category}'")
    return [{"id": 2, "name": "Another Product"}]
```

---

TITLE: Run FastMCP Server with HTTP Transport
DESCRIPTION: Demonstrates how to run a local FastMCP server using the `fastmcp run` command, explicitly setting the transport protocol to HTTP and a custom port, overriding any transport specified in the server's `__main__` block.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_6

LANGUAGE: bash
CODE:

```
fastmcp run server.py --transport http --port 8000
```

---

TITLE: Mounting FastMCP Servers for Live Linking
DESCRIPTION: This Python example illustrates the `mount()` method, which creates a live link between a main server and a subserver. It shows how components added to the subserver _after_ mounting are immediately accessible through the main server, demonstrating dynamic updates and prefixed access.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_2

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import FastMCP, Client

# Define subserver
dynamic_mcp = FastMCP(name="DynamicService")

@dynamic_mcp.tool
def initial_tool():
    """Initial tool demonstration."""
    return "Initial Tool Exists"

# Mount subserver (synchronous operation)
main_mcp = FastMCP(name="MainAppLive")
main_mcp.mount(dynamic_mcp, prefix="dynamic")

# Add a tool AFTER mounting - it will be accessible through main_mcp
@dynamic_mcp.tool
def added_later():
    """Tool added after mounting."""
    return "Tool Added Dynamically!"

# Testing access to mounted tools
async def test_dynamic_mount():
    tools = await main_mcp.get_tools()
    print("Available tools:", list(tools.keys()))
    # Shows: ['dynamic_initial_tool', 'dynamic_added_later']

    async with Client(main_mcp) as client:
        result = await client.call_tool("dynamic_added_later")
        print("Result:", result[0].text)
        # Shows: "Tool Added Dynamically!"

if __name__ == "__main__":
    asyncio.run(test_dynamic_mount())
```

---

TITLE: Create FastMCP Proxy for Remote HTTP Server
DESCRIPTION: This Python snippet demonstrates how to create a proxy server using `FastMCP.as_proxy` to forward requests to a remote HTTP server. It shows a basic setup for connecting to an SSE endpoint and running the proxy for Claude Desktop.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Create a proxy to a remote server
proxy = FastMCP.as_proxy(
    "https://example.com/mcp/sse",
    name="Remote Server Proxy"
)

if __name__ == "__main__":
    proxy.run()  # Runs via STDIO for Claude Desktop
```

---

TITLE: Accessing Full HTTP Request Object in FastMCP
DESCRIPTION: This snippet demonstrates how to access the complete HTTP request object within a FastMCP tool using the `get_http_request()` dependency. It shows how to extract details like user agent, client IP, and URL path from the `starlette.requests.Request` object. This method is suitable when full request context is needed, including in helper or nested functions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/http-requests.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from starlette.requests import Request

mcp = FastMCP(name="HTTP Request Demo")

@mcp.tool
async def user_agent_info() -> dict:
    """Return information about the user agent."""
    # Get the HTTP request
    request: Request = get_http_request()

    # Access request data
    user_agent = request.headers.get("user-agent", "Unknown")
    client_ip = request.client.host if request.client else "Unknown"

    return {
        "user_agent": user_agent,
        "client_ip": client_ip,
        "path": request.url.path,
    }
```

---

TITLE: Implement Conditional Python Log Handling by Level
DESCRIPTION: This Python example illustrates a `log_handler` that conditionally processes `LogMessage` objects based on their `level` attribute. It demonstrates how to apply specific logic or formatting for different log severities, such as printing custom prefixes for error or warning messages, enhancing log readability.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/logging.mdx#_snippet_2

LANGUAGE: python
CODE:

```
async def detailed_log_handler(message: LogMessage):
    if message.level == "error":
        print(f"ERROR: {message.data}")
    elif message.level == "warning":
        print(f"WARNING: {message.data}")
    else:
        print(f"{message.level.upper()}: {message.data}")
```

---

TITLE: Understanding FastMCP Middleware Hook Structure
DESCRIPTION: Illustrates the common structure of a FastMCP middleware hook, detailing the four key stages: pre-processing (inspecting/modifying request), chain continuation (calling the next middleware/handler), post-processing (inspecting/modifying response), and returning the result.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_5

LANGUAGE: python
CODE:

```
async def on_message(self, context: MiddlewareContext, call_next):
    # 1. Pre-processing: Inspect and optionally modify the request
    print(f"Processing {context.method}")

    # 2. Chain continuation: Call the next middleware/handler
    result = await call_next(context)

    # 3. Post-processing: Inspect and optionally modify the response
    print(f"Completed {context.method}")

    # 4. Return the result (potentially modified)
    return result
```

---

TITLE: Importing a Contrib Module in Python
DESCRIPTION: This snippet demonstrates the standard way to import a community-contributed module from the `fastmcp.contrib` package. It shows how to access extended functionalities that are not part of the core FastMCP library, noting that these modules might have their own specific dependencies.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/src/fastmcp/contrib/README.md#_snippet_0

LANGUAGE: Python
CODE:

```
from fastmcp.contrib import my_module
```

---

TITLE: Importing a Contrib Module in Python
DESCRIPTION: This snippet demonstrates the basic syntax for importing a community-contributed module from the `fastmcp.contrib` package. Replace `my_module` with the specific name of the module you intend to use. This import statement makes the module's functionalities available for use in your Python application.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/contrib.mdx#_snippet_0

LANGUAGE: python
CODE:

```
from fastmcp.contrib import my_module
```

---

TITLE: FastMCP: Call Parent Tool with `forward()` (Same Arguments)
DESCRIPTION: Demonstrates how to use `forward()` to call the parent tool from within a `transform_fn` when the transformed tool has the same argument names as the parent. This pattern is commonly used for pre-validation of inputs before delegating to the original tool.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_11

LANGUAGE: Python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward

mcp = FastMCP()

@mcp.tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

async def ensure_positive(x: int, y: int) -> int:
    if x <= 0 or y <= 0:
        raise ValueError("x and y must be positive")
    return await forward(x=x, y=y)

new_tool = Tool.from_tool(
    add,
    transform_fn=ensure_positive,
)

mcp.add_tool(new_tool)
```

---

TITLE: Install Google Generative AI SDK (Bash)
DESCRIPTION: This Bash command installs the `google-genai` Python package, which is the official Google Generative AI SDK. This SDK is essential for interacting with the Gemini API and integrating with FastMCP servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
pip install google-genai
```

---

TITLE: Implement Simple Timing Middleware in FastMCP
DESCRIPTION: Provides an example of a custom `SimpleTimingMiddleware` that measures the execution time of MCP requests. It demonstrates how to use `time.perf_counter()` to capture duration for both successful and failed requests, providing basic performance monitoring.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_11

LANGUAGE: python
CODE:

```
import time
from fastmcp.server.middleware import Middleware, MiddlewareContext

class SimpleTimingMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        start_time = time.perf_counter()

        try:
            result = await call_next(context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            print(f"Request {context.method} completed in {duration_ms:.2f}ms")
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            print(f"Request {context.method} failed after {duration_ms:.2f}ms: {e}")
            raise
```

---

TITLE: Deploying FastMCP Server with ngrok
DESCRIPTION: These commands demonstrate how to run the FastMCP server locally and expose it to the internet using `ngrok`. The first command starts the Python FastMCP server, and the second command creates a public HTTP tunnel to the server's port, making it accessible to ChatGPT.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/chatgpt.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
python server.py
```

LANGUAGE: bash
CODE:

```
ngrok http 8000
```

---

TITLE: Configuring FastMCP Settings via Environment Variables in Bash
DESCRIPTION: Illustrates how to set FastMCP global settings using environment variables (e.g., FASTMCP_LOG_LEVEL, FASTMCP_MASK_ERROR_DETAILS, FASTMCP_RESOURCE_PREFIX_FORMAT). This method provides a convenient way to configure the server externally.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_17

LANGUAGE: bash
CODE:

```
export FASTMCP_LOG_LEVEL=DEBUG
export FASTMCP_MASK_ERROR_DETAILS=True
export FASTMCP_RESOURCE_PREFIX_FORMAT=protocol
```

---

TITLE: Customize Prompts with Decorator Arguments in FastMCP
DESCRIPTION: Illustrates how to override inferred prompt metadata like name and description, and add categorization tags using arguments directly within the `@mcp.prompt` decorator. It also shows how `Field` can be used for parameter descriptions within the function signature, demonstrating advanced customization options for prompt definitions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_1

LANGUAGE: python
CODE:

```
@mcp.prompt(
    name="analyze_data_request",          # Custom prompt name
    description="Creates a request to analyze data with specific parameters",  # Custom description
    tags={"analysis", "data"}             # Optional categorization tags
)
def data_analysis_prompt(
    data_uri: str = Field(description="The URI of the resource containing the data."),
    analysis_type: str = Field(default="summary", description="Type of analysis.")
) -> str:
    """This docstring is ignored when description is provided."""
    return f"Please perform a '{analysis_type}' analysis on the data found at {data_uri}."
```

---

TITLE: FastMCP Transport Selection: SSE vs. Streamable HTTP
DESCRIPTION: Provides guidelines on when to choose Server-Sent Events (SSE) versus Streamable HTTP for FastMCP deployments, considering factors like new deployments, bidirectional streaming needs, and compatibility with legacy servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_7

LANGUAGE: APIDOC
CODE:

```
When to Use SSE vs. Streamable HTTP:
- Use Streamable HTTP when:
  - Setting up new deployments (recommended default)
  - You need bidirectional streaming
  - You're connecting to FastMCP servers running in `http` mode
- Use SSE when:
  - Connecting to legacy FastMCP servers running in `sse` mode
  - Working with infrastructure optimized for Server-Sent Events
```

---

TITLE: Access Resources with Multi-Server Python MCP Client
DESCRIPTION: Demonstrates how a multi-server client automatically prefixes resource URIs with the server name. It shows examples of reading resources from different servers using a single client instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_6

LANGUAGE: python
CODE:

```
async with client:  # Multi-server client
    # Access resources from different servers
    weather_icons = await client.read_resource("weather://weather/icons/sunny")
    templates = await client.read_resource("resource://assistant/templates/list")

    print(f"Weather icon: {weather_icons[0].blob}")
    print(f"Templates: {templates[0].text}")
```

---

TITLE: FastMCP Standard Tool Annotations Reference
DESCRIPTION: Provides a reference for standard annotations supported by FastMCP, detailing their type, default values, and purpose for client application integration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_16

LANGUAGE: APIDOC
CODE:

```
Annotation | Type | Default | Purpose
:--------- | :--- | :------ | :------
`title` | string | - | Display name for user interfaces
`readOnlyHint` | boolean | false | Indicates if the tool only reads without making changes
`destructiveHint` | boolean | true | For non-readonly tools, signals if changes are destructive
`idempotentHint` | boolean | false | Indicates if repeated identical calls have the same effect as a single call
`openWorldHint` | boolean | true | Specifies if the tool interacts with external systems
```

---

TITLE: Run FastMCP Server with SSE Transport
DESCRIPTION: This example demonstrates how to configure and run the FastMCP server with the Server-Sent Events (SSE) transport protocol. This option provides compatibility with existing SSE clients and requires specifying the host and port.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_14

LANGUAGE: python
CODE:

```
mcp.run(transport="sse", host="127.0.0.1", port=8000)
```

---

TITLE: Read from Resource Templates with Python MCP Client
DESCRIPTION: Demonstrates reading content generated from a resource template by providing the URI with parameters to `client.read_resource()`. It assumes a text JSON response and prints its content.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_3

LANGUAGE: python
CODE:

```
async with client:
    # Read a resource generated from a template
    # For example, a template like "weather://{{city}}/current"
    weather_content = await client.read_resource("weather://london/current")

    # Access the generated content
    print(weather_content[0].text)  # Assuming text JSON response
```

---

TITLE: Run FastMCP Client Script
DESCRIPTION: Executes the api_client.py script to connect to and test the running FastMCP server, demonstrating tool listing and invocation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_4

LANGUAGE: bash
CODE:

```
python api_client.py
```

---

TITLE: Streamable HTTP Transport: Explicit Instantiation
DESCRIPTION: Shows how to explicitly create an instance of `StreamableHttpTransport` and pass it to the `Client` constructor for more direct control over the transport configuration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(url="https://example.com/mcp")
client = Client(transport)
```

---

TITLE: List Static Resources with Python MCP Client
DESCRIPTION: Demonstrates how to use `client.list_resources()` to retrieve a list of all static resources available on an MCP server. It iterates through the returned `Resource` objects, printing their URI, name, description, and MIME type.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_0

LANGUAGE: python
CODE:

```
async with client:
    resources = await client.list_resources()
    # resources -> list[mcp.types.Resource]

    for resource in resources:
        print(f"Resource URI: {resource.uri}")
        print(f"Name: {resource.name}")
        print(f"Description: {resource.description}")
        print(f"MIME Type: {resource.mimeType}")
```

---

TITLE: Authenticate FastMCP Client using BearerAuth Helper Class
DESCRIPTION: Demonstrates using the `BearerAuth` class, which implements the `httpx.Auth` interface, for more explicit Bearer token handling compared to providing a raw string.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/bearer.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

async with Client(
    "https://fastmcp.cloud/mcp",
    auth=BearerAuth(token="<your-token>"),
) as client:
    await client.ping()
```

---

TITLE: FastMCP Available Middleware Hooks
DESCRIPTION: List of available hooks in FastMCP middleware, detailing their purpose and the types of messages/operations they intercept. These hooks allow developers to inject custom logic at various stages of message processing and component interaction.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
FastMCP Hooks:
  on_message:
    Description: Called for all MCP messages (requests and notifications)
  on_request:
    Description: Called specifically for MCP requests (that expect responses)
  on_notification:
    Description: Called specifically for MCP notifications (fire-and-forget)
  on_call_tool:
    Description: Called when tools are being executed
  on_read_resource:
    Description: Called when resources are being read
  on_get_prompt:
    Description: Called when prompts are being retrieved
  on_list_tools:
    Description: Called when listing available tools
  on_list_resources:
    Description: Called when listing available resources
  on_list_resource_templates:
    Description: Called when listing resource templates
  on_list_prompts:
    Description: Called when listing available prompts
```

---

TITLE: Advanced Tool Execution Options in FastMCP Client
DESCRIPTION: Illustrates using `client.call_tool()` with optional parameters like `timeout` for execution limits and `progress_handler` for monitoring long-running tasks.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_2

LANGUAGE: python
CODE:

```
async with client:
    # With timeout (aborts if execution takes longer than 2 seconds)
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        timeout=2.0
    )

    # With progress handler (to track execution progress)
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )
```

---

TITLE: Tool.from_tool() Method Parameters
DESCRIPTION: Details the parameters available for the `Tool.from_tool()` class method, used for creating transformed tools, including options for modifying metadata, arguments, and behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
Tool.from_tool(
  tool: The tool to transform. This is the only required argument.
  name: An optional name for the new tool.
  description: An optional description for the new tool.
  transform_args: A dictionary of ArgTransform objects, one for each argument you want to modify.
  transform_fn: An optional function that will be called instead of the parent tool's logic.
  tags: An optional set of tags for the new tool.
  annotations: An optional set of ToolAnnotations for the new tool.
  serializer: An optional function that will be called to serialize the result of the new tool.
)
```

---

TITLE: Run FastMCP Server in Development Mode
DESCRIPTION: Executes a FastMCP server in an isolated environment for testing with the MCP Inspector. All dependencies must be explicitly specified using `--with` and/or `--with-editable` options.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_8

LANGUAGE: bash
CODE:

```
fastmcp dev server.py
```

---

TITLE: Python: Create Log Callback Function
DESCRIPTION: Generates a logging callback function. It can optionally take a `LogHandler` to customize log handling. Returns a `LoggingFnT` type, which is the type of the generated logging function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-logging.mdx#_snippet_0

LANGUAGE: Python
CODE:

```
create_log_callback(handler: LogHandler | None = None) -> LoggingFnT
```

---

TITLE: Advanced Configuration for FastMCP FastAPI Integration
DESCRIPTION: Shows how to apply advanced configurations when converting a FastAPI app to a FastMCP server, including custom naming, global tags, and route mapping rules to control how endpoints are exposed as MCP components.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_17

LANGUAGE: python
CODE:

```
from fastmcp.server.openapi import RouteMap, MCPType

# Custom route mapping with FastAPI
mcp = FastMCP.from_fastapi(
    app=app,
    name="My Custom Server",
    timeout=5.0,
    tags={"api-v1", "fastapi"},  # Global tags for all components
    mcp_names={"operationId": "friendly_name"},  # Custom component names
    route_maps=[
        # Admin endpoints become tools with custom tags
        RouteMap(
            methods="*",
            pattern=r"^/admin/.*",
            mcp_type=MCPType.TOOL,
            mcp_tags={"admin", "privileged"}
        ),
        # Internal endpoints are excluded
        RouteMap(methods="*", pattern=r".*", mcp_type=MCPType.EXCLUDE, tags={"internal"}),
    ],
    route_map_fn=my_route_mapper,
    mcp_component_fn=my_component_customizer,
    mcp_names={
        "get_user_details_users__user_id__get": "get_user_details",
    }
)
```

---

TITLE: FastMCP: Handle Arguments with `**kwargs` in `transform_fn`
DESCRIPTION: Explains how `transform_fn` can use `**kwargs` to capture all arguments from the parent tool after `ArgTransform` configurations have been applied. This allows for flexible validation functions that don't require explicitly listing every argument in the signature, passing through unvalidated arguments to the parent tool.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_14

LANGUAGE: Python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward

mcp = FastMCP()

@mcp.tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

async def ensure_a_positive(a: int, **kwargs) -> int:
    if a <= 0:
        raise ValueError("a must be positive")
    return await forward(a=a, **kwargs)

new_tool = Tool.from_tool(
    add,
    transform_fn=ensure_a_positive,
    transform_args={
        "x": ArgTransform(name="a"),
        "y": ArgTransform(name="b"),
    }
)

mcp.add_tool(new_tool)
```

---

TITLE: Clone FastMCP Repository
DESCRIPTION: This command sequence shows how to clone the FastMCP GitHub repository to your local machine and navigate into the newly created project directory. This is the first step for setting up a development environment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_15

LANGUAGE: bash
CODE:

```
git clone https://github.com/jlowin/fastmcp.git
cd fastmcp
```

---

TITLE: Adding Custom Routes to FastMCP Server
DESCRIPTION: Explains how to add custom web routes directly to a FastMCP server using the `@mcp.custom_route` decorator. This feature is useful for simple endpoints like health checks, which will be exposed alongside the main MCP endpoint when the server is mounted.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("MyServer")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")
```

---

TITLE: FastMCP Context Object Capabilities
DESCRIPTION: Lists the key functionalities provided by the 'Context' object in FastMCP, including logging, progress reporting, resource access, LLM sampling, and request information.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_18

LANGUAGE: APIDOC
CODE:

```
The Context object provides access to:

- Logging: `ctx.debug()`, `ctx.info()`, `ctx.warning()`, `ctx.error()`
- Progress Reporting: `ctx.report_progress(progress, total)`
- Resource Access: `ctx.read_resource(uri)`
- LLM Sampling: `ctx.sample(...)`
- Request Information: `ctx.request_id`, `ctx.client_id`
```

---

TITLE: fastmcp.client.client.Client Class API Reference
DESCRIPTION: Detailed API documentation for the `Client` class, outlining its constructor parameters and available methods for managing MCP connections and interactions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-client.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
Client:
  description: MCP client that delegates connection management to a Transport instance. The Client class is responsible for MCP protocol logic, while the Transport handles connection establishment and management. Client provides methods for working with resources, prompts, tools and other MCP capabilities.
  Constructor Arguments:
    transport: Connection source specification, which can be:
      - ClientTransport: Direct transport instance
      - FastMCP: In-process FastMCP server
      - AnyUrl | str: URL to connect to
      - Path: File path for local socket
      - MCPConfig: MCP server configuration
      - dict: Transport configuration
    roots: Optional RootsList or RootsHandler for filesystem access
    sampling_handler: Optional handler for sampling requests
    log_handler: Optional handler for log messages
    message_handler: Optional handler for protocol messages
    progress_handler: Optional handler for progress notifications
    timeout: Optional timeout for requests (seconds or timedelta)
    init_timeout: Optional timeout for initial connection (seconds or timedelta). Set to 0 to disable. If None, uses the value in the FastMCP global settings.
  Methods:
    session():
      signature: session(self) -> ClientSession
      returns: ClientSession
      description: Get the current active session. Raises RuntimeError if not connected.
    initialize_result():
      signature: initialize_result(self) -> mcp.types.InitializeResult
      returns: mcp.types.InitializeResult
      description: Get the result of the initialization request.
    set_roots(roots: RootsList | RootsHandler):
      signature: set_roots(self, roots: RootsList | RootsHandler) -> None
      parameters:
        roots: RootsList | RootsHandler
      returns: None
      description: Set the roots for the client. This does not automatically call `send_roots_list_changed`.
    set_sampling_callback(sampling_callback: SamplingHandler):
      signature: set_sampling_callback(self, sampling_callback: SamplingHandler) -> None
      parameters:
        sampling_callback: SamplingHandler
      returns: None
      description: Set the sampling callback for the client.
    is_connected():
      signature: is_connected(self) -> bool
      returns: bool
      description: Check if the client is currently connected.
```

---

TITLE: APIDOC: FastMCP Class
DESCRIPTION: Documents the 'FastMCP' class, providing an ergonomic interface for MCP servers. It includes methods for server configuration, execution, middleware management, custom route registration, and tool handling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
FastMCP:
  settings(self) -> Settings
  name(self) -> str
  instructions(self) -> str | None
  run(self, transport: Literal['stdio', 'streamable-http', 'sse'] | None = None, **transport_kwargs: Any) -> None
    Run the FastMCP server. Note this is a synchronous function.
    Args:
      - transport: Transport protocol to use ("stdio", "sse", or "streamable-http")
  add_middleware(self, middleware: Middleware) -> None
  custom_route(self, path: str, methods: list[str], name: str | None = None, include_in_schema: bool = True)
    Decorator to register a custom HTTP route on the FastMCP server.
    Allows adding arbitrary HTTP endpoints outside the standard MCP protocol,
    which can be useful for OAuth callbacks, health checks, or admin APIs.
    The handler function must be an async function that accepts a Starlette
    Request and returns a Response.
    Args:
      - path: URL path for the route (e.g., "/oauth/callback")
      - methods: List of HTTP methods to support (e.g., ["GET", "POST"])
      - name: Optional name for the route (to reference this route with Starlette's reverse URL lookup feature)
      - include_in_schema: Whether to include in OpenAPI schema, defaults to True
  add_tool(self, tool: Tool) -> None
    Add a tool to the server.
    The tool function can optionally request a Context object by adding a parameter
    with the Context type annotation. See the @tool decorator for examples.
    Args:
      - tool: The Tool instance to register
  remove_tool(self, name: str) -> None
    Remove a tool from the server.
    Args:
      - name: The name of the tool to remove
    Raises:
      - NotFoundError: If the tool is not found
  tool(self, name_or_fn: AnyFunction) -> FunctionTool
  tool(self, name_or_fn: str | None = None) -> Callable[[AnyFunction], FunctionTool]
```

---

TITLE: FastMCP: Call Parent Tool with `forward()` (Renamed Arguments)
DESCRIPTION: Illustrates using `forward()` when the `transform_fn` uses different argument names than the parent tool. FastMCP's `ArgTransform` configuration automatically maps the renamed arguments back to the parent tool's original arguments, simplifying the call to `forward()`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_12

LANGUAGE: Python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward

mcp = FastMCP()

@mcp.tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

async def ensure_positive(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    return await forward(a=a, b=b)

new_tool = Tool.from_tool(
    add,
    transform_fn=ensure_positive,
    transform_args={
        "x": ArgTransform(name="a"),
        "y": ArgTransform(name="b"),
    }
)

mcp.add_tool(new_tool)
```

---

TITLE: Install FastMCP Server in Claude Desktop App
DESCRIPTION: The `install` command sets up a FastMCP server for use within the Claude desktop application. It's crucial to explicitly specify all dependencies using `--with` or `--with-editable` due to Claude's isolated environment. `uv` must be installed globally for dependency management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_12

LANGUAGE: bash
CODE:

```
fastmcp install server.py
```

---

TITLE: Create OAuth Client Provider for MCP Server
DESCRIPTION: This function creates an OAuthClientProvider instance, intended for use with an `httpx.AsyncClient` or FastMCP client. It configures the OAuth client with the MCP server URL, requested scopes, client name, and optional token storage directory and additional metadata.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-auth-oauth.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
OAuth(mcp_url: str, scopes: str | list[str] | None = None, client_name: str = 'FastMCP Client', token_storage_cache_dir: Path | None = None, additional_client_metadata: dict[str, Any] | None = None) -> _MCPOAuthClientProvider
  mcp_url: Full URL to the MCP endpoint (e.g. "http://host/mcp/sse/")
  scopes: OAuth scopes to request. Can be a string or list of strings.
  client_name: Name for this client during registration.
  token_storage_cache_dir: Directory for FileTokenStorage.
  additional_client_metadata: Extra fields for OAuthClientMetadata.
  Returns: OAuthClientProvider
```

---

TITLE: Creating FastMCP Proxy Server from Client
DESCRIPTION: This class method constructs a FastMCP proxy server directly from an existing FastMCP client instance, simplifying the setup of proxy functionalities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_19

LANGUAGE: python
CODE:

```
from_client(cls, client: Client[ClientTransportT], **settings: Any) -> FastMCPProxy
```

---

TITLE: Authenticate Anthropic Client for FastMCP Server Access
DESCRIPTION: This example demonstrates how to configure an Anthropic client to authenticate with a FastMCP server. It shows how to include the `authorization_token` within the `mcp_servers` configuration when making a request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_9

LANGUAGE: python
CODE:

```
import anthropic
from rich import print

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

# Your access token (replace with your actual token)
access_token = 'your-access-token'

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Roll a few dice!"}],
    mcp_servers=[
        {
            "type": "url",
            "url": f"{url}/mcp/",
            "name": "dice-server",
            "authorization_token": access_token
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)
```

---

TITLE: Create and Sync Python Environment with uv
DESCRIPTION: This command uses `uv` to create and synchronize the Python virtual environment for the FastMCP project. It installs all necessary dependencies, including development tools, as defined in the project's configuration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_16

LANGUAGE: bash
CODE:

```
uv sync
```

---

TITLE: Setting Up the FastMCP Development Environment
DESCRIPTION: Provides the initial command to synchronize dependencies and run all pre-commit hooks, ensuring the environment is correctly set up for development.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/AGENTS.md#_snippet_1

LANGUAGE: bash
CODE:

```
uv sync && uv run pre-commit run --all-files
```

---

TITLE: Run FastMCP Development Server
DESCRIPTION: This command starts a FastMCP development server. The `-e .` flag enables editable mode, allowing changes in the current directory to be reflected without restarting. The `--with` flags specify additional Python packages like `pandas` and `matplotlib` to be included in the server's environment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_11

LANGUAGE: bash
CODE:

```
fastmcp dev server.py -e . --with pandas --with matplotlib
```

---

TITLE: Disabling Original Tool After Transformation
DESCRIPTION: Illustrates how to disable the original tool after creating a transformed version to prevent confusion for LLM clients, ensuring only the enhanced tool is visible.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool

mcp = FastMCP()

# The original, generic tool
@mcp.tool
def search(query: str, category: str = "all") -> list[dict]:
    ...

# Create a more domain-specific version
product_search_tool = Tool.from_tool(search, ...)
mcp.add_tool(product_search_tool)

# Disable the original tool
search.disable()
```

---

TITLE: FastMCP Context Properties (`ctx`)
DESCRIPTION: Lists available properties on the `ctx` object for accessing request metadata and session identifiers within FastMCP tools.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_11

LANGUAGE: APIDOC
CODE:

```
ctx.request_id -> str: Get the unique ID for the current MCP request
ctx.client_id -> str | None: Get the ID of the client making the request, if provided during initialization
ctx.session_id -> str | None: Get the MCP session ID for session-based data sharing (HTTP transports only)
```

---

TITLE: Register FastMCP Components with Server
DESCRIPTION: Demonstrates how to instantiate a class inheriting from `MCPMixin` and register all its decorated methods (tools, resources, prompts) with a `FastMCP` server, including the use of an optional prefix to avoid name collisions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/src/fastmcp/contrib/mcp_mixin/README.md#_snippet_1

LANGUAGE: python
CODE:

```
mcp_server = FastMCP()
component = MyComponent()

# Register all decorated methods with a prefix
# Useful if you will have multiple instantiated objects of the same class
# and want to avoid name collisions.
component.register_all(mcp_server, prefix="my_comp")

# Register without a prefix
# component.register_all(mcp_server)

# Now 'my_comp_my_tool' tool and 'my_comp+component://data' resource are registered (if prefix used)
# Or 'my_tool' and 'component://data' are registered (if no prefix used)
```

---

TITLE: Making an Argument Required in fastmcp
DESCRIPTION: This snippet demonstrates how to make an optional argument required using `ArgTransform` within the `transform_args` dictionary. Setting `required=True` ensures the argument must be provided, even if it was originally optional.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_9

LANGUAGE: python
CODE:

```
transform_args = {
    'user_id': ArgTransform(
        required=True,
    )
}
```

---

TITLE: Create FastMCP Server with Dice Rolling Tool (Python)
DESCRIPTION: This Python script defines a FastMCP server named 'Dice Roller' and exposes a single tool, `roll_dice`, which simulates rolling a specified number of 6-sided dice. The server can be run locally to make this tool available for use by clients like the Gemini SDK.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_0

LANGUAGE: python
CODE:

```
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: Setting Dynamic Roots with fastmcp Client Callback
DESCRIPTION: Illustrates how to provide roots dynamically using an asynchronous callback function during fastmcp client initialization. The callback allows the client to respond to server requests for roots with context-specific paths, such as a request ID.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/roots.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.roots import RequestContext

async def roots_callback(context: RequestContext) -> list[str]:
    print(f"Server requested roots (Request ID: {context.request_id})")
    return ["/path/to/root1", "/path/to/root2"]

client = Client(
    "my_mcp_server.py",
    roots=roots_callback
)
```

---

TITLE: Install pre-commit Hooks for FastMCP
DESCRIPTION: This command installs the `pre-commit` hooks for the FastMCP repository using `uv`. These hooks automate code formatting, linting, and type-checking, ensuring consistent code quality before commits.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_19

LANGUAGE: bash
CODE:

```
uv run pre-commit install
```

---

TITLE: APIDOC: FastMCPProxy Class
DESCRIPTION: A FastMCP server that acts as a proxy to a remote MCP-compliant server. It uses specialized managers that fulfill requests via an HTTP client.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-proxy.mdx#_snippet_7

LANGUAGE: APIDOC
CODE:

```
FastMCPProxy:
  description: A FastMCP server that acts as a proxy to a remote MCP-compliant server.
  It uses specialized managers that fulfill requests via an HTTP client.
```

---

TITLE: Install Dependencies and Run FastMCP Server
DESCRIPTION: This snippet provides commands to install the necessary dependencies using `uv pip` and then run the FastMCP server. The server will then be ready to handle AT Protocol interactions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/examples/atproto_mcp/README.md#_snippet_1

LANGUAGE: bash
CODE:

```
# Install dependencies
uv pip install -e .

# Run the server
uv run atproto-mcp
```

---

TITLE: Retrieve HTTP Request Object in fastmcp.server.dependencies
DESCRIPTION: This function retrieves the current HTTP request object.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-dependencies.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
get_http_request() -> Request
```

---

TITLE: Define Tool Parameters with Python Built-in Scalar Types in FastMCP
DESCRIPTION: This snippet demonstrates how to use Python's fundamental scalar types (string, integer, float, boolean) as parameters in a FastMCP tool. FastMCP automatically handles type validation and coercion, converting client inputs to the expected Python types, even if initially provided as strings.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_19

LANGUAGE: python
CODE:

```
@mcp.tool
def process_values(
    name: str,             # Text data
    count: int,            # Integer numbers
    amount: float,         # Floating point numbers
    enabled: bool          # Boolean values (True/False)
):
    """Process various value types."""
    # Implementation...
```

---

TITLE: Access Current Request and Client Information
DESCRIPTION: Shows how to retrieve metadata about the current FastMCP request and client using properties available on the `ctx` object, such as `request_id` and `client_id`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_10

LANGUAGE: python
CODE:

```
@mcp.tool
async def request_info(ctx: Context) -> dict:
    """Return information about the current request."""
    return {
        "request_id": ctx.request_id,
        "client_id": ctx.client_id or "Unknown client"
    }
```

---

TITLE: Use Built-in FastMCP Timing Middleware
DESCRIPTION: Shows how to integrate FastMCP's pre-built `TimingMiddleware` and `DetailedTimingMiddleware` for performance monitoring. This snippet demonstrates adding these ready-to-use middleware components to your server for basic and granular timing capabilities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_12

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware.timing import (
    TimingMiddleware,
    DetailedTimingMiddleware
)

# Basic timing for all requests
mcp.add_middleware(TimingMiddleware())

# Detailed per-operation timing (tools, resources, prompts)
mcp.add_middleware(DetailedTimingMiddleware())
```

---

TITLE: Create Custom Logging Middleware in FastMCP
DESCRIPTION: Demonstrates how to create a custom middleware by subclassing `Middleware` and overriding the `on_message` hook. This example logs MCP operations before and after execution, illustrating the basic structure for pre- and post-processing logic within middleware.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

class LoggingMiddleware(Middleware):
    """Middleware that logs all MCP operations."""

    async def on_message(self, context: MiddlewareContext, call_next):
        """Called for all MCP messages."""
        print(f"Processing {context.method} from {context.source}")

        result = await call_next(context)

        print(f"Completed {context.method}")
        return result

# Add middleware to your server
mcp = FastMCP("MyServer")
mcp.add_middleware(LoggingMiddleware())
```

---

TITLE: Interacting with FastMCP Server via File-Based Client in Python
DESCRIPTION: Demonstrates how a FastMCP client can connect to a server by referencing its file path. This allows a separate client script to call tools on a running server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_4

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

client = Client("my_server.py")

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))
```

---

TITLE: FastMCP Client Transports API Overview
DESCRIPTION: Provides an overview of the core `ClientTransport` classes available in FastMCP, detailing their inference rules and server compatibility.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.StreamableHttpTransport
  Description: Recommended transport for web-based deployments, providing efficient bidirectional communication over HTTP.
  Inferred From: URLs starting with http:// or https:// (default for HTTP URLs since v2.3.0) that do not contain /sse/ in the path
  Server Compatibility: Works with FastMCP servers running in http mode

Class: fastmcp.client.transports.SSETransport
  Description: Allows servers to push data to clients over HTTP connections. While still supported, Streamable HTTP is now recommended.
  Inferred From: HTTP URLs containing /sse/ in the path
  Server Compatibility: Works with FastMCP servers running in sse mode
```

---

TITLE: Install FastMCP Server using CLI `fastmcp install`
DESCRIPTION: This snippet shows how to install a FastMCP server in Claude Desktop using the `fastmcp install` command. It covers basic installation and specifying the server object, automating configuration and dependency management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
fastmcp install server.py
fastmcp install server.py:mcp

fastmcp install server.py:my_custom_server
```

---

TITLE: Safely Accessing HTTP Headers in FastMCP
DESCRIPTION: This example illustrates how to retrieve HTTP headers safely using the `get_http_headers()` helper function. This function is designed to prevent errors by returning an empty dictionary if no request context is available. It demonstrates how to check for specific headers like 'authorization' and 'content-type', and notes that it excludes problematic headers by default.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/http-requests.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

mcp = FastMCP(name="Headers Demo")

@mcp.tool
async def safe_header_info() -> dict:
    """Safely get header information without raising errors."""
    # Get headers (returns empty dict if no request context)
    headers = get_http_headers()

    # Get authorization header
    auth_header = headers.get("authorization", "")
    is_bearer = auth_header.startswith("Bearer ")

    return {
        "user_agent": headers.get("user-agent", "Unknown"),
        "content_type": headers.get("content-type", "Unknown"),
        "has_auth": bool(auth_header),
        "auth_type": "Bearer" if is_bearer else "Other" if auth_header else "None",
        "headers_count": len(headers)
    }
```

---

TITLE: Setting Global Timeouts for FastMCP API Requests
DESCRIPTION: Demonstrates how to configure a global timeout for all API requests made by the FastMCP server, ensuring requests do not hang indefinitely.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_15

LANGUAGE: python
CODE:

```
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=api_client,
    timeout=30.0  # 30 second timeout for all requests
)
```

---

TITLE: FastMCP Client Session Management with keep_alive=True
DESCRIPTION: Demonstrates how `keep_alive=True` (default) maintains the MCP server subprocess and session across multiple client context manager exits and re-entries, improving performance. It also shows how to manually close the session.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_8

LANGUAGE: python
CODE:

```
from fastmcp import Client

# Client with keep_alive=True (default)
client = Client("my_mcp_server.py")

async def example():
    # First session
    async with client:
        await client.ping()

    # Second session - uses the same subprocess
    async with client:
        await client.ping()

    # Manually close the session
    await client.close()

    # Third session - will start a new subprocess
    async with client:
        await client.ping()

asyncio.run(example())
```

---

TITLE: List Resource Templates with Python MCP Client
DESCRIPTION: Shows how to use `client.list_resource_templates()` to fetch available resource templates from an MCP server. It iterates through the `ResourceTemplate` objects, displaying their URI template, name, and description.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/resources.mdx#_snippet_1

LANGUAGE: python
CODE:

```
async with client:
    templates = await client.list_resource_templates()
    # templates -> list[mcp.types.ResourceTemplate]

    for template in templates:
        print(f"Template URI: {template.uriTemplate}")
        print(f"Name: {template.name}")
        print(f"Description: {template.description}")
```

---

TITLE: FastMCP CLI Commands Overview
DESCRIPTION: A summary of the main FastMCP CLI commands, their purpose, and dependency management characteristics.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
Command: run
  Purpose: Run a FastMCP server directly
  Dependency Management: Uses your current environment; you are responsible for ensuring all dependencies are available

Command: dev
  Purpose: Run a server with the MCP Inspector for testing
  Dependency Management: Creates an isolated environment; dependencies must be explicitly specified with --with and/or --with-editable

Command: install
  Purpose: Install a server in the Claude desktop app
  Dependency Management: Creates an isolated environment; dependencies must be explicitly specified with --with and/or --with-editable

Command: inspect
  Purpose: Generate a JSON report about a FastMCP server
  Dependency Management: Uses your current environment; you are responsible for ensuring all dependencies are available

Command: version
  Purpose: Display version information
  Dependency Management: N/A
```

---

TITLE: Manual Error Checking for FastMCP Tool Results
DESCRIPTION: Shows how to use `client.call_tool_mcp()` to get the raw MCP protocol object, allowing manual checking of the `isError` flag for more granular control over error handling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/tools.mdx#_snippet_6

LANGUAGE: python
CODE:

```
async with client:
    result = await client.call_tool_mcp("potentially_failing_tool", {"param": "value"})
    # result -> mcp.types.CallToolResult

    if result.isError:
        print(f"Tool failed: {result.content}")
    else:
        print(f"Tool succeeded: {result.content}")
```

---

TITLE: API Documentation for ToolManager Class
DESCRIPTION: Detailed API reference for the `ToolManager` class, which is responsible for managing FastMCP tools. It includes methods for adding tools from functions or existing tool objects, removing tools, and integrating tools from mounted servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-tools-tool_manager.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
ToolManager:
  Manages FastMCP tools.
  Methods:
    mount(self, server: MountedServer) -> None
      Adds a mounted server as a source for tools.
    add_tool_from_fn(self, fn: Callable[..., Any], name: str | None = None, description: str | None = None, tags: set[str] | None = None, annotations: ToolAnnotations | None = None, serializer: Callable[[Any], str] | None = None, exclude_args: list[str] | None = None) -> Tool
      Add a tool to the server.
    add_tool(self, tool: Tool) -> Tool
      Register a tool with the server.
    remove_tool(self, key: str) -> None
      Remove a tool from the server.
      Args:
        key: The key of the tool to remove
      Raises:
        NotFoundError: If the tool is not found
```

---

TITLE: Configure FastMCP Server with Bearer Authentication Provider
DESCRIPTION: This code shows how to initialize a `BearerAuthProvider` using the previously generated public key and audience. The configured authentication provider is then passed to the `FastMCP` application instance to secure the server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)
```

---

TITLE: Run FastMCP Server Script
DESCRIPTION: Executes the api_server.py script to start the FastMCP server, making the defined API endpoints accessible.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
python api_server.py
```

---

TITLE: Tagging FastMCP Components for Filtering in Python
DESCRIPTION: Shows how to apply tags to FastMCP tools using the `tags` parameter in the decorator, enabling selective exposure of components based on include/exclude tag sets.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_6

LANGUAGE: python
CODE:

```
@mcp.tool(tags={"public", "utility"})
def public_tool() -> str:
    return "This tool is public"

@mcp.tool(tags={"internal", "admin"})
def admin_tool() -> str:
    return "This tool is for admins only"
```

---

TITLE: Add Resource Template to FastMCP Server
DESCRIPTION: Adds a pre-existing ResourceTemplate instance to the FastMCP server, allowing dynamic resource generation based on parameters.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_6

LANGUAGE: APIDOC
CODE:

```
add_template(self, template: ResourceTemplate) -> None

Args:
- template: A ResourceTemplate instance to add.
```

---

TITLE: Add a Single Middleware to FastMCP Server
DESCRIPTION: Illustrates the straightforward method for adding a single instance of middleware to a FastMCP server. This snippet shows the basic syntax for integrating a custom or built-in middleware into your server's processing pipeline.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_8

LANGUAGE: python
CODE:

```
mcp = FastMCP("MyServer")
mcp.add_middleware(LoggingMiddleware())
```

---

TITLE: FastMCP Path Parameter Validation
DESCRIPTION: Illustrates how FastMCP handles path parameters, filtering out `None` values and raising errors for missing required parameters. This ensures REST API compliance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_12

LANGUAGE: python
CODE:

```
# ✅ This works
await client.call_tool("get_user", {"user_id": 123})

# ❌ This raises: "Missing required path parameters: {'user_id'}"
await client.call_tool("get_user", {"user_id": None})
```

---

TITLE: Generate RSA Key Pair and Access Token for FastMCP Server
DESCRIPTION: This snippet demonstrates how to generate an RSA key pair using `RSAKeyPair.generate()` and then create an access token with a specified audience. This token will be used for server authentication.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp.server.auth.providers.bearer import RSAKeyPair

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")
```

---

TITLE: Creating a Basic FastMCP Proxy
DESCRIPTION: This snippet demonstrates the simplest way to create a FastMCP proxy using the `FastMCP.as_proxy()` class method. It shows how to provide the backend server's location as a file path or an existing Client instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Provide the backend in any form accepted by Client
proxy_server = FastMCP.as_proxy(
    "backend_server.py",  # Could also be a FastMCP instance, config dict, or a remote URL
    name="MyProxyServer"  # Optional settings for the proxy
)

# Or create the Client yourself for custom configuration
backend_client = Client("backend_server.py")
proxy_from_client = FastMCP.as_proxy(backend_client)
```

---

TITLE: Configure FastMCP Logging Settings
DESCRIPTION: Configures the logging settings for FastMCP, allowing specification of the log level and enabling rich tracebacks. This function can apply settings to a specific logger or the default FastMCP logger.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-utilities-logging.mdx#_snippet_1

LANGUAGE: python
CODE:

```
configure_logging(level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] | int = 'INFO', logger: logging.Logger | None = None, enable_rich_tracebacks: bool = True) -> None
```

LANGUAGE: APIDOC
CODE:

```
configure_logging(level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] | int = 'INFO', logger: logging.Logger | None = None, enable_rich_tracebacks: bool = True) -> None
  level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] | int = 'INFO'
    The log level to use.
  logger: logging.Logger | None = None
    The logger to configure.
  enable_rich_tracebacks: bool = True
```

---

TITLE: Verify FastMCP Installation
DESCRIPTION: Run this command to confirm that FastMCP has been installed correctly. It displays the installed FastMCP version, underlying MCP version, Python version, and platform details.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp version
```

---

TITLE: Create Server-Sent Events (SSE) Application (Python)
DESCRIPTION: Generates a Starlette application configured for Server-Sent Events (SSE). It integrates with a FastMCP server instance to handle real-time message and SSE connections, with optional authentication, debug mode, and custom routing or middleware.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-http.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
create_sse_app(server: FastMCP[LifespanResultT], message_path: str, sse_path: str, auth: OAuthProvider | None = None, debug: bool = False, routes: list[BaseRoute] | None = None, middleware: list[Middleware] | None = None) -> StarletteWithLifespan
  server: The FastMCP server instance.
  message_path: Path for SSE messages.
  sse_path: Path for SSE connections.
  auth: Optional auth provider.
  debug: Whether to enable debug mode.
  routes: Optional list of custom routes.
  middleware: Optional list of middleware.
  Returns: A Starlette application with RequestContextMiddleware.
```

---

TITLE: Register Static and Predefined FastMCP Resources
DESCRIPTION: Demonstrates how to register various types of static or predefined resources using `mcp.add_resource()` and concrete `Resource` subclasses like `FileResource`, `TextResource`, and `DirectoryResource`. This method is suitable for content that doesn't require dynamic generation via a Python function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.resources import FileResource, TextResource, DirectoryResource

mcp = FastMCP(name="DataServer")

# 1. Exposing a static file directly
readme_path = Path("./README.md").resolve()
if readme_path.exists():
    # Use a file:// URI scheme
    readme_resource = FileResource(
        uri=f"file://{readme_path.as_posix()}",
        path=readme_path, # Path to the actual file
        name="README File",
        description="The project's README.",
        mime_type="text/markdown",
        tags={"documentation"}
    )
    mcp.add_resource(readme_resource)

# 2. Exposing simple, predefined text
notice_resource = TextResource(
    uri="resource://notice",
    name="Important Notice",
    text="System maintenance scheduled for Sunday.",
    tags={"notification"}
)
mcp.add_resource(notice_resource)

# 3. Using a custom key different from the URI
special_resource = TextResource(
    uri="resource://common-notice",
    name="Special Notice",
    text="This is a special notice with a custom storage key.",
)
mcp.add_resource(special_resource, key="resource://custom-key")

# 4. Exposing a directory listing
data_dir_path = Path("./app_data").resolve()
if data_dir_path.is_dir():
    data_listing_resource = DirectoryResource(
        uri="resource://data-files",
        path=data_dir_path, # Path to the directory
        name="Data Directory Listing",
        description="Lists files available in the data directory.",
        recursive=False # Set to True to list subdirectories
    )
    mcp.add_resource(data_listing_resource) # Returns JSON list of files
```

---

TITLE: FastMCP: Using Wildcard Parameters in Resource Templates
DESCRIPTION: Explains FastMCP's extension for wildcard parameters (`{param*}`) in resource templates, which allows matching multiple URI path segments including slashes. This contrasts with standard parameters (`{param}`) that only match a single segment, providing greater flexibility for capturing complex paths.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_11

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")


# Standard parameter only matches one segment
@mcp.resource("files://{filename}")
def get_file(filename: str) -> str:
    """Retrieves a file by name."""
    # Will only match files://<single-segment>
    return f"File content for: {filename}"


# Wildcard parameter can match multiple segments
@mcp.resource("path://{filepath*}")
def get_path_content(filepath: str) -> str:
    """Retrieves content at a specific path."""
    # Can match path://docs/server/resources.mdx
    return f"Content at path: {filepath}"
```

---

TITLE: FastMCP Array Parameter Serialization
DESCRIPTION: Explains how FastMCP serializes array parameters for both query and path segments according to OpenAPI specifications, including the effect of the `explode` parameter for query arrays.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_13

LANGUAGE: APIDOC
CODE:

```
# Query array with explode=true (default)
# ?tags=red&tags=blue&tags=green

# Query array with explode=false
# ?tags=red,blue,green

# Path array (always comma-separated)
# /items/red,blue,green
```

---

TITLE: Prompt Class API Reference
DESCRIPTION: Defines the `Prompt` class, a template for prompts, including methods for converting to MCP prompts and creating prompts from functions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-prompts-prompt.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
Prompt:
  Methods:
    to_mcp_prompt(self, **overrides: Any) -> MCPPrompt
      Description: Convert the prompt to an MCP prompt.
    from_function(fn: Callable[..., PromptResult | Awaitable[PromptResult]], name: str | None = None, description: str | None = None, tags: set[str] | None = None, enabled: bool | None = None) -> FunctionPrompt
      Description: Create a Prompt from a function.
      Returns:
        - A string (converted to a message)
        - A Message object
        - A dict (converted to a message)
        - A sequence of any of the above
```

---

TITLE: Customize FastMCP Component Names with mcp_names
DESCRIPTION: This snippet illustrates how to override FastMCP's default component naming strategy. By providing an `mcp_names` dictionary, you can map specific `operationId` values from your OpenAPI spec to custom, more descriptive names for the generated MCP components. The provided names are automatically slugified and truncated.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP.from_openapi(
    ...
    mcp_names={
        "list_users__with_pagination": "user_list",
        "create_user__admin_required": "create_user",
        "get_user_details__admin_required": "user_detail",
    }
)
```

---

TITLE: Handling Raw Binary Data with Bytes Parameters in FastMCP
DESCRIPTION: Illustrates how to accept raw binary data directly by annotating a tool parameter with the `bytes` type. FastMCP converts raw string inputs directly to bytes and validates that the input can be properly represented as binary data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_25

LANGUAGE: python
CODE:

```
@mcp.tool
def process_binary(data: bytes):
    """Process binary data directly.

    The client can send a binary string, which will be
    converted directly to bytes.
    """
    # Implementation using binary data
    data_length = len(data)
    # ...
```

---

TITLE: Install Anthropic Python SDK
DESCRIPTION: This command installs the official Anthropic Python SDK, which is required to interact with the Anthropic Messages API and call your deployed FastMCP server. It is not included with FastMCP and needs to be installed separately.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
pip install anthropic
```

---

TITLE: APIDOC: BearerAuthProvider Class
DESCRIPTION: Simple JWT Bearer Token validator for hosted MCP servers. Uses RS256 asymmetric encryption. Supports either static public key or JWKS URI for key rotation. Note that this provider DOES NOT permit client registration or revocation, or any OAuth flows. It is intended to be used with a control plane that manages clients and tokens.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-auth-providers-bearer.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
BearerAuthProvider:
  Description: Simple JWT Bearer Token validator for hosted MCP servers. Uses RS256 asymmetric encryption. Supports either static public key or JWKS URI for key rotation. Note that this provider DOES NOT permit client registration or revocation, or any OAuth flows. It is intended to be used with a control plane that manages clients and tokens.
```

---

TITLE: FastMCP: Call Parent Tool with `forward_raw()`
DESCRIPTION: Shows how to use `forward_raw()` to directly call the parent tool, bypassing all argument transformation. This function requires passing the original argument names of the parent tool, even if the `transform_fn` uses different names, and is typically used for complex argument manipulations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_13

LANGUAGE: Python
CODE:

```
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import forward

mcp = FastMCP()

@mcp.tool
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y

async def ensure_positive(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    return await forward_raw(x=a, y=b)

new_tool = Tool.from_tool(
    add,
    transform_fn=ensure_positive,
    transform_args={
        "x": ArgTransform(name="a"),
        "y": ArgTransform(name="b"),
    }
)

mcp.add_tool(new_tool)
```

---

TITLE: Common FastMCP Resource Classes Overview
DESCRIPTION: Provides a summary of common `fastmcp.resources` classes available for registering predefined content. These classes simplify exposing various data types like text, binary, files, HTTP content, and directory listings without custom functions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_8

LANGUAGE: APIDOC
CODE:

```
TextResource: For simple string content.
BinaryResource: For raw `bytes` content.
FileResource: Reads content from a local file path. Handles text/binary modes and lazy reading.
HttpResource: Fetches content from an HTTP(S) URL (requires `httpx`).
DirectoryResource: Lists files in a local directory (returns JSON).
(FunctionResource: Internal class used by `@mcp.resource`).
```

---

TITLE: Create Starlette SSE Application
DESCRIPTION: Creates a Starlette application specifically configured for Server-Sent Events (SSE). This method allows defining the endpoint paths and applying custom ASGI middleware.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_12

LANGUAGE: APIDOC
CODE:

```
sse_app(self, path: str | None = None, message_path: str | None = None, middleware: list[ASGIMiddleware] | None = None) -> StarletteWithLifespan
	path: The path to the SSE endpoint
	message_path: The path to the message endpoint
	middleware: A list of middleware to apply to the app
```

---

TITLE: fastmcp.settings Module API Reference
DESCRIPTION: Comprehensive API documentation for the fastmcp.settings module, outlining the structure and usage of its core classes and their associated methods. This includes details on environment variable handling, custom source configuration, and logging setup.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-settings.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
Class: ExtendedEnvSettingsSource
  Description: A special EnvSettingsSource that allows for multiple env var prefixes to be used. Raises a deprecation warning if the old `FASTMCP_SERVER_` prefix is used.
  Methods:
    get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]

Class: ExtendedSettingsConfigDict

Class: Settings
  Description: FastMCP settings.
  Methods:
    settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource, env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource, file_secret_settings: PydanticBaseSettingsSource) -> tuple[PydanticBaseSettingsSource, ...]
    settings(self) -> Self
      Description: This property is for backwards compatibility with FastMCP < 2.8.0, which accessed fastmcp.settings.settings
    setup_logging(self) -> Self
      Description: Finalize the settings.
```

---

TITLE: Configure FastMCP Resource Prefix Format Globally in Python
DESCRIPTION: This Python code snippet shows how to set the default resource prefix format for all FastMCP servers globally by modifying fastmcp.settings.resource_prefix_format. This configuration impacts all subsequent server instances created within the application.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_5

LANGUAGE: python
CODE:

```
import fastmcp
fastmcp.settings.resource_prefix_format = "protocol"
```

---

TITLE: Add Prompt to FastMCP Server
DESCRIPTION: Adds a pre-existing Prompt instance to the FastMCP server, making it available for use.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_9

LANGUAGE: APIDOC
CODE:

```
add_prompt(self, prompt: Prompt) -> None

Args:
- prompt: A Prompt instance to add.
```

---

TITLE: Configure Middleware with FastMCP Server Composition
DESCRIPTION: Shows how middleware behaves when using server composition with `mount` or `import_server`. This example illustrates a parent server with its own middleware and a child server with distinct middleware, demonstrating how requests flow through layered middleware architectures.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_10

LANGUAGE: python
CODE:

```
# Parent server with middleware
parent = FastMCP("Parent")
parent.add_middleware(AuthenticationMiddleware("token"))

# Child server with its own middleware
child = FastMCP("Child")
child.add_middleware(LoggingMiddleware())

@child.tool
def child_tool() -> str:
    return "from child"

# Mount the child server
parent.mount(child, prefix="child")
```

---

TITLE: Create PromptMessage with Message Function
DESCRIPTION: Provides a user-friendly constructor for `PromptMessage`, allowing flexible content and role assignment for prompt messages.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-prompts-prompt.mdx#_snippet_0

LANGUAGE: python
CODE:

```
Message(content: str | MCPContent, role: Role | None = None, **kwargs: Any) -> PromptMessage
```

---

TITLE: Installing Local FastMCP Development Version into Server Project
DESCRIPTION: This set of commands outlines the process for installing a local development version of FastMCP into a separate FastMCP server project. It includes steps for uninstalling previous versions, cleaning build artifacts, reinstalling with `uv pip`, and verifying the installation using `pip show`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/Windows_Notes.md#_snippet_2

LANGUAGE: Bash
CODE:

```
# First uninstall
uv pip uninstall fastmcp

# Clean any build artifacts in your fastmcp directory
cd C:\path\to\fastmcp
del /s /q *.egg-info

# Then reinstall in your weather project
cd C:\path\to\new\fastmcp_server
uv pip install --no-cache-dir -e C:\Users\justj\PycharmProjects\fastmcp

# Check that it installed properly and has the correct git hash
pip show fastmcp
```

---

TITLE: File-Based OAuth Token Storage
DESCRIPTION: This class implements a file-based token storage mechanism for OAuth credentials and tokens. It adheres to the `mcp.client.auth.TokenStorage` protocol and ensures token isolation by tying each instance to a specific server URL.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-auth-oauth.mdx#_snippet_4

LANGUAGE: APIDOC
CODE:

```
FileTokenStorage:
  File-based token storage implementation for OAuth credentials and tokens.
  Implements the mcp.client.auth.TokenStorage protocol.
  Each instance is tied to a specific server URL for proper token isolation.
```

---

TITLE: Create FastMCP Proxy from Single Server Configuration
DESCRIPTION: This snippet demonstrates how to initialize a FastMCP proxy using a configuration dictionary for a single server. It defines the server's URL and transport, then creates and runs the proxy for local access via stdio.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_5

LANGUAGE: python
CODE:

```
config = {
    "mcpServers": {
        "default": {  # For single server configs, 'default' is commonly used
            "url": "https://example.com/mcp",
            "transport": "http"
        }
    }
}

# Create a proxy to the configured server
proxy = FastMCP.as_proxy(config, name="Config-Based Proxy")

# Run the proxy with stdio transport for local access
if __name__ == "__main__":
    proxy.run()
```

---

TITLE: Manually Run All pre-commit Checks
DESCRIPTION: This snippet provides commands to manually execute all configured `pre-commit` checks across all files in the FastMCP repository. This is useful for verifying code quality before committing changes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_20

LANGUAGE: bash
CODE:

```
pre-commit run --all-files
uv run pre-commit run --all-files
```

---

TITLE: Configuring Duplicate Prompt Handling in FastMCP Server (Python)
DESCRIPTION: Explains how to configure the FastMCP server's behavior when attempting to register multiple prompts with the same name. The 'on_duplicate_prompts' setting during 'FastMCP' initialization controls this behavior, with options like "warn", "error", "replace", and "ignore".
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_11

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(
    name="PromptServer",
    on_duplicate_prompts="error"  # Raise an error if a prompt name is duplicated
)

@mcp.prompt
def greeting(): return "Hello, how can I help you today?"

# This registration attempt will raise a ValueError because
# "greeting" is already registered and the behavior is "error".
# @mcp.prompt
# def greeting(): return "Hi there! What can I do for you?"
```

---

TITLE: Set Gemini API Key Environment Variable (Bash)
DESCRIPTION: This Bash command sets the `GEMINI_API_KEY` environment variable. This key is required by the Google Generative AI SDK for authenticating API requests to the Gemini service.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
export GEMINI_API_KEY="your-api-key"
```

---

TITLE: FastMCP: Excluding Routes by Pattern or Tag
DESCRIPTION: This snippet demonstrates how to explicitly exclude routes from the FastMCP server. It uses `RouteMap` with `MCPType.EXCLUDE` to prevent routes matching the `/admin/` pattern or those tagged 'internal' from being processed. This is useful for removing sensitive or internal API endpoints.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

mcp = FastMCP.from_openapi(
    ...,
    route_maps=[
        RouteMap(pattern=r"^/admin/.*", mcp_type=MCPType.EXCLUDE),
        RouteMap(tags={"internal"}, mcp_type=MCPType.EXCLUDE),
    ],
)
```

---

TITLE: Example FastMCP Server Code
DESCRIPTION: A basic Python script defining a FastMCP server with a single tool. This example demonstrates how the `if __name__ == "__main__"` block is ignored when running with `fastmcp run`, allowing the CLI to override transport settings.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_5

LANGUAGE: python
CODE:

```
# server.py
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    # This is ignored when using `fastmcp run`!
    mcp.run(transport="stdio")
```

---

TITLE: Handle Unauthorized Client Access Error
DESCRIPTION: This snippet illustrates the `401 (Unauthorized)` error response received when an unauthenticated client attempts to access a FastMCP server secured with bearer token authentication. It highlights the importance of client authentication for successful API calls.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_9

LANGUAGE: python
CODE:

```
pythonAPIStatusError: Error code: 424 - {
    "error": {
        "message": "Error retrieving tool list from MCP server: 'dice_server'. Http status code: 401 (Unauthorized)",
        "type": "external_connector_error",
        "param": "tools",
        "code": "http_error"
    }
}
```

---

TITLE: Configure Advanced Logging Middleware in FastMCP
DESCRIPTION: This Python example illustrates how to integrate FastMCP's built-in `LoggingMiddleware` and `StructuredLoggingMiddleware` to enable human-readable or JSON-structured request/response logging, including payload support and size limits.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/middleware.mdx#_snippet_14

LANGUAGE: python
CODE:

```
from fastmcp.server.middleware.logging import (
    LoggingMiddleware,
    StructuredLoggingMiddleware
)

# Human-readable logging with payload support
mcp.add_middleware(LoggingMiddleware(
    include_payloads=True,
    max_payload_length=1000
))

# JSON-structured logging for log aggregation tools
mcp.add_middleware(StructuredLoggingMiddleware(include_payloads=True))
```

---

TITLE: Example of automatic argument serialization (Python)
DESCRIPTION: Further examples of how FastMCP automatically serializes nested dictionaries and lists into JSON strings when used as prompt arguments, while simple strings remain unchanged, ensuring consistent data transfer.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_4

LANGUAGE: python
CODE:

```
async with client:
    result = await client.get_prompt("data_analysis", {
        # These will be automatically serialized to JSON strings:
        "config": {
            "format": "csv",
            "include_headers": True,
            "delimiter": ","
        },
        "filters": [
            {"field": "age", "operator": ">", "value": 18},
            {"field": "status", "operator": "==", "value": "active"}
        ],
        # This remains a string:
        "report_title": "Monthly Analytics Report"
    })
```

---

TITLE: Clear Tokens for Specific Server (Python)
DESCRIPTION: This snippet demonstrates how to clear authentication tokens for a specific server URL. It requires instantiating `FileTokenStorage` with the server's URL and then calling the asynchronous `clear()` method on the instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp.client.auth.oauth import FileTokenStorage

storage = FileTokenStorage(server_url="https://fastmcp.cloud/mcp")
await storage.clear()
```

---

TITLE: SSE Transport: Authentication with Headers
DESCRIPTION: Illustrates how to include custom HTTP headers, such as an `Authorization` header, when instantiating `SSETransport` to authenticate requests to a FastMCP server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import SSETransport

# Create SSE transport with authentication headers
transport = SSETransport(
    url="https://example.com/sse",
    headers={"Authorization": "Bearer your-token-here"}
)

client = Client(transport)
```

---

TITLE: Install Pre-Commit Hooks for FastMCP
DESCRIPTION: Install pre-commit hooks using `uv run pre-commit install` to automate code quality checks. These hooks enforce formatting, linting, and type-safety, and must pass for all pull requests.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_7

LANGUAGE: bash
CODE:

```
uv run pre-commit install
```

---

TITLE: Configure FastMCP Server Dependencies with uv
DESCRIPTION: This JSON configuration snippet demonstrates how to specify server dependencies using `uv` within the `mcpServers` block. It shows how to define the command and arguments to run a Python server with specific packages like `pandas` and `requests`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_5

LANGUAGE: json
CODE:

```
{
  "mcpServers": {
    "dice-roller": {
      "command": "uv",
      "args": [
        "run",
        "--with", "pandas",
        "--with", "requests",
        "python",
        "path/to/your/server.py"
      ]
    }
  }
}
```

---

TITLE: Hiding Tool Arguments with Dynamic Default Values using ArgTransform
DESCRIPTION: Demonstrates using `default_factory` with `hide=True` in `ArgTransform` to provide dynamically generated default values for hidden arguments, such as timestamps or unique IDs. Note that `default_factory` can only be used when `hide=True`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_8

LANGUAGE: python
CODE:

```
transform_args = {
    'timestamp': ArgTransform(
        hide=True,
        default_factory=lambda: datetime.now()
    )
}
```

---

TITLE: Python Function: Parse OpenAPI to HTTP Routes
DESCRIPTION: Parses an OpenAPI schema dictionary into a list of HTTPRoute objects using the openapi-pydantic library. Supports both OpenAPI 3.0.x and 3.1.x versions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-utilities-openapi.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
parse_openapi_to_http_routes(openapi_dict: dict[str, Any]) -> list[HTTPRoute]
```

---

TITLE: Configure FastMCP Server Duplicate Tool Behavior
DESCRIPTION: Demonstrates how to configure the `on_duplicate_tools` argument when initializing a `FastMCP` instance to control the server's behavior when attempting to register multiple tools with the same name. It shows an example of setting the behavior to "error".
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_33

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(
    name="StrictServer",
    # Configure behavior for duplicate tool names
    on_duplicate_tools="error"
)

@mcp.tool
def my_tool(): return "Version 1"

# This will now raise a ValueError because 'my_tool' already exists
# and on_duplicate_tools is set to "error".
# @mcp.tool
# def my_tool(): return "Version 2"
```

---

TITLE: Unified Post Tool API Definition
DESCRIPTION: This is the API signature for the `post` tool, a flexible interface for creating various types of posts on the AT Protocol. It supports text, images, links, mentions, replies, and quotes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/examples/atproto_mcp/README.md#_snippet_2

LANGUAGE: APIDOC
CODE:

```
async def post(
    text: str,                          # Required: Post content
    images: list[str] = None,           # Optional: Image URLs (max 4)
    image_alts: list[str] = None,       # Optional: Alt text for images
    links: list[RichTextLink] = None,   # Optional: Embedded links
    mentions: list[RichTextMention] = None,  # Optional: User mentions
    reply_to: str = None,               # Optional: Reply to post URI
    reply_root: str = None,             # Optional: Thread root URI
    quote: str = None,                  # Optional: Quote post URI
)
```

---

TITLE: Initialize FastMCP with Tag-Based Filtering
DESCRIPTION: Demonstrates how to initialize a FastMCP instance to filter components based on tags. Components with 'admin' tag will be included, while those with 'deprecated' tag will be excluded. This filtering applies to all component types (tools, resources, resource templates, and prompts) and affects both listing and access.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/server.mdx#_snippet_8

LANGUAGE: python
CODE:

```
mcp = FastMCP(include_tags={"admin"}, exclude_tags={"deprecated"})
```

---

TITLE: Proxying In-Memory FastMCP Instances
DESCRIPTION: This snippet shows how to create a proxy for an existing in-memory FastMCP server. This is useful for adjusting the configuration or behavior of a server that you don't completely control, by placing a proxy in front of it.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Original server
original_server = FastMCP(name="Original")

@original_server.tool
def tool_a() -> str:
    return "A"

# Create a proxy of the original server directly
proxy = FastMCP.as_proxy(
    original_server,
    name="Proxy Server"
)

# proxy is now a regular FastMCP server that forwards
# requests to original_server
```

---

TITLE: FastMCP Client Configuration Dictionary Format
DESCRIPTION: This snippet outlines the structure of an MCP configuration dictionary used to define multiple servers for the FastMCP client. It details the `mcpServers` key, which contains definitions for both remote HTTP/SSE servers (with URL, headers, and auth) and local Stdio servers (with command, arguments, environment variables, and current working directory).
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_2

LANGUAGE: python
CODE:

```
config = {
    "mcpServers": {
        "server_name": {
            # Remote HTTP/SSE server
            "transport": "http",  # or "sse"
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
            "auth": "oauth"  # or bearer token string
        },
        "local_server": {
            # Local stdio server
            "transport": "stdio",
            "command": "python",
            "args": ["./server.py", "--verbose"],
            "env": {"DEBUG": "true"},
            "cwd": "/path/to/server"
        }
    }
}
```

---

TITLE: Handling UUID Parameters in FastMCP Tools
DESCRIPTION: Demonstrates the use of `uuid.UUID` for unique identifiers in FastMCP tool parameters. FastMCP automatically converts string UUIDs (e.g., "123e4567-e89b-12d3-a456-426614174000") provided by clients into `UUID` objects.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_28

LANGUAGE: python
CODE:

```
import uuid

@mcp.tool
def process_item(
    item_id: uuid.UUID  # String UUID or UUID object
) -> str:
    """Process an item with the given UUID."""
    assert isinstance(item_id, uuid.UUID)  # Properly converted to UUID
    return f"Processing item {item_id}"
```

---

TITLE: Managing Git Forks After Initial Development
DESCRIPTION: This snippet provides a 'get out of jail free card' for developers who start working before creating a fork. It details the Git commands to add a new remote for your fork, verify it, commit local changes, and push them to your personal fork, preparing for a pull request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/Windows_Notes.md#_snippet_4

LANGUAGE: Git
CODE:

```
git remote add fork git@github.com:YOUR-USERNAME/REPOSITORY-NAME.git
```

LANGUAGE: Git
CODE:

```
git remote -v
```

LANGUAGE: Git
CODE:

```
git push fork <branch>
```

---

TITLE: Create Starlette HTTP Application with Transport Options
DESCRIPTION: Creates a generic Starlette application, providing flexibility to choose between 'streamable-http' (default) and 'sse' transport protocols. It supports custom endpoint paths and middleware integration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_14

LANGUAGE: APIDOC
CODE:

```
http_app(self, path: str | None = None, middleware: list[ASGIMiddleware] | None = None, json_response: bool | None = None, stateless_http: bool | None = None, transport: Literal['streamable-http', 'sse'] = 'streamable-http') -> StarletteWithLifespan
	path: The path for the HTTP endpoint
	middleware: A list of middleware to apply to the app
	transport: Transport protocol to use - either "streamable-http" (default) or "sse"

Returns:
	A Starlette application configured with the specified transport
```

---

TITLE: Use prompts with multi-server clients (Python)
DESCRIPTION: Illustrates that prompts are directly accessible without prefixes when using a multi-server FastMCP client, unlike tools. It shows fetching prompts from different conceptual servers seamlessly.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/prompts.mdx#_snippet_7

LANGUAGE: python
CODE:

```
async with client:  # Multi-server client
    # Prompts from any server are directly accessible
    result1 = await client.get_prompt("weather_prompt", {"city": "London"})
    result2 = await client.get_prompt("assistant_prompt", {"query": "help"})
```

---

TITLE: Customizing FastMCP App Mount Paths
DESCRIPTION: Illustrates how to change the default mount paths for FastMCP ASGI applications by passing a `path` argument to the `http_app()` method for both Streamable HTTP and SSE transports.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_1

LANGUAGE: python
CODE:

```
# For Streamable HTTP transport
http_app = mcp.http_app(path="/custom-mcp-path")

# For SSE transport (deprecated)
sse_app = mcp.http_app(path="/custom-sse-path", transport="sse")
```

---

TITLE: Add Annotations to FastMCP Tools
DESCRIPTION: Illustrates how to apply annotations like 'title', 'readOnlyHint', and 'openWorldHint' to a FastMCP tool using the '@mcp.tool' decorator, providing metadata for client applications.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_15

LANGUAGE: python
CODE:

```
@mcp.tool(
    annotations={
        "title": "Calculate Sum",
        "readOnlyHint": True,
        "openWorldHint": False
    }
)
def calculate_sum(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b
```

---

TITLE: Creating Nested Mounts with FastMCP in Starlette
DESCRIPTION: Shows a partial example of how to set up complex routing structures by nesting mounts within a Starlette application, using a FastMCP ASGI app.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

# Create your FastMCP server as well as any tools, resources, etc.
mcp = FastMCP("MyServer")

# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')
```

---

TITLE: FastMCP: Custom Route Maps for Specific Endpoint Handling
DESCRIPTION: This example illustrates how to apply custom `RouteMap` configurations to `FastMCP.from_openapi`. It defines rules to convert GET requests under `/analytics/` to `Tool`s, and to exclude all routes under `/admin/` or those tagged as 'internal'. This allows for fine-grained control over how different API endpoints are categorized.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

mcp = FastMCP.from_openapi(
    ...,
    route_maps=[

        # Analytics `GET` endpoints are tools
        RouteMap(
            methods=["GET"],
            pattern=r"^/analytics/.*",
            mcp_type=MCPType.TOOL,
        ),

        # Exclude all admin endpoints
        RouteMap(
            pattern=r"^/admin/.*",
            mcp_type=MCPType.EXCLUDE,
        ),

        # Exclude all routes tagged "internal"
        RouteMap(
            tags={"internal"},
            mcp_type=MCPType.EXCLUDE,
        ),
    ],
)
```

---

TITLE: Apply Custom Tags to FastMCP Components via RouteMap
DESCRIPTION: This snippet demonstrates how to use `RouteMap` with the `mcp_tags` parameter to apply custom tags to FastMCP components. Tags are applied based on matching HTTP methods and URL patterns, allowing for granular categorization of components like write operations, detail views, or list data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

mcp = FastMCP.from_openapi(
    ...,
    route_maps=[
        # Add custom tags to all POST endpoints
        RouteMap(
            methods=["POST"],
            pattern=r".*",
            mcp_type=MCPType.TOOL,
            mcp_tags={"write-operation", "api-mutation"}
        ),

        # Add different tags to detail view endpoints
        RouteMap(
            methods=["GET"],
            pattern=r".*\\{.*\\}.*",
            mcp_type=MCPType.RESOURCE_TEMPLATE,
            mcp_tags={"detail-view", "parameterized"}
        ),

        # Add tags to list endpoints
        RouteMap(
            methods=["GET"],
            pattern=r".*",
            mcp_type=MCPType.RESOURCE,
            mcp_tags={"list-data", "collection"}
        ),
    ],
)
```

---

TITLE: FastMCP on_duplicate_tools Options
DESCRIPTION: Describes the available options for the `on_duplicate_tools` argument in the `FastMCP` constructor, detailing how each option ("warn", "error", "replace", "ignore") affects the server's response to duplicate tool registrations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_34

LANGUAGE: APIDOC
CODE:

```
FastMCP(on_duplicate_tools) Options:
- "warn" (default): Logs a warning and the new tool replaces the old one.
- "error": Raises a ValueError, preventing the duplicate registration.
- "replace": Silently replaces the existing tool with the new one.
- "ignore": Keeps the original tool and ignores the new registration attempt.
```

---

TITLE: Manually Run All Pre-Commit Hooks
DESCRIPTION: Use this command to manually execute all configured pre-commit hooks across all files in the repository. This is useful for checking code quality before committing changes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_8

LANGUAGE: bash
CODE:

```
pre-commit run --all-files
```

---

TITLE: Overriding Progress Handler for Specific FastMCP Calls
DESCRIPTION: Shows how to provide a different progress handler for an individual tool call using the progress_handler argument within client.call_tool, allowing fine-grained control over progress updates.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/progress.mdx#_snippet_2

LANGUAGE: python
CODE:

```
async with client:
    # Override with specific progress handler for this call
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )
```

---

TITLE: FastMCP Client Session Management with keep_alive=False
DESCRIPTION: Illustrates how setting `keep_alive=False` ensures that a new subprocess is started for each client context, providing complete isolation between sessions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import Client

# Client with keep_alive=False
client = Client("my_mcp_server.py", keep_alive=False)

async def example():
    # First session
    async with client:
        await client.ping()

    # Second session - will start a new subprocess
    async with client:
        await client.ping()

    # Third session - will start a new subprocess
    async with client:
        await client.ping()

asyncio.run(example())
```

---

TITLE: APIDOC: Python Function create_oauth_callback_server
DESCRIPTION: Documents the `create_oauth_callback_server` function, which creates an OAuth callback server. It requires a port, an optional callback path, an optional server URL for display, and an optional future to resolve upon callback reception. It returns a configured uvicorn Server instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-client-oauth_callback.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
create_oauth_callback_server:
  Parameters:
    port: int - The port to run the server on
    callback_path: str = '/callback' - The path to listen for OAuth redirects on
    server_url: str | None = None - Optional server URL to display in success messages
    response_future: asyncio.Future | None = None - Optional future to resolve when OAuth callback is received
  Returns: Server - Configured uvicorn Server instance (not yet running)
```

---

TITLE: Add Custom Web Routes to FastMCP Server
DESCRIPTION: This example shows how to add custom web routes to your FastMCP server using the `@mcp.custom_route` decorator. This feature allows for simple endpoints, such as health checks, to be exposed alongside the main MCP endpoint, though it's less flexible than a full ASGI framework.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_11

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("MyServer")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

if __name__ == "__main__":
    mcp.run()
```

---

TITLE: FastMCP Function with Multiple URI Templates
DESCRIPTION: Shows how to register a single FastMCP function with multiple URI templates, enabling different access patterns to the same underlying logic. The `lookup_user` function can be accessed by either email or name, with parameters defaulting to `None` if not provided in the URI.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_14

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

# Define a user lookup function that can be accessed by different identifiers
@mcp.resource("users://email/{email}")
@mcp.resource("users://name/{name}")
def lookup_user(name: str | None = None, email: str | None = None) -> dict:
    """Look up a user by either name or email."""
    if email:
        return find_user_by_email(email) # pseudocode
    elif name:
        return find_user_by_name(name) # pseudocode
    else:
        return {"error": "No lookup parameters provided"}
```

---

TITLE: Set HTTP Request Context (Python)
DESCRIPTION: Sets the current HTTP request in a context variable, typically for access within a request scope, allowing other parts of the application to retrieve the active request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-http.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
set_http_request(request: Request) -> Generator[Request, None, None]
  request: The incoming HTTP request object.
  Returns: A generator that yields the request, typically used as a dependency injection.
```

---

TITLE: BearerAuthProvider Configuration Parameters
DESCRIPTION: This section details the configuration parameters available for the `BearerAuthProvider` class, including required fields like `public_key` or `jwks_uri`, and optional validation criteria such as `issuer`, `audience`, and `required_scopes`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
BearerAuthProvider Configuration:
  public_key: str
    RSA public key in PEM format for static key validation. Required if `jwks_uri` is not provided
  jwks_uri: str
    URL for JSON Web Key Set endpoint. Required if `public_key` is not provided
  issuer: str | None
    Expected JWT `iss` claim value
  audience: str | None
    Expected JWT `aud` claim value
  required_scopes: list[str] | None
    Global scopes required for all requests
```

---

TITLE: Update Claude Configuration with FastMCP Server
DESCRIPTION: Adds or modifies a FastMCP server entry within Claude's configuration, handling various installation and environment options. Raises RuntimeError if Claude Desktop's config directory is not found.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-cli-claude.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
update_claude_config(file_spec: str, server_name: str) -> bool
  Description: Add or update a FastMCP server in Claude's configuration.
  Parameters:
    file_spec (str): Path to the server file, optionally with :object suffix
    server_name (str): Name for the server in Claude's config
    with_editable (Optional[str]): Optional directory to install in editable mode
    with_packages (Optional[list]): Optional list of additional packages to install
    env_vars (Optional[dict]): Optional dictionary of environment variables. These are merged with any existing variables, with new values taking precedence.
  Raises:
    RuntimeError: If Claude Desktop's config directory is not found, indicating Claude Desktop may not be installed or properly set up.
```

---

TITLE: Modify FastMCP Components In-Place with mcp_component_fn
DESCRIPTION: This snippet demonstrates using the `mcp_component_fn` parameter for advanced, fine-grained customization of FastMCP components. The provided function is called on each component after its creation, allowing in-place modification of properties like descriptions or tags based on the component's type or associated route.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_10

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import (
    HTTPRoute,
    OpenAPITool,
    OpenAPIResource,
    OpenAPIResourceTemplate,
)

def customize_components(
    route: HTTPRoute,
    component: OpenAPITool | OpenAPIResource | OpenAPIResourceTemplate,
) -> None:

    # Add custom tags to all components
    component.tags.add("openapi")

    # Customize based on component type
    if isinstance(component, OpenAPITool):
        component.description = f"🔧 {component.description} (via API)"

    if isinstance(component, OpenAPIResource):
        component.description = f"📊 {component.description}"
        component.tags.add("data")

mcp = FastMCP.from_openapi(
    ...,
    mcp_component_fn=customize_components,
)
```

---

TITLE: Defining and Using Enum Parameters in FastMCP
DESCRIPTION: Demonstrates how to define and use Python's Enum class for tool parameters in FastMCP. FastMCP automatically coerces string values provided by clients into the appropriate Enum objects, ensuring type safety and enabling built-in validation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_24

LANGUAGE: python
CODE:

```
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

@mcp.tool
def process_image(
    image_path: str,
    color_filter: Color = Color.RED
):
    """Process an image with a color filter."""
    # Implementation...
    # color_filter will be a Color enum member
```

---

TITLE: Fixing Python AttributeError: collections.Callable
DESCRIPTION: This snippet offers a solution for the `AttributeError: module 'collections' has no attribute 'Callable'` often encountered in newer Python versions. It instructs to modify `pyreadline`'s `py3k_compat.py` file to import `Callable` from `collections.abc`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/Windows_Notes.md#_snippet_1

LANGUAGE: Python
CODE:

```
from collections.abc import Callable
return isinstance(x, Callable)
```

---

TITLE: Extract HTTP Headers from Current Request in fastmcp.server.dependencies
DESCRIPTION: This function extracts headers from the current HTTP request. It returns an empty dictionary if no active HTTP request is found, ensuring no exceptions are raised. By default, it strips problematic headers like `content-length` to prevent issues when forwarding. Setting `include_all` to True will return all headers without filtering.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-dependencies.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
get_http_headers(include_all: bool = False) -> dict[str, str]

Extract headers from the current HTTP request if available.

Never raises an exception, even if there is no active HTTP request (in which case
an empty dict is returned).

By default, strips problematic headers like `content-length` that cause issues if forwarded to downstream clients.
If `include_all` is True, all headers are returned.
```

---

TITLE: FastMCP Python Stdio Transport Usage
DESCRIPTION: Explains the `PythonStdioTransport` class, its inference from `.py` files, and its use case for running Python-based MCP server scripts as subprocesses. It provides a Python example demonstrating both inferred and explicit transport configuration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_10

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.PythonStdioTransport
Inferred From: Paths to .py files
Use Case: Running a Python-based MCP server script in a subprocess
```

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

server_script = "my_mcp_server.py" # Path to your server script

# Option 1: Inferred transport
client = Client(server_script)

# Option 2: Explicit transport with custom configuration
transport = PythonStdioTransport(
    script_path=server_script,
    python_cmd="/usr/bin/python3.11", # Optional: specify Python interpreter
    # args=["--some-server-arg"],      # Optional: pass arguments to the script
    # env={"MY_VAR": "value"},         # Optional: set environment variables
)
client = Client(transport)

async def main():
    async with client:
        tools = await client.list_tools()
        print(f"Connected via Python Stdio, found tools: {tools}")

asyncio.run(main())
```

---

TITLE: FastMCP OAuth Class Parameters
DESCRIPTION: Detailed documentation for the parameters available when initializing the `fastmcp.client.auth.OAuth` helper class. These parameters allow for fine-grained control over the OAuth 2.1 Authorization Code Grant flow, including scope requests, client naming, token storage, and additional metadata for dynamic client registration.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
OAuth Class Parameters:
- mcp_url (str): The full URL of the target MCP server endpoint. Used to discover OAuth server metadata.
- scopes (str | list[str], optional): OAuth scopes to request. Can be space-separated string or list of strings.
- client_name (str, optional): Client name for dynamic registration. Defaults to "FastMCP Client".
- token_storage_cache_dir (Path, optional): Token cache directory. Defaults to ~/.fastmcp/oauth-mcp-client-cache/.
- additional_client_metadata (dict[str, Any], optional): Extra metadata for client registration.
```

---

TITLE: Add Resource or Template from Function to FastMCP Server
DESCRIPTION: Registers a Python function as a resource or resource template on the FastMCP server. If the URI contains parameters (e.g., "resource://{param}") or the function itself has parameters, it will be registered as a template resource.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-server.mdx#_snippet_7

LANGUAGE: APIDOC
CODE:

```
add_resource_fn(self, fn: AnyFunction, uri: str, name: str | None = None, description: str | None = None, mime_type: str | None = None, tags: set[str] | None = None) -> None

Args:
- fn: The function to register as a resource.
- uri: The URI for the resource.
- name: Optional name for the resource.
- description: Optional description of the resource.
- mime_type: Optional MIME type for the resource.
- tags: Optional set of tags for categorizing the resource.
```

---

TITLE: FastMCP Resource with Wildcard Path Parameter
DESCRIPTION: Demonstrates defining a FastMCP resource using a wildcard parameter (`{path*}`) to capture variable-length path segments. The function `get_template_file` retrieves content from a repository, ensuring the resource path ends with `template.py`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_12

LANGUAGE: python
CODE:

```
@mcp.resource("repo://{owner}/{path*}/template.py")
def get_template_file(owner: str, path: str) -> dict:
    """Retrieves a file from a specific repository and path, but
    only if the resource ends with `template.py`"""
    # Can match repo://jlowin/fastmcp/src/resources/template.py
    return {
        "owner": owner,
        "path": path + "/template.py",
        "content": f"File at {path}/template.py in {owner}'s repository"
    }
```

---

TITLE: Running FastMCP Server with Explicit STDIO Transport
DESCRIPTION: This Python snippet explicitly sets the `transport` argument to `"stdio"` when calling the `mcp.run()` method. While STDIO is the default transport, explicit specification clarifies intent for local command-line integrations and client-managed server processes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

TITLE: @mcp.tool Decorator Arguments API Reference
DESCRIPTION: Comprehensive API documentation for the arguments available when using the @mcp.tool decorator. This includes detailed descriptions of each parameter, their types, default values, and the nested ToolAnnotations attributes for adding advanced metadata about the tool's behavior and interaction with its environment.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
@mcp.tool Decorator Arguments:
  name: str | None
    Sets the explicit tool name exposed via MCP. If not provided, uses the function name.
  description: str | None
    Provides the description exposed via MCP. If set, the function's docstring is ignored for this purpose.
  tags: set[str] | None
    A set of strings to categorize the tool. Clients might use tags to filter or group available tools.
  enabled: bool = True
    A boolean to enable or disable the tool.
  exclude_args: list[str] | None
    A list of argument names to exclude from the tool schema shown to the LLM.
  annotations: ToolAnnotations | dict | None
    An optional ToolAnnotations object or dictionary to add additional metadata about the tool.
    ToolAnnotations attributes:
      title: str | None
        A human-readable title for the tool.
      readOnlyHint: bool | None
        If true, the tool does not modify its environment.
      destructiveHint: bool | None
        If true, the tool may perform destructive updates to its environment.
      idempotentHint: bool | None
        If true, calling the tool repeatedly with the same arguments will have no additional effect on the its environment.
      openWorldHint: bool | None
        If true, this tool may interact with an "open world" of external entities. If false, the tool's domain of interaction is closed.
```

---

TITLE: Create Base Starlette Application (Python)
DESCRIPTION: Initializes a foundational Starlette application with a specified set of routes and middleware. It provides options for enabling debug mode and integrating a custom lifespan manager for application startup and shutdown events.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/python-sdk/fastmcp-server-http.mdx#_snippet_2

LANGUAGE: APIDOC
CODE:

```
create_base_app(routes: list[BaseRoute], middleware: list[Middleware], debug: bool = False, lifespan: Callable | None = None) -> StarletteWithLifespan
  routes: List of routes to include in the app.
  middleware: List of middleware to include in the app.
  debug: Whether to enable debug mode.
  lifespan: Optional lifespan manager for the app.
  Returns: A Starlette application instance.
```

---

TITLE: Controlling FastMCP Prompt Visibility and Availability in Python
DESCRIPTION: Explains how to control the visibility and availability of prompts in FastMCP. Prompts can be disabled during creation using the 'enabled' parameter or toggled programmatically after creation. Disabled prompts will not appear and will cause an 'Unknown prompt' error if called.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_8

LANGUAGE: python
CODE:

```
@mcp.prompt(enabled=False)
def experimental_prompt():
    """This prompt is not ready for use."""
    return "This is an experimental prompt."
```

LANGUAGE: python
CODE:

```
@mcp.prompt
def seasonal_prompt(): return "Happy Holidays!"

# Disable and re-enable the prompt
seasonal_prompt.disable()
seasonal_prompt.enable()
```
