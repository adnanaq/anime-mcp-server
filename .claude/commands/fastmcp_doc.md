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

TITLE: Defining a FastMCP Tool with @mcp.tool Decorator
DESCRIPTION: This snippet demonstrates how to define a simple tool in FastMCP by decorating a Python function with `@mcp.tool`. It shows a basic `add` function that takes two integers and returns their sum, illustrating automatic tool naming, description from docstrings, and schema generation from type annotations.
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

TITLE: Define a simple FastMCP server with a tool in Python
DESCRIPTION: This Python code demonstrates how to initialize a FastMCP server, define a simple 'add' tool using the '@mcp.tool' decorator, and run the server. The tool takes two integers and returns their sum, making it available for LLM applications via the Model Context Protocol.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_0

LANGUAGE: python
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

TITLE: Instantiate FastMCP Server with Name and Instructions
DESCRIPTION: Demonstrates how to create a basic FastMCP server instance, optionally providing a human-readable name and initial instructions that help clients understand the server's purpose and available functionality.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_0

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

TITLE: Define and Run a FastMCP Tool (STDIO Default)
DESCRIPTION: Demonstrates how to initialize FastMCP, define a simple tool using the `@mcp.tool` decorator, and run the server using the default STDIO transport.
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

TITLE: Define Tool Parameters with Python Built-in Scalar Types in FastMCP
DESCRIPTION: This snippet demonstrates how to define tool parameters using Python's fundamental scalar types: `str`, `int`, `float`, and `bool`. FastMCP leverages these types to provide clear expectations to LLMs and perform automatic input validation and type coercion, converting strings like '42' to integers if annotated as `int`.
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

TITLE: Configure BearerAuthProvider with a JWKS URI
DESCRIPTION: This snippet shows how to configure `BearerAuthProvider` using a JSON Web Key Set (JWKS) URI. This approach is recommended for production environments as it supports automatic key rotation and multiple signing keys, enhancing security and flexibility.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_2

LANGUAGE: python
CODE:

```
provider = BearerAuthProvider(
    jwks_uri="https://idp.example.com/.well-known/jwks.json"
)
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

TITLE: Analyze Sentiment with FastMCP LLM Sampling
DESCRIPTION: This tool demonstrates how to use `ctx.sample` to perform sentiment analysis on a given text. It constructs a prompt for the client's LLM, sends the request, and processes the response to return a standardized sentiment (positive, negative, or neutral).
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

TITLE: Define Asynchronous FastMCP Resources for I/O (Python)
DESCRIPTION: Resource functions that perform I/O operations, such as reading from a database or network, should be defined as `async def` functions. This prevents blocking the server and ensures efficient handling of concurrent requests, as demonstrated by asynchronously reading a log file.
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

TITLE: FastMCP Client Initialization and Asynchronous Operations
DESCRIPTION: This snippet demonstrates how to initialize the FastMCP client with a configuration, and then perform asynchronous operations such as calling tools and reading resources. It highlights the use of server name prefixes for tools and URI paths for resources in a multi-server setup, and how the client automatically handles direct connections for single-server configurations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_17

LANGUAGE: Python
CODE:

```
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

TITLE: Mounting FastMCP Server in a FastAPI Application
DESCRIPTION: This example illustrates how to integrate a FastMCP server into a FastAPI application. It shows the creation of a FastMCP server, its ASGI app, and then mounting it into a FastAPI instance, emphasizing the importance of passing the lifespan context for proper initialization.
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

TITLE: Managing FastMCP Client Connection Lifecycle with async with
DESCRIPTION: Illustrates the asynchronous connection lifecycle of a FastMCP client using an `async with` block. It shows how the client establishes a connection upon entering the block, allows MCP calls within the context, and automatically closes the connection upon exiting, demonstrating `is_connected()` and `list_tools()`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_3

LANGUAGE: Python
CODE:

```
import asyncio
from fastmcp import Client

client = Client("my_mcp_server.py") # Assumes my_mcp_server.py exists

async def main():
    # Connection is established here
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Make MCP calls within the context
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        if any(tool.name == "greet" for tool in tools):
            result = await client.call_tool("greet", {"name": "World"})
            print(f"Greet result: {result}")

    # Connection is closed automatically here
    print(f"Client connected: {client.is_connected()}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: Run FastMCP Server from Python using run() method
DESCRIPTION: Demonstrates how to run a FastMCP server directly from a Python script by calling the `run()` method on a `FastMCP` instance. It's recommended to place the call within an `if __name__ == "__main__":` block for proper execution when imported as a module.
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

TITLE: Handle Flexible and Optional Parameters with Union Types in FastMCP
DESCRIPTION: This example demonstrates using Python's union types (`|`) to define parameters that can accept multiple types or be optional. FastMCP supports `str | int` for parameters that can be either a string or an integer, and `Type | None` for optional parameters, preferring modern Python syntax over older `Union` or `Optional` forms.
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

TITLE: Access MCP Context for Logging, Resources, and Progress
DESCRIPTION: Illustrates how to use the Context object within an asynchronous FastMCP tool to perform actions such as logging information, reading external resources, reporting progress, and sampling the client's LLM.
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

TITLE: Configure FastMCP Server with Bearer Token Authentication
DESCRIPTION: This snippet demonstrates the basic setup for enabling Bearer Token authentication on a FastMCP server. It shows how to instantiate `BearerAuthProvider` with a JWKS URI, issuer, and audience, and then pass this provider to the `FastMCP` instance to secure its HTTP endpoints.
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

TITLE: Adding Parameter Metadata with Pydantic Annotated Field
DESCRIPTION: This snippet demonstrates the preferred method for adding rich metadata to tool parameters using Pydantic's `Annotated` and `Field`. It allows for detailed descriptions, validation constraints (like `ge`, `le`), and enumeration types, which are used by FastMCP to generate comprehensive schemas for LLMs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_2

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

TITLE: Utilize Type Annotations and Pydantic Fields for FastMCP Prompts
DESCRIPTION: Highlights the importance of type annotations for parameter validation and schema generation in FastMCP prompts. It demonstrates using `pydantic.Field` for descriptions, `typing.Literal` for constrained choices, and `typing.Optional` for optional parameters. This ensures robust input handling and clear API definitions for prompts.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from pydantic import Field
from typing import Literal, Optional

@mcp.prompt
def generate_content_request(
    topic: str = Field(description="The main subject to cover"),
    format: Literal["blog", "email", "social"] = "blog",
    tone: str = "professional",
    word_count: Optional[int] = None
) -> str:
    """Create a request for generating content in a specific format."""
    prompt = f"Please write a {format} post about {topic} in a {tone} tone."

    if word_count:
        prompt += f" It should be approximately {word_count} words long."

    return prompt
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

TITLE: In-memory Testing of FastMCP Servers
DESCRIPTION: Shows how to connect `fastmcp.Client` directly to a `FastMCP` server instance for efficient in-memory testing, eliminating the need for process management or network calls during tests.
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

TITLE: Define a FastMCP Parameterized Resource Template
DESCRIPTION: Demonstrates creating a parameterized resource template with `@mcp.resource`, where parts of the URI (e.g., `{user_id}`) are extracted and passed as arguments to the function. This enables clients to request specific, dynamic data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_3

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

TITLE: Define a FastMCP Tool for LLM Actions
DESCRIPTION: Decorate a Python function with `@mcp.tool` to expose it as an LLM-executable action. FastMCP automatically generates schema from type hints and docstrings, supporting various return types including text, JSON, images, or audio.
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

TITLE: Using the @mcp.tool decorator in Python
DESCRIPTION: This snippet demonstrates the use of the 'naked' @mcp.tool decorator in Python, allowing for a more Pythonic way to define tools. It shows a basic function decorated as a tool within the FastMCP framework, simplifying tool creation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/changelog.mdx#_snippet_0

LANGUAGE: python
CODE:

```
@mcp.tool
def my_tool():
    ...
```

---

TITLE: Define a FastMCP Tool for Functionality
DESCRIPTION: Shows how to define a callable tool using the `@mcp.tool` decorator. Tools are functions that the client can call to perform actions or access external systems, making Python functions accessible to MCP clients.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_1

LANGUAGE: python
CODE:

```
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b
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

TITLE: Connect to MCP Servers using fastmcp.Client
DESCRIPTION: Demonstrates connecting to an MCP server via stdio (local script) and SSE (HTTP endpoint) using the `fastmcp.Client`. It shows how to list available tools and call a specific tool.
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

TITLE: Create a FastMCP Dice Roller Server
DESCRIPTION: This Python code defines a FastMCP server named 'Dice Roller' with a single tool, `roll_dice`, which simulates rolling 6-sided dice. It returns a list of integers representing the results. The server runs using SSE transport on port 8000.
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
    mcp.run(transport="sse", port=8000)
```

---

TITLE: Using Standard Type Annotations for Tool Parameters
DESCRIPTION: This example illustrates the use of standard Python type annotations for tool parameters. These annotations are crucial for informing the LLM about expected data types, enabling FastMCP to validate inputs, and generating accurate JSON schemas for the MCP protocol. It shows parameters with default values and optional types.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_1

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

TITLE: FastMCP Standard Tool Annotations Reference
DESCRIPTION: Documents the standard annotations supported by FastMCP, including their type, default value, and purpose, which help client applications present appropriate UI and safety controls.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_16

LANGUAGE: APIDOC
CODE:

```
Annotation: title
  Type: string
  Default: -
  Purpose: Display name for user interfaces
Annotation: readOnlyHint
  Type: boolean
  Default: false
  Purpose: Indicates if the tool only reads without making changes
Annotation: destructiveHint
  Type: boolean
  Default: true
  Purpose: For non-readonly tools, signals if changes are destructive
Annotation: idempotentHint
  Type: boolean
  Default: false
  Purpose: Indicates if repeated identical calls have the same effect as a single call
Annotation: openWorldHint
  Type: boolean
  Default: true
  Purpose: Specifies if the tool interacts with external systems
```

---

TITLE: Call FastMCP Server via OpenAI Responses API
DESCRIPTION: This Python code demonstrates how to use the OpenAI Python SDK to call a FastMCP server through the Responses API. It configures a tool of type 'mcp' with the server's URL and sends an input prompt, then prints the AI's response.
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
            "server_url": f"{url}/sse",
            "require_approval": "never",
        }
    ],
    input="Roll a few dice!"
)

print(resp.output_text)
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

TITLE: Install FastMCP Library
DESCRIPTION: Installs the FastMCP Python library using pip, which is a prerequisite for building and interacting with MCP servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_0

LANGUAGE: bash
CODE:

```
pip install fastmcp
```

---

TITLE: Run FastMCP Server with Default Streamable HTTP Transport
DESCRIPTION: This snippet demonstrates how to start a FastMCP server using the recommended Streamable HTTP transport with default settings. It also shows a client connecting to this server and performing a simple ping operation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        await client.ping()

if __name__ == "__main__":
    asyncio.run(example())
```

---

TITLE: Create a FastMCP Server with Dice Rolling Tool
DESCRIPTION: This Python script demonstrates how to set up a FastMCP server with a single tool, `roll_dice`, which simulates rolling a specified number of 6-sided dice. The server is initialized with the name 'Dice Roller' and runs when executed directly.
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

TITLE: Initializing FastMCP Client with Inferred Transports - Python
DESCRIPTION: This snippet demonstrates how the FastMCP Client automatically infers the appropriate transport type based on the input provided during initialization. It shows examples for connecting to an in-memory FastMCP server instance, an HTTP server URL, and a Python script acting as a server via standard I/O. The output displays the inferred transport objects for each client.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_0

LANGUAGE: Python
CODE:

```
import asyncio
from fastmcp import Client, FastMCP

# Example transports (more details in Transports page)
server_instance = FastMCP(name="TestServer") # In-memory server
http_url = "https://example.com/mcp"        # HTTP server URL
server_script = "my_mcp_server.py"         # Path to a Python server file

# Client automatically infers the transport type
client_in_memory = Client(server_instance)
client_http = Client(http_url)

client_stdio = Client(server_script)

print(client_in_memory.transport)
print(client_http.transport)
print(client_stdio.transport)

# Expected Output (types may vary slightly based on environment):
# <FastMCP(server='TestServer')>
# <StreamableHttp(url='https://example.com/mcp')>
# <PythonStdioTransport(command='python', args=['/path/to/your/my_mcp_server.py'])>
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

TITLE: Define Static and Dynamic FastMCP Resources
DESCRIPTION: Expose read-only data sources using `@mcp.resource`. Static resources provide fixed data, while dynamic resources use URI placeholders for parameterized data retrieval, allowing clients to request specific data subsets.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_5

LANGUAGE: python
CODE:

```
# Static resource
@mcp.resource("config://version")
def get_version():
    return "2.0.1"

# Dynamic resource template
@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    # Fetch profile for user_id...
    return {"name": f"User {user_id}", "status": "active"}
```

---

TITLE: Deploy FastMCP Server Locally and Expose with ngrok
DESCRIPTION: These commands demonstrate how to run the FastMCP server locally and expose it to the internet using `ngrok`. The first command starts the Python server, and the second command creates a public tunnel for port 8000, making the server accessible externally.
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

TITLE: Calling a FastMCP Tool with Arguments
DESCRIPTION: Shows a basic example of using `call_tool()` to execute a server-side tool named 'add' with specified arguments. The result is a list of content objects, typically `TextContent`, from which the text can be extracted.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_5

LANGUAGE: Python
CODE:

```
result = await client.call_tool("add", {"a": 5, "b": 3})
# result -> list[mcp.types.TextContent | mcp.types.ImageContent | ...]
print(result[0].text) # Assuming TextContent, e.g., '8'
```

---

TITLE: Handling fastmcp Client Errors in Python
DESCRIPTION: This example illustrates robust error handling for `fastmcp` client interactions. It demonstrates catching `ClientError` for server-side tool exceptions, `ConnectionError` for network issues, and a general `Exception` for unexpected errors during `call_tool` operations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_15

LANGUAGE: Python
CODE:

```
async def safe_call_tool():
    async with client:
        try:
            # Assume 'divide' tool exists and might raise ZeroDivisionError
            result = await client.call_tool("divide", {"a": 10, "b": 0})
            print(f"Result: {result}")
        except ClientError as e:
            print(f"Tool call failed: {e}")
        except ConnectionError as e:
            print(f"Connection failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
```

---

TITLE: Running FastMCP ASGI App with Uvicorn (CLI)
DESCRIPTION: Shows the command-line interface (CLI) method for starting a FastMCP ASGI application using `uvicorn`. This approach is common for deploying ASGI applications and requires specifying the module path and the ASGI application instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/asgi.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
uvicorn path.to.your.app:http_app --host 0.0.0.0 --port 8000
```

---

TITLE: Injecting Context into FastMCP Tool Functions
DESCRIPTION: Demonstrates how to automatically inject the `Context` object into a FastMCP tool function by adding a type-hinted parameter. This allows the tool to access MCP capabilities like logging and resource management during execution. The parameter name is flexible, and context methods are asynchronous.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_0

LANGUAGE: Python
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

TITLE: FastMCP Pydantic Field Validation with Annotated
DESCRIPTION: Demonstrates how to use Pydantic's Field class with Python's Annotated type for robust parameter validation in FastMCP tools. This approach allows defining various constraints like numeric ranges, string patterns, and length limits directly within function signatures.
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

TITLE: Set OpenAI API Key Environment Variable
DESCRIPTION: This command sets the `OPENAI_API_KEY` environment variable, authenticating your requests to the OpenAI API. Replace `"your-api-key"` with your actual OpenAI API key.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
export OPENAI_API_KEY="your-api-key"
```

---

TITLE: Connect to Multiple MCP Servers with Unified Client
DESCRIPTION: Illustrates how to configure `fastmcp.Client` to connect to multiple MCP servers (e.g., HTTP URL, local command) using a single client instance and access their tools with server prefixes.
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

TITLE: Create FastMCP Server from OpenAPI Specification
DESCRIPTION: This Python script (`api_server.py`) initializes an `httpx.AsyncClient` and defines a simplified OpenAPI specification for the JSONPlaceholder API. It then uses `FastMCP.from_openapi` to automatically generate an MCP server, exposing the API's endpoints as callable tools for LLMs. The server is configured to run on port 8000.
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
    mcp.run(transport="streamable-http", port=8000)
```

---

TITLE: Initialize FastMCP Server Instance
DESCRIPTION: Create the central FastMCP application object, which manages tools, resources, and connections. Configure with a server name.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Create a server instance
mcp = FastMCP(name="MyAssistantServer")
```

---

TITLE: Complete FastMCP Server with Bearer Authentication and Dice Rolling Tool
DESCRIPTION: This comprehensive example demonstrates a full FastMCP server setup with bearer token authentication. It includes key pair generation, `BearerAuthProvider` configuration, and a `roll_dice` tool, showing how to integrate authentication into a functional server application. It also prints the access token for development purposes.
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
    mcp.run(transport="sse", port=8000)
```

---

TITLE: Define Reusable FastMCP LLM Prompts
DESCRIPTION: Create reusable message templates to guide LLM interactions by decorating functions with `@mcp.prompt`. These functions can return strings or `Message` objects.
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

TITLE: Handle Diverse Return Types in FastMCP Prompt Functions
DESCRIPTION: Illustrates how FastMCP processes various return types from prompt functions, including `str`, `PromptMessage`, and `list[PromptMessage | str]`. This example specifically shows returning a list of `Message` objects to set up a multi-turn roleplaying scenario, enabling complex conversational flows.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_1

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

TITLE: Run FastMCP Server using CLI
DESCRIPTION: Shows how to run a FastMCP server from the command line using the `fastmcp run` command, specifying the server's Python file. This method ignores the `if __name__ == "__main__"` block and directly calls the `run()` method of a FastMCP object.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
fastmcp run server.py
```

---

TITLE: Integrating FastMCP with Starlette for Nested Applications
DESCRIPTION: This snippet demonstrates how to mount a FastMCP server within a Starlette application, creating a nested structure. It highlights the necessity of passing the FastMCP app's lifespan to the outer Starlette app for correct session manager initialization.
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

TITLE: Define Complex Data Structures with Pydantic Models in FastMCP
DESCRIPTION: Illustrates how to use Pydantic models for complex, structured data inputs in FastMCP tools. Pydantic provides automatic validation, clear structure, and JSON schema generation, accepting JSON objects or dictionaries as input.
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

TITLE: FastMCP Tool Return Value Conversion Rules
DESCRIPTION: Details how FastMCP automatically converts various Python return types from tools into appropriate MCP content formats for the client. This includes mappings for strings, dictionaries, lists, Pydantic models, bytes, and specialized FastMCP helper classes like `Image` and `Audio`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_12

LANGUAGE: APIDOC
CODE:

```
FastMCP Return Value Conversions:
- str: Sent as TextContent.
- dict, list, Pydantic BaseModel: Serialized to a JSON string and sent as TextContent.
- bytes: Base64 encoded and sent as BlobResourceContents (often within an EmbeddedResource).
- fastmcp.utilities.types.Image: Sent as ImageContent.
- fastmcp.utilities.types.Audio: Sent as AudioContent.
- A list of any of the above: Automatically converts each item appropriately.
- None: Results in an empty response (no content is sent back to the client).
- Other types: Attempted serialization to a string if possible.
```

---

TITLE: Run FastMCP Server with Different Transport Protocols
DESCRIPTION: FastMCP supports various transport protocols for different deployment scenarios. This snippet shows how to configure the server to use STDIO, Streamable HTTP, or SSE transports.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_12

LANGUAGE: python
CODE:

```
mcp.run(transport="stdio")  # Default, so transport argument is optional
```

LANGUAGE: python
CODE:

```
mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
```

LANGUAGE: python
CODE:

```
mcp.run(transport="sse", host="127.0.0.1", port=8000)
```

---

TITLE: Define Asynchronous FastMCP Prompt for I/O Operations (Python)
DESCRIPTION: This example demonstrates how to define an asynchronous FastMCP prompt using `async def`. This is recommended for prompt functions that perform I/O-bound operations like network requests or database queries, allowing for non-blocking execution.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_9

LANGUAGE: python
CODE:

```
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

TITLE: Install FastMCP Directly with uv pip or pip
DESCRIPTION: These commands install the FastMCP package directly into your Python environment. You can choose between uv's pip functionality or the standard pip installer for direct installation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_1

LANGUAGE: bash uv
CODE:

```
uv pip install fastmcp
```

LANGUAGE: bash pip
CODE:

```
pip install fastmcp
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

TITLE: Customize FastMCP Tool Metadata with Decorator Arguments
DESCRIPTION: Illustrates how to override the inferred tool name and description, and add tags using arguments to the `@mcp.tool` decorator. The `name` sets the explicit tool name, `description` provides a custom description (overriding the docstring), and `tags` allow categorization. Other arguments like `enabled` and `exclude_args` are also mentioned for further customization.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_6

LANGUAGE: python
CODE:

```
@mcp.tool(
    name="find_products",           # Custom tool name for the LLM
    description="Search the product catalog with optional category filtering.", # Custom description
    tags={"catalog", "search"}      # Optional tags for organization/filtering
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    """Internal function description (ignored if description is provided above)."""
    # Implementation...
    print(f"Searching for '{query}' in category '{category}'")
    return [{"id": 2, "name": "Another Product"}]
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

TITLE: Configure Authenticated FastMCP Proxy with Bearer Token
DESCRIPTION: This Python example extends the remote server proxy by showing how to integrate authentication. It uses `BearerAuth` to create an authenticated client, which is then passed to the `FastMCP.as_proxy` method, enabling secure communication with protected remote services.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_11

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

TITLE: Define a Simple FastMCP Dice Roller Server
DESCRIPTION: This Python code defines a FastMCP server named 'Dice Roller' with a single tool function `roll_dice` that simulates rolling 6-sided dice. It uses the `fastmcp` library to expose this functionality for Claude Desktop.
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

TITLE: Reading Resource Content using FastMCP Client (Python)
DESCRIPTION: This example illustrates reading content from both static resources and resources generated from templates using `read_resource()`. It takes a URI (string or `AnyUrl`) and returns a list of `mcp.types.TextResourceContents` or `mcp.types.BlobResourceContents`, allowing access to the resource's data.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_10

LANGUAGE: python
CODE:

```
# Read a static resource
readme_content = await client.read_resource("file:///path/to/README.md")
# readme_content -> list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]
print(readme_content[0].text) # Assuming text

# Read a resource generated from a template
weather_content = await client.read_resource("data://weather/london")
print(weather_content[0].text) # Assuming text JSON
```

---

TITLE: Common Pydantic Field Validation Options
DESCRIPTION: Lists and describes common validation options available with Pydantic's Field class, which can be used to enforce specific constraints on input parameters in FastMCP tools.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_32

LANGUAGE: APIDOC
CODE:

```
Validation Options for Pydantic Field:
- ge, gt (Number): Greater than (or equal) constraint
- le, lt (Number): Less than (or equal) constraint
- multiple_of (Number): Value must be a multiple of this number
- min_length, max_length (String, List, etc.): Length constraints
- pattern (String): Regular expression pattern constraint
- description (Any): Human-readable description (appears in schema)
```

---

TITLE: Defining Basic Dynamic Resources with @mcp.resource
DESCRIPTION: This Python code demonstrates how to define simple dynamic resources using the `@mcp.resource` decorator. It shows examples of returning a plain string and a dictionary (which is automatically serialized to JSON), highlighting the use of unique URIs for resource access.
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
        "features": ["tools", "resources"]
    }
```

---

TITLE: Mounting FastMCP Server in Starlette Application
DESCRIPTION: Explains how to mount a FastMCP ASGI application within a larger Starlette application. This allows FastMCP functionality to be exposed under a specific URL path within an existing web service. It highlights the critical step of passing the FastMCP app's `lifespan` context to the parent Starlette app for proper initialization.
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

TITLE: Obtaining FastMCP Starlette App Instance
DESCRIPTION: Demonstrates how to retrieve a Starlette ASGI application instance from a FastMCP server using the `http_app()` method. This method is recommended for Streamable HTTP transport, while `http_app(transport="sse")` can be used for legacy SSE transport. The example also shows how to define a simple tool within FastMCP.
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

TITLE: Deploy FastMCP Server Locally with ngrok
DESCRIPTION: These commands demonstrate how to run the FastMCP server locally and expose it to the internet using `ngrok`. The first command starts the Python server, and the second command creates an `ngrok` tunnel for port 8000, making the server publicly accessible.
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

TITLE: Access Authenticated Token Claims in FastMCP Tools
DESCRIPTION: This Python example shows how to use the `get_access_token()` dependency function within a FastMCP tool to retrieve `AccessToken` information. It demonstrates accessing `client_id` (user ID) and `scopes`, and how to enforce permission checks based on granted scopes.
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

TITLE: Calling a FastMCP Tool with a Progress Handler
DESCRIPTION: Demonstrates how to provide a `progress_handler` to `call_tool()`. This handler receives updates during the execution of a long-running tool, allowing the client application to display real-time progress to the user.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_7

LANGUAGE: Python
CODE:

```
result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )
```

---

TITLE: Run FastMCP Server Asynchronously with run_async()
DESCRIPTION: This snippet illustrates how to run a FastMCP server within an existing asynchronous context using the `run_async()` method. It defines a simple tool and demonstrates calling `run_async()` from an `async def main()` function, emphasizing its use in async applications to avoid event loop conflicts.
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
    await mcp.run_async(transport="streamable-http")

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: Adding Descriptions to Tool Arguments with ArgTransform
DESCRIPTION: Demonstrates how to use `ArgTransform` to add a helpful description to a tool argument, improving LLM understanding. The example shows adding a detailed description to the `user_id` argument of a `find_user` tool.
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

TITLE: Calling a FastMCP Tool with a Timeout
DESCRIPTION: Illustrates how to set a timeout for a `call_tool()` invocation. If the tool execution exceeds the specified timeout (e.g., 2 seconds), the call will abort, preventing long-running operations from blocking the client indefinitely.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_6

LANGUAGE: Python
CODE:

```
result = await client.call_tool("long_running_task", {"param": "value"}, timeout=2.0)
```

---

TITLE: Generate Python Code Examples with LLM and System Prompt
DESCRIPTION: This example demonstrates using `ctx.sample` with both a user message and a system prompt to guide the LLM. It instructs the LLM to act as an expert Python programmer and generate concise, working code examples for a given concept without explanations.
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

TITLE: Add Custom GET Route to FastMCP Server
DESCRIPTION: This Python snippet demonstrates how to add a custom GET route, `/health`, to a FastMCP server using the `@custom_route` decorator. It utilizes Starlette's `Request` and `PlainTextResponse` to handle the incoming request and return a simple 'OK' response. This approach is suitable for adding basic, standalone endpoints like health checks to a FastMCP application.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_11

LANGUAGE: Python
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

TITLE: Adding Custom Middleware to FastMCP ASGI App
DESCRIPTION: Illustrates how to integrate custom Starlette middleware, such as `CORSMiddleware`, directly into a FastMCP ASGI application. Middleware instances are passed as a list to the `middleware` argument during the `http_app()` creation, allowing for centralized request processing.
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
    Middleware(CORSMiddleware, allow_origins=["*"]),
]

# Create ASGI app with custom middleware
http_app = mcp.http_app(middleware=custom_middleware)
```

---

TITLE: Add FastMCP as Project Dependency with uv
DESCRIPTION: This command adds FastMCP as a dependency to your project using the uv package manager. It integrates FastMCP into your project's environment, making it available for use.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_0

LANGUAGE: bash
CODE:

```
uv add fastmcp
```

---

TITLE: Hiding Tool Arguments with ArgTransform and Default Value
DESCRIPTION: Explains how to hide tool arguments from the LLM using `hide=True` in `ArgTransform`, typically for sensitive or internal parameters like API keys. It demonstrates hiding an `api_key` argument and supplying its value automatically from an environment variable.
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

TITLE: Connect to Remote Authenticated FastMCP Server
DESCRIPTION: This Python snippet shows how to configure a `fastmcp.Client` to connect to a remote FastMCP server using an HTTPS endpoint and `BearerAuth` for authentication. This allows the same Gemini integration code to work with various FastMCP server configurations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_4

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

mcp_client = Client(
    "https://my-server.com/sse",
    auth=BearerAuth("<your-token>")
)
```

---

TITLE: Install FastMCP Server with CLI
DESCRIPTION: This command installs a FastMCP server using the `fastmcp install` CLI tool. It automatically handles configuration and dependency management, looking for a server object like `mcp`, `server`, or `app` in your file.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
fastmcp install server.py
```

---

TITLE: Injecting Context into FastMCP Prompt Functions
DESCRIPTION: Shows how to inject the `Context` object into a FastMCP prompt function. This allows the prompt to incorporate contextual information or interact with MCP capabilities while generating text, such as requesting LLM sampling.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_2

LANGUAGE: Python
CODE:

```
@mcp.prompt
async def data_analysis_request(dataset: str, ctx: Context) -> str:
    """Generate a request to analyze data with contextual information."""
    # Context is available as the ctx parameter
    return f"Please analyze the following dataset: {dataset}"
```

---

TITLE: Replacing Tool Logic with a Transform Function in FastMCP
DESCRIPTION: This example shows how to define an asynchronous `transform_fn` that completely replaces the original logic of a FastMCP tool. The arguments of this function dynamically determine the new tool's schema, allowing for custom validation or post-processing, and can be passed to `Tool.from_tool`.
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

TITLE: Constrain Parameters with Literal Types in FastMCP Tools
DESCRIPTION: This snippet illustrates how to use `Literal` types from the `typing` module to constrain tool parameters to a predefined set of exact values. Literal types provide strong input validation, help LLMs understand acceptable inputs, and create clear schemas for clients, ensuring parameters like 'order' or 'algorithm' only receive specified values.
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

TITLE: Customize FastMCP Streamable HTTP Server Host, Port, and Path
DESCRIPTION: This example illustrates how to configure a FastMCP server using Streamable HTTP with custom host, port, path, and log level settings. The corresponding client code is provided to connect to the server at the specified custom address.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_7

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

mcp = FastMCP()

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
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

TITLE: FastMCP Supported Type Annotations
DESCRIPTION: Lists the various type annotations supported by FastMCP for tool parameters, including basic types, collections, optional types, constrained types, and Pydantic models. These types allow for rich and precise schema generation for LLMs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_4

LANGUAGE: APIDOC
CODE:

```
Type Annotation | Example | Description
----------------------|-----------------------------|-----------------------------------------
Basic types           | `int`, `float`, `str`, `bool` | Simple scalar values
Binary data           | `bytes`                     | Binary content
Date and Time         | `datetime`, `date`, `timedelta` | Date and time objects
Collection types      | `list[str]`, `dict[str, int]`, `set[int]` | Collections of items
Optional types        | `float | None`, `Optional[float]`| Parameters that may be null/omitted
Union types           | `str | int`, `Union[str, int]`| Parameters accepting multiple types
Constrained types     | `Literal["A", "B"]`, `Enum`   | Parameters with specific allowed values
Paths                 | `Path`                      | File system paths
UUIDs                 | `UUID`                      | Universally unique identifiers
Pydantic models       | `UserData`                  | Complex structured data
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

TITLE: Proxying FastMCP Servers with Python
DESCRIPTION: Shows how to use `FastMCP.as_proxy` to create a proxy for an existing MCP server, enabling bridging of transports or adding a frontend. The example proxies a remote HTTP SSE server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_9

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP, Client

backend = Client("http://example.com/mcp/sse")
proxy = FastMCP.as_proxy(backend, name="ProxyServer")
# Now use the proxy like any FastMCP server
```

---

TITLE: Create FastMCP Proxy from Single Server Configuration
DESCRIPTION: This snippet demonstrates how to initialize a FastMCP proxy using a Python dictionary that defines a single MCP server. It shows how to configure the server URL and transport, and then run the proxy, typically for local access via stdio.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_5

LANGUAGE: python
CODE:

```
config = {
    "mcpServers": {
        "default": {  # For single server configs, 'default' is commonly used
            "url": "https://example.com/mcp",
            "transport": "streamable-http"
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

TITLE: Define a FastMCP Prompt for LLM Guidance
DESCRIPTION: Shows how to create a reusable message template for guiding Large Language Models (LLMs) using the `@mcp.prompt` decorator. Prompts help structure interactions and provide context for AI models.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_4

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

TITLE: Running FastMCP Server using CLI Command
DESCRIPTION: Shows how to start a FastMCP server using the `fastmcp run` command-line interface. This method automatically handles server execution and ignores the `__main__` block.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/quickstart.mdx#_snippet_5

LANGUAGE: bash
CODE:

```
fastmcp run my_server.py:mcp
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

TITLE: Define FastMCP Tool with Optional Arguments
DESCRIPTION: Demonstrates how FastMCP handles required and optional function parameters based on Python's standard conventions. Parameters without default values are required, while those with default values are optional. The `| None` union type also indicates an optional parameter.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_5

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

TITLE: FastMCP Client Initialization with String Bearer Token
DESCRIPTION: Demonstrates the simplest way to authenticate a FastMCP client by passing a raw Bearer token string to the `auth` parameter. The client automatically prefixes the token with 'Bearer'.
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

TITLE: Test FastMCP Server with Client
DESCRIPTION: This Python script (`api_client.py`) demonstrates how to connect to the running FastMCP server using `fastmcp.Client`. It lists the automatically generated tools (e.g., `get_users`, `get_user_by_id`) and then calls the `get_user_by_id` tool to fetch and print user data from the live JSONPlaceholder API, verifying the server's functionality.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/tutorials/rest-api.mdx#_snippet_2

LANGUAGE: python
CODE:

```
import asyncio
from fastmcp import Client

async def main():
    # Connect to the MCP server we just created
    async with Client("http://127.0.0.1:8000/mcp") as client:

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

TITLE: Generate Development Tokens with RSAKeyPair
DESCRIPTION: This Python snippet demonstrates how to generate a new RSA key pair, configure a `BearerAuthProvider` with the public key, and create a test JWT token using `RSAKeyPair.create_token()`. This utility is strictly for development and testing, not production environments.
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

TITLE: Define and Use Python Enums in FastMCP Tools
DESCRIPTION: Demonstrates how to define a Python Enum class and use it as a parameter type in a FastMCP tool. FastMCP automatically coerces string values from clients into the corresponding Enum member, providing type safety and validation.
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

TITLE: Define Basic FastMCP Prompts with @mcp.prompt Decorator
DESCRIPTION: Demonstrates defining prompt functions using the `@mcp.prompt` decorator. Examples include returning a simple string (auto-converted to a user message) and returning a `PromptMessage` object for explicit control over message role and content. This enables creating reusable message templates for LLMs.
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

TITLE: Complete FastMCP Server with Bearer Authentication Example
DESCRIPTION: This comprehensive example demonstrates a full FastMCP server setup with bearer token authentication. It includes RSA key pair generation, `BearerAuthProvider` configuration, and a sample `roll_dice` tool. The access token is printed for development purposes, but this practice is strongly discouraged in production environments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_8

LANGUAGE: Python
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
    mcp.run(transport="sse", port=8000)
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

TITLE: Running FastMCP Server with Python
DESCRIPTION: Demonstrates how to start a FastMCP server using `mcp.run()` within a Python script. It shows the default STDIO transport and an example of configuring HTTP transport with host and port.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_7

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

    # To use a different transport, e.g., HTTP:
    # mcp.run(transport="streamable-http", host="127.0.0.1", port=9000)
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

TITLE: Install FastMCP for Development
DESCRIPTION: These commands set up the FastMCP development environment by cloning the repository and navigating into it. It then installs all project and development dependencies using uv, preparing the environment for contributions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_4

LANGUAGE: bash
CODE:

```
git clone https://github.com/jlowin/fastmcp.git
cd fastmcp
uv sync
```

---

TITLE: Creating Nested Mounts with FastMCP in Starlette
DESCRIPTION: Demonstrates the concept of creating complex routing structures by nesting mounts within a Starlette application. This allows for hierarchical organization of different services or sub-applications, including a FastMCP server, under various URL prefixes.
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

TITLE: Configuring Client and Per-Call Timeouts in fastmcp Python
DESCRIPTION: This snippet demonstrates how to set a global timeout for all requests using the `Client` constructor and how to override it for specific `call_tool` invocations. It also shows how to catch `McpError` for timeout-related exceptions. Timeout behavior varies by transport type (SSE vs. HTTP).
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_14

LANGUAGE: Python
CODE:

```
client = Client(
    my_mcp_server,
    timeout=5.0  # Default timeout in seconds
)

async with client:
    # This uses the global 5-second timeout
    result1 = await client.call_tool("quick_task", {"param": "value"})

    # This specifies a 10-second timeout for this specific call
    result2 = await client.call_tool("slow_task", {"param": "value"}, timeout=10.0)

    try:
        # This will likely timeout
        result3 = await client.call_tool("medium_task", {"param": "value"}, timeout=0.01)
    except McpError as e:
        # Handle timeout error
        print(f"The task timed out: {e}")
```

---

TITLE: Logging in FastMCP Tools with Context
DESCRIPTION: Demonstrates how to send log messages (debug, info, warning, error) from an `mcp.tool` function back to the MCP client using the `ctx` object. This is crucial for debugging and providing visibility into function execution during a request, including handling exceptions with appropriate logging levels.
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
ctx.debug(message: str)
ctx.info(message: str)
ctx.warning(message: str)
ctx.error(message: str)
ctx.log(level: Literal["debug", "info", "warning", "error"], message: str, logger_name: str | None = None)
```

---

TITLE: FastMCP Pydantic Field Validation as Default Values
DESCRIPTION: Illustrates an alternative method for applying Pydantic Field validation by using Field as a default value for parameters in FastMCP tools. This approach supports various constraints for numbers, strings, and collections, ensuring data integrity.
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

TITLE: Generate RSA Key Pair and Access Token for FastMCP Server
DESCRIPTION: This snippet demonstrates how to generate an RSA key pair using FastMCP's `RSAKeyPair` utility and create an access token for a specified audience, which is used for server authentication. It highlights the use of `RSAKeyPair.generate()` and `key_pair.create_token()`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp.server.auth.providers.bearer import RSAKeyPair

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")
```

---

TITLE: Call FastMCP Server with Gemini Python SDK
DESCRIPTION: This Python script demonstrates how to connect to a FastMCP server using `fastmcp.Client` and integrate its tools with the Google Gemini API. It instantiates a Gemini client, passes the FastMCP client session as a tool configuration, and generates content using a Gemini model, leveraging the `roll_dice` tool from the FastMCP server.
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
                tools=[mcp_client.session]  # Pass the FastMCP client session
            )
        )
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

TITLE: FastMCP Client Default OAuth Configuration
DESCRIPTION: This snippet demonstrates the simplest way to configure a FastMCP client to use OAuth by passing the string 'oauth' to the `auth` parameter. FastMCP automatically handles default OAuth settings, suitable for basic authentication needs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_0

LANGUAGE: Python
CODE:

```
from fastmcp import Client

# Uses default OAuth settings
async with Client("https://fastmcp.cloud/mcp", auth="oauth") as client:
    await client.ping()
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

TITLE: Define Parameters with Python Collection Types in FastMCP
DESCRIPTION: This snippet showcases how FastMCP supports standard Python collection types like `list`, `dict`, `set`, and `tuple` for tool parameters. These types, including nested combinations, allow for the representation of complex data structures, with JSON strings automatically parsed and converted to the appropriate Python collection.
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

TITLE: Configure FastMCP Server with Bearer Authentication
DESCRIPTION: This snippet shows how to initialize a `BearerAuthProvider` using the public key from a previously generated RSA key pair and a specified audience. It then integrates this authentication provider into a `FastMCP` server instance, naming it "Dice Roller".
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

TITLE: Clone Repository and Setup Development Environment
DESCRIPTION: Instructions for cloning the FastMCP repository and setting up the development environment using `uv` for dependency management.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_13

LANGUAGE: bash
CODE:

```
git clone https://github.com/jlowin/fastmcp.git
cd fastmcp
```

LANGUAGE: bash
CODE:

```
uv sync
```

---

TITLE: Reporting Progress in FastMCP Long-Running Operations
DESCRIPTION: Illustrates how to notify the client about the progress of long-running operations using `ctx.report_progress`. This feature allows clients to display progress indicators, enhancing the user experience. It requires the client to have sent a `progressToken` in the initial request; otherwise, calls will have no effect.
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
ctx.report_progress(progress: float, total: float | None = None)
  progress: Current progress value (e.g., 24)
  total: Optional total value (e.g., 100). If provided, clients may interpret this as a percentage.
```

---

TITLE: Define Synchronous and Asynchronous FastMCP Tools
DESCRIPTION: Illustrates how to define both synchronous (`def`) and asynchronous (`async def`) functions as FastMCP tools. Synchronous tools are suitable for CPU-bound tasks, while asynchronous tools are ideal for I/O-bound operations like network requests, preventing server blocking.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_10

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

TITLE: FastMCP: Catch-All Rule for Route Exclusion
DESCRIPTION: This example shows how to implement a catch-all exclusion rule in FastMCP route mapping. By placing a `RouteMap` with `MCPType.EXCLUDE` at the end of the `route_maps` list, all routes not explicitly handled by preceding custom maps will be excluded. This effectively creates an allow-list approach for route processing.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_5

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

mcp = FastMCP.from_openapi(
    ...,
    route_maps=[
        # custom mapping logic goes here
        ...,
        # exclude all remaining routes
        RouteMap(mcp_type=MCPType.EXCLUDE),
    ],
)
```

---

TITLE: Standard HTTP Bearer Token Header
DESCRIPTION: Illustrates the common format for including a Bearer token in the `Authorization` header of an HTTP request, following the `Bearer` scheme.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/bearer.mdx#_snippet_0

LANGUAGE: http
CODE:

```
Authorization: Bearer <token>
```

---

TITLE: FastMCP `run` Command Usage Examples
DESCRIPTION: Practical examples demonstrating how to run a local FastMCP server with custom transport and port, connect to a remote server as a stdio proxy, and specify a log level for a remote connection.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_5

LANGUAGE: bash
CODE:

```
# Run a local server with Streamable HTTP transport on a custom port
fastmcp run server.py --transport streamable-http --port 8000
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

TITLE: Customizing Resource Metadata with @mcp.resource
DESCRIPTION: This Python example illustrates how to provide explicit metadata for a resource using arguments within the `@mcp.resource` decorator. It shows how to set a custom URI, name, description, MIME type, and categorization tags, overriding inferred values.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_2

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

TITLE: FastMCP `ctx.sample` Method API Reference
DESCRIPTION: Detailed API documentation for the `ctx.sample` method, which enables FastMCP tools to interact with the client's LLM for text generation or processing. It supports various parameters for controlling the sampling behavior, including messages, system prompts, temperature, token limits, and model preferences.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_8

LANGUAGE: APIDOC
CODE:

```
ctx.sample(
  messages: str | list[str | SamplingMessage],
  system_prompt: str | None = None,
  temperature: float | None = None,
  max_tokens: int | None = None,
  model_preferences: ModelPreferences | str | list[str] | None = None
) -> TextContent | ImageContent

Parameters:
  messages: A string or list of strings/message objects to send to the LLM.
  system_prompt: Optional system prompt to guide the LLM's behavior.
  temperature: Optional sampling temperature (controls randomness).
  max_tokens: Optional maximum number of tokens to generate (defaults to 512).
  model_preferences: Optional model selection preferences (e.g., a model hint string, list of hints, or a ModelPreferences object).

Returns:
  The LLM's response as TextContent or ImageContent.
```

---

TITLE: Configuring LLM Sampling Handler with FastMCP Client (Python)
DESCRIPTION: This code shows how to integrate an LLM sampling handler with the FastMCP client. The `sampling_handler` is an asynchronous function that receives a list of `SamplingMessage` objects, `SamplingParams`, and `RequestContext` from the server, and is responsible for generating and returning a string completion, exemplified here using the `marvin` library.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/advanced-features.mdx#_snippet_3

LANGUAGE: Python
CODE:

```
import marvin
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
    return await marvin.say_async(
        message=[m.content.text for m in messages],
        instructions=params.systemPrompt,
    )

client = Client(
    ...,
    sampling_handler=sampling_handler,
)
```

---

TITLE: Transforming Tool Metadata with `Tool.from_tool()` in fastmcp
DESCRIPTION: This Python snippet demonstrates how to use `Tool.from_tool()` to create a new tool from an existing generic `search` tool. It specifically shows how to update the `name` to `find_products` and provide a more descriptive, domain-specific `description` for an e-commerce context, making it more intuitive for an LLM client. The transformed tool is then added to the `FastMCP` instance.
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

TITLE: Running FastMCP ASGI App with Uvicorn (Python)
DESCRIPTION: Provides a Python script example for running a FastMCP ASGI application using the `uvicorn` server. This method allows embedding the server startup directly within your application's main execution block, typically guarded by `if __name__ == "__main__":`.
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

TITLE: Run FastMCP Server with CLI Transport Options
DESCRIPTION: Illustrates how to specify transport options and other configurations when running a FastMCP server via the `fastmcp run` command, such as setting the transport to SSE and defining a port.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp run server.py --transport sse --port 9000
```

---

TITLE: Install and Run Pre-commit Hooks for Static Checks
DESCRIPTION: Commands to install `pre-commit` hooks for automated code formatting, linting, and type-checking, and how to run them manually.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_15

LANGUAGE: bash
CODE:

```
uv run pre-commit install
```

LANGUAGE: bash
CODE:

```
pre-commit run --all-files
```

LANGUAGE: bash
CODE:

```
uv run pre-commit run --all-files
```

---

TITLE: FastMCP: Defining Resource with Wildcard Path Parameter
DESCRIPTION: This snippet demonstrates how to define a FastMCP resource using a wildcard parameter (`{path*}`) to capture variable-length path segments. It shows how to retrieve a file from a repository where the path ends with `template.py`, illustrating the use of both standard and wildcard parameters in the URI template and function signature.
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

TITLE: Registering FastMCP Tools and Resources with MCPMixin in Python
DESCRIPTION: This snippet demonstrates how to define a class `MyComponent` that inherits from `MCPMixin` to register its methods as `FastMCP` tools and resources. It uses `@mcp_tool` and `@mcp_resource` decorators to mark methods for registration. The `register_all()` method is then called on an instance of `MyComponent` to register these methods with a `FastMCP` server, optionally using a `prefix` to avoid naming collisions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/src/fastmcp/contrib/mcp_mixin/README.md#_snippet_0

LANGUAGE: Python
CODE:

```
from fastmcp import FastMCP
from fastmcp.contrib.mcp_mixin import MCPMixin, mcp_tool, mcp_resource

class MyComponent(MCPMixin):
    @mcp_tool(name="my_tool", description="Does something cool.")
    def tool_method(self):
        return "Tool executed!"

    @mcp_resource(uri="component://data")
    def resource_method(self):
        return {"data": "some data"}

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

TITLE: Handle Image and Null Return Values in FastMCP Tools
DESCRIPTION: Demonstrates how FastMCP tools can return complex data types like images using the `fastmcp.utilities.types.Image` helper, and how to return `None` for tools that perform side effects without sending data back to the client. It includes an example of generating an image with Pillow and returning it.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_11

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

TITLE: StreamableHttpTransport Class Overview
DESCRIPTION: Provides an overview of the `StreamableHttpTransport` class, detailing how it's inferred from URLs and its compatibility with FastMCP servers running in `streamable-http` mode. This is the recommended transport for web-based deployments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_0

LANGUAGE: APIDOC
CODE:

```
Class: fastmcp.client.transports.StreamableHttpTransport
Inferred From: URLs starting with http:// or https:// (default for HTTP URLs since v2.3.0) that do not contain /sse/ in the path
Server Compatibility: Works with FastMCP servers running in streamable-http mode
```

---

TITLE: Initializing FastMCP Client from Configuration - Python
DESCRIPTION: This snippet illustrates how to initialize a FastMCP client using a Python dictionary that conforms to the MCPConfig schema. This method allows for defining multiple MCP servers (e.g., local and remote) within a single configuration, enabling the client to connect to and manage interactions with these diverse servers through a unified interface.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_1

LANGUAGE: Python
CODE:

```
from fastmcp import Client

config = {
    "mcpServers": {
        "local": {"command": "python", "args": ["local_server.py"]},
        "remote": {"url": "https://example.com/mcp"}
    }
}

client_config = Client(config)
```

---

TITLE: FastMCP: Resource Template with Optional Parameters and Defaults
DESCRIPTION: This example illustrates how FastMCP handles function parameters with default values that are not explicitly included in the URI template. It shows a search function where `max_results` and `include_archived` are optional, allowing clients to request resources with only the required `query` parameter, while FastMCP uses the defined defaults for the others.
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

TITLE: FastMCP 2.7: Python Decorator Usage for Tools
DESCRIPTION: Demonstrates the new 'naked' decorator usage introduced in FastMCP 2.7, allowing for a more Pythonic way to register functions as tools. This enhancement also ensures decorators return the created objects for improved usability.
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

TITLE: Call FastMCP Server via Anthropic Messages API
DESCRIPTION: This Python code demonstrates how to call the deployed FastMCP server using the Anthropic Messages API. It initializes an Anthropic client, constructs a message with a user prompt, and specifies the MCP server URL. It also includes the required `anthropic-beta` header for MCP client functionality.
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
            "url": f"{url}/sse",
            "name": "dice-server",
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)
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

TITLE: Configuring FastMCP Transport Settings in Python
DESCRIPTION: Explains how to apply transport-specific settings when running a FastMCP server using `mcp.run()` or `mcp.run_async()`. It includes examples for configuring host, port, and overriding the global log level for HTTP transport.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_12

LANGUAGE: python
CODE:

```
# Configure transport when running
mcp.run(
    transport="streamable-http",
    host="0.0.0.0",           # Bind to all interfaces
    port=9000,                # Custom port
    log_level="DEBUG"        # Override global log level
)

# Or for async usage
await mcp.run_async(
    transport="streamable-http",
    host="127.0.0.1",
    port=8080
)
```

---

TITLE: Configure FastMCP with Custom Route Maps for GET Requests
DESCRIPTION: This Python code demonstrates how to configure a FastMCP server to use custom `RouteMap` rules. It shows how to convert `GET` requests with path parameters (e.g., `/users/{id}`) into `ResourceTemplates` and other `GET` requests (e.g., `/users`) into `Resources`, while other methods default to `Tools`. It utilizes `httpx` for the client and a simplified OpenAPI specification for JSONPlaceholder.
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
    mcp.run(transport="streamable-http", port=8000)
```

---

TITLE: Implement Detailed and Masked Error Responses in FastMCP
DESCRIPTION: This example illustrates the use of `ResourceError` to explicitly send error details to clients regardless of masking settings, and `ValueError` for errors that can be masked. It shows how different exceptions behave based on the `mask_error_details` configuration.
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

TITLE: Add Metadata Annotations to FastMCP Tool
DESCRIPTION: Provides an example of how to apply annotations like title, readOnlyHint, and openWorldHint to a tool using the annotations parameter in the @mcp.tool decorator.
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

TITLE: Define a FastMCP Resource for Data Access
DESCRIPTION: Illustrates how to expose a static data source as a resource using the `@mcp.resource` decorator. Resources allow clients to read application configuration or other fixed data, identified by a URI.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_2

LANGUAGE: python
CODE:

```
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}
```

---

TITLE: Configuring Multi-Server FastMCP Client and Calling Tools
DESCRIPTION: Demonstrates how to configure a FastMCP client to connect to multiple servers (HTTP and local stdio) using a dictionary-based configuration. It shows how to call tools and read resources from different servers by prefixing the server name to the tool/resource identifier within an `async with` block.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_2

LANGUAGE: Python
CODE:

```
config = {
    "mcpServers": {
        # A remote HTTP server
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "streamable-http"
        },
        # A local server running via stdio
        "assistant": {
            "command": "python",
            "args": ["./my_assistant_server.py"],
            "env": {"DEBUG": "true"}
        }
    }
}

# Create a client that connects to both servers
client = Client(config)

async def main():
    async with client:
        # Access tools from different servers with prefixes
        weather_data = await client.call_tool("weather_get_forecast", {"city": "London"})
        response = await client.call_tool("assistant_answer_question", {"question": "What's the capital of France?"})

        # Access resources with prefixed URIs
        weather_icons = await client.read_resource("weather://weather/icons/sunny")
        templates = await client.read_resource("resource://assistant/templates/list")

        print(f"Weather: {weather_data}")
        print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
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

TITLE: FastMCP Client OAuth Helper Configuration
DESCRIPTION: This example shows how to use the `fastmcp.client.auth.OAuth` helper for more granular control over the OAuth flow. The `OAuth` instance is passed to the `auth` parameter of the `Client`, enabling advanced configuration and managing PKCE for enhanced security.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/oauth.mdx#_snippet_1

LANGUAGE: Python
CODE:

```
from fastmcp import Client
from fastmcp.client.auth import OAuth

oauth = OAuth(mcp_url="https://fastmcp.cloud/mcp")

async with Client("https://fastmcp.cloud/mcp", auth=oauth) as client:
    await client.ping()
```

---

TITLE: Connect to FastMCP Server via In-Memory Transport
DESCRIPTION: Shows how to connect directly to a `fastmcp.server.FastMCP` instance within the same Python process using `fastmcp.client.transports.FastMCPTransport`. This method is highly efficient due to in-memory queues and is ideal for testing FastMCP servers. The transport is automatically inferred from a `FastMCP` server instance.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_15

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

---

TITLE: Authenticating Streamable HTTP Transport with Headers
DESCRIPTION: Illustrates how to include custom HTTP headers, such as an `Authorization` token, when instantiating `StreamableHttpTransport`. This is essential for connecting to FastMCP servers that require authentication.
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

TITLE: Configuring Direct vs. Proxy Mounting in FastMCP
DESCRIPTION: Illustrates the two modes of mounting in FastMCP: direct and proxy. Direct mounting is the default and accesses objects in memory, while proxy mounting treats the mounted server as a separate entity, preserving its full client lifecycle.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_3

LANGUAGE: python
CODE:

```
# Direct mounting (default when no custom lifespan)
main_mcp.mount("api", api_server)

# Proxy mounting (preserves full client lifecycle)
main_mcp.mount("api", api_server, as_proxy=True)
```

---

TITLE: Run a FastMCP server from the command line
DESCRIPTION: This command-line snippet shows how to execute a FastMCP server defined in a Python file (e.g., 'server.py'). The 'fastmcp run' command starts the server, making its defined tools and resources accessible.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/README.md#_snippet_1

LANGUAGE: bash
CODE:

```
fastmcp run server.py
```

---

TITLE: Accessing Resources in FastMCP Tools
DESCRIPTION: Explains how to read data from resources registered with the FastMCP server using `ctx.read_resource`. This enables functions to access files, configuration, or dynamically generated content. The content is typically accessed via `content_list[0].content` and can be text or binary data.
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
ctx.read_resource(uri: str | AnyUrl) -> list[ReadResourceContents]
  uri: The resource URI to read
  Returns a list of resource content parts (usually containing just one item)
```

---

TITLE: Define FastMCP Resource Templates with URI Parameters
DESCRIPTION: This example illustrates how to create dynamic resource templates in FastMCP using the `@mcp.resource` decorator. It demonstrates defining templates with single and multiple URI placeholders (e.g., `{city}`, `{owner}/{repo}`), which map directly to function arguments, enabling on-demand resource generation based on client requests.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_10

LANGUAGE: Python
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

TITLE: Disabling Original Tool After Transformation in fastmcp
DESCRIPTION: This Python example illustrates the best practice of disabling the original tool after creating a transformed version using `Tool.from_tool()`. By calling `search.disable()`, you prevent the LLM from being confused by two similar tools, ensuring it only interacts with the newly enhanced `product_search_tool`.
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

TITLE: Configure FastMCP to Mask Error Details
DESCRIPTION: Demonstrates how to initialize a FastMCP instance with mask_error_details=True to prevent sensitive internal error information from being sent to client LLMs.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_13

LANGUAGE: python
CODE:

```
mcp = FastMCP(name="SecureServer", mask_error_details=True)
```

---

TITLE: Install Google Generative AI SDK
DESCRIPTION: This command installs the necessary Python package for interacting with the Google Generative AI SDK, which is required to use Gemini's API with FastMCP.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_1

LANGUAGE: bash
CODE:

```
pip install google-genai
```

---

TITLE: Injecting Context into FastMCP Resource and Template Functions
DESCRIPTION: Illustrates how to inject the `Context` object into FastMCP resource and resource template functions. This enables these functions to fetch personalized data or perform context-aware logging based on the current request, leveraging MCP capabilities.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_1

LANGUAGE: Python
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

TITLE: FastMCP CLI Commands Overview
DESCRIPTION: A summary of the main commands available in the FastMCP CLI, their purpose, and dependency management characteristics.
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
Command: version
  Purpose: Display version information
  Dependency Management: N/A
```

---

TITLE: Customize FastMCP Prompt Metadata with Decorator Arguments (Python)
DESCRIPTION: This example shows how to override the inferred name and description of a FastMCP prompt and add custom categorization tags using arguments directly in the `@mcp.prompt` decorator. It highlights the `name`, `description`, and `tags` parameters.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_4

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

TITLE: Access FastMCP Session Context in Functions
DESCRIPTION: Inject the `Context` object into any FastMCP-decorated function by adding a `ctx: Context` parameter. This provides access to session-specific capabilities such as logging, LLM sampling, HTTP requests, resource access, and progress reporting.
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

TITLE: Composing FastMCP Servers in Python
DESCRIPTION: Illustrates how to combine multiple FastMCP servers using `main.mount()` to organize applications into modular components. It shows a main server mounting a sub-server with a simple tool.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_8

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
main.mount("sub", sub)
```

---

TITLE: Create FastMCP Proxy from Multi-Server Configuration
DESCRIPTION: This example illustrates how to configure FastMCP to proxy requests to multiple MCP servers using a single configuration dictionary. Each server is automatically mounted with its config name as a prefix, allowing unified access to tools and resources across different services.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/proxy.mdx#_snippet_6

LANGUAGE: python
CODE:

```
from fastmcp import FastMCP

# Multi-server configuration
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "streamable-http"
        },
        "calendar": {
            "url": "https://calendar-api.example.com/mcp",
            "transport": "streamable-http"
        }
    }
}

# Create a proxy to multiple servers
composite_proxy = FastMCP.as_proxy(config, name="Composite Proxy")

# Tools and resources are accessible with prefixes:
# - weather_get_forecast, calendar_add_event
# - weather://weather/icons/sunny, calendar://calendar/events/today
```

---

TITLE: Tag FastMCP Components for Filtering
DESCRIPTION: Illustrates how to assign tags to FastMCP tools (and other components) using the `tags` parameter in the decorator. These tags enable selective exposure of components based on configurable include/exclude tag sets.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_5

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

TITLE: Register Static and Predefined Resources with FastMCP (Python)
DESCRIPTION: Beyond dynamic resources defined with `@mcp.resource`, FastMCP allows direct registration of static or predefined content using `mcp.add_resource()` and concrete `Resource` subclasses. This approach is suitable for exposing static files, simple text, or directory listings, offering fine-grained control over resource properties like URI, name, and MIME type.
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

TITLE: Reference for Advanced Context Properties
DESCRIPTION: This snippet documents advanced properties available on the `ctx` object. It includes `ctx.fastmcp` for accessing the server instance, and `ctx.session` and `ctx.request_context` for direct access to low-level MCP SDK objects. A warning is provided regarding the stability and understanding required for direct use of `session` and `request_context`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_14

LANGUAGE: APIDOC
CODE:

```
ctx Properties:
  fastmcp:
    Type: FastMCP
    Description: Access the server instance the context belongs to.
  session:
    Type: mcp.server.session.ServerSession
    Description: Access the raw ServerSession object. Direct use requires understanding the low-level MCP Python SDK and may be less stable than using methods provided directly on the Context object.
  request_context:
    Type: mcp.shared.context.RequestContext
    Description: Access the raw RequestContext object. Direct use requires understanding the low-level MCP Python SDK and may be less stable than using methods provided directly on the Context object.
```

---

TITLE: FastMCP Client Authentication with BearerAuth Helper Class
DESCRIPTION: Illustrates using the `BearerAuth` class for explicit Bearer token authentication. This class implements the `httpx.Auth` interface, providing a more structured way to pass the token.
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

TITLE: Hiding Tool Arguments with ArgTransform and Default Factory
DESCRIPTION: Illustrates using `default_factory` with `hide=True` to generate dynamic default values for hidden arguments, such as timestamps or unique IDs, for each tool call. This ensures visible parameters have static defaults for JSON schema representation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_8

LANGUAGE: python
CODE:

```
transform_args = {
    'timestamp': ArgTransform(
        hide=True,
        default_factory=lambda: datetime.now(),
    )
}
```

---

TITLE: FastMCP Transport Initialization with String Bearer Token
DESCRIPTION: Shows how to apply a Bearer token directly to a FastMCP transport instance, such as `StreamableHttpTransport`, before it's used by the client. This method also handles the 'Bearer' prefix automatically.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/auth/bearer.mdx#_snippet_2

LANGUAGE: python
CODE:

```
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    "http://fastmcp.cloud/mcp",
    auth="<your-token>",
)

async with Client(transport) as client:
    await client.ping()
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

TITLE: FastMCP: Registering Multiple URI Templates for a Single Function
DESCRIPTION: This snippet demonstrates how to register a single Python function with multiple FastMCP URI templates using stacked decorators. This pattern allows the same underlying function to be accessed via different resource identifiers, such as looking up a user by email or by name, providing flexible API access points.
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

TITLE: FastMCP: Importing a Subserver for Static Composition
DESCRIPTION: This Python example illustrates the `FastMCP.import_server()` method. It defines a `WeatherService` subserver with a tool and a resource, then imports it into a `MainApp` server. After import, the subserver's components are copied to the main server with a 'weather\_' prefix, demonstrating how static composition bundles functionalities.
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
    await main_mcp.import_server("weather", weather_mcp)

# Result: main_mcp now contains prefixed components:
# - Tool: "weather_get_forecast"
# - Resource: "data://weather/cities/supported"

if __name__ == "__main__":
    asyncio.run(setup())
    main_mcp.run()
```

---

TITLE: Listing Resources using FastMCP Client (Python)
DESCRIPTION: This snippet demonstrates how to retrieve a list of static resources using the `list_resources()` method of the FastMCP client. It returns a list of `mcp.types.Resource` objects, representing available static resources.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_8

LANGUAGE: python
CODE:

```
resources = await client.list_resources()
# resources -> list[mcp.types.Resource]
```

---

TITLE: Listing Resource Templates using FastMCP Client (Python)
DESCRIPTION: This snippet shows how to fetch a list of resource templates using the `list_resource_templates()` method. The method returns a list of `mcp.types.ResourceTemplate` objects, which can be used to generate dynamic resources.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_9

LANGUAGE: python
CODE:

```
templates = await client.list_resource_templates()
# templates -> list[mcp.types.ResourceTemplate]
```

---

TITLE: Configuring Static Roots with FastMCP Client (Python)
DESCRIPTION: This snippet demonstrates how to configure static roots for the FastMCP client by providing a list of strings directly to the `roots` parameter during client initialization. Roots inform the server about the client's accessible resources or boundaries, allowing the server to adjust its behavior accordingly.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/advanced-features.mdx#_snippet_4

LANGUAGE: Python
CODE:

```
from fastmcp import Client

client = Client(
    ...,
    roots=["/path/to/root1", "/path/to/root2"],
)
```

---

TITLE: FastMCP: Calling Parent Tool with forward() and Same Arguments
DESCRIPTION: Demonstrates how to use `forward()` within a `transform_fn` to call the parent tool. The example validates that input arguments `x` and `y` are positive before forwarding them to the `add` tool. Arguments in the `transform_fn` match the parent tool's arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_11

LANGUAGE: python
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

TITLE: FastMCP Client Authentication with Custom HTTP Headers
DESCRIPTION: Explains how to configure the FastMCP client to use custom HTTP headers for authentication, such as `X-API-Key`, by setting them directly on the transport instance. This is useful for non-standard authentication schemes.
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

TITLE: Authenticating SSE Transport with Headers
DESCRIPTION: Illustrates how to include custom HTTP headers, such as an `Authorization` token, when instantiating `SSETransport`. This is necessary for connecting to SSE-based FastMCP servers that require authentication.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/transports.mdx#_snippet_7

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

TITLE: FastMCP `run` Command Options
DESCRIPTION: Available command-line options for configuring the FastMCP `run` command, including transport protocol, host, port, and logging level.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
Option: Transport
  Flag: --transport, -t
  Description: Transport protocol to use (stdio, streamable-http, or sse)
Option: Host
  Flag: --host
  Description: Host to bind to when using http transport (default: 127.0.0.1)
Option: Port
  Flag: --port, -p
  Description: Port to bind to when using http transport (default: 8000)
Option: Log Level
  Flag: --log-level, -l
  Description: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```

---

TITLE: Retrieve Current Request and Client Information
DESCRIPTION: This tool shows how to access fundamental metadata about the current FastMCP request and the client that initiated it. It retrieves the unique `request_id` and the `client_id` from the `Context` object.
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

TITLE: Run FastMCP Server with MCP Inspector
DESCRIPTION: Starts a FastMCP server in an isolated environment with the MCP Inspector for testing. This command requires explicit dependency specification using `--with` or `--with-editable`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_6

LANGUAGE: bash
CODE:

```
fastmcp dev server.py
```

---

TITLE: Process Base64-Encoded Binary Data in FastMCP Tools
DESCRIPTION: Shows how to handle base64-encoded binary data by annotating a parameter as `Annotated[str, Field(description="Base64-encoded image data")]`. This approach requires manual decoding of the base64 string within the tool function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_26

LANGUAGE: python
CODE:

```
from typing import Annotated
from pydantic import Field

@mcp.tool
def process_image_data(
    image_data: Annotated[str, Field(description="Base64-encoded image data")]
):
    """Process an image from base64-encoded string.

    The client is expected to provide base64-encoded data as a string.
    You'll need to decode it manually.
    """
    # Manual base64 decoding
    import base64
    binary_data = base64.b64decode(image_data)
    # Process binary_data...
```

---

TITLE: Define Synchronous FastMCP Prompt (Python)
DESCRIPTION: This snippet shows a basic synchronous FastMCP prompt defined using `def`. It's suitable for prompts that do not perform I/O operations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_8

LANGUAGE: python
CODE:

```
# Synchronous prompt
@mcp.prompt
def simple_question(question: str) -> str:
    """Generates a simple question to ask the LLM."""
    return f"Question: {question}"
```

---

TITLE: FastMCP: Handling Parent Tool Arguments with **kwargs in transform_fn
DESCRIPTION: Demonstrates how a `transform_fn` can accept `**kwargs`to receive all transformed arguments from the parent tool. This allows for flexible validation where not all arguments need to be explicitly listed in the`transform_fn`signature. The example validates a renamed argument`a`while passing`b`via`\*\*kwargs`to`forward()`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_14

LANGUAGE: python
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

TITLE: Access FastMCP Context in Resource Functions (Python)
DESCRIPTION: Resources and resource templates can access additional FastMCP information and features through the `Context` object. By adding a parameter with a type annotation of `Context` to a resource function, developers can retrieve details like `request_id` for logging or specific request handling.
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

TITLE: Specify Environment Variables for FastMCP Servers
DESCRIPTION: This JSON configuration shows how to pass environment variables to a FastMCP server. It's crucial for Claude Desktop's isolated environment, allowing servers to access necessary keys or debug flags like `API_KEY` and `DEBUG`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/claude-desktop.mdx#_snippet_9

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

TITLE: Run FastMCP Server with Explicit STDIO Transport
DESCRIPTION: Shows how to explicitly specify `stdio` as the transport option when calling the `mcp.run()` method. While STDIO is the default, explicit declaration clarifies intent, especially for local tools and command-line integrations where clients manage server processes.
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

TITLE: Pass Command Line Arguments to FastMCP Server via CLI
DESCRIPTION: Explains how to pass custom command-line arguments to a FastMCP server when running it via the CLI, by placing them after a `--` separator. This is useful for providing configuration files, database paths, or other runtime options.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/deployment/running-server.mdx#_snippet_4

LANGUAGE: bash
CODE:

```
fastmcp run config_server.py -- --config config.json
fastmcp run database_server.py -- --database-path /tmp/db.sqlite --debug
```

---

TITLE: Implement FastMCP Resource Templates with Wildcard Parameters
DESCRIPTION: This snippet demonstrates FastMCP's extension for wildcard parameters in resource templates. It contrasts standard parameters, which match single path segments, with wildcard parameters (`{param*}`), which can capture multiple path segments including slashes, allowing for more flexible URI matching.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_11

LANGUAGE: Python
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

TITLE: Configuring Resource Prefix Format in FastMCP (Environment Variable)
DESCRIPTION: Shows how to configure the resource prefix format using the `FASTMCP_RESOURCE_PREFIX_FORMAT` environment variable. This provides a system-wide way to set the prefixing behavior for FastMCP applications.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_6

LANGUAGE: bash
CODE:

```
FASTMCP_RESOURCE_PREFIX_FORMAT=protocol
```

---

TITLE: FastMCP: Calling Parent Tool with forward() and Renamed Arguments
DESCRIPTION: Illustrates using `forward()` when the `transform_fn` has different argument names than the parent tool. `ArgTransform` is used to map `x` to `a` and `y` to `b`. The `transform_fn` validates `a` and `b` before calling `forward()` with the renamed arguments, which `forward()` automatically maps back to the parent tool's original arguments.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_12

LANGUAGE: python
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

TITLE: Customizing FastMCP ASGI App Paths
DESCRIPTION: Illustrates how to change the default mount paths for FastMCP ASGI applications. By default, Streamable HTTP transport is mounted at `/mcp` and SSE transport at `/sse`. This snippet shows how to specify a custom path like `/custom-mcp-path` or `/custom-sse-path` using the `path` argument in `http_app()`.
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

TITLE: Install OpenAI Python SDK
DESCRIPTION: This command installs the necessary OpenAI Python SDK, which is required to interact with the OpenAI API, including the Responses API.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
pip install openai
```

---

TITLE: FastMCP: Calling Parent Tool with forward_raw() and Renamed Arguments
DESCRIPTION: Shows how to use `forward_raw()` to call the parent tool directly, bypassing all argument transformations. Even with `ArgTransform` configurations, `forward_raw()` requires calling the parent tool with its original argument names (`x` and `y`), using the transformed arguments (`a` and `b`) from the `transform_fn`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_13

LANGUAGE: python
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

TITLE: Accessing Context via Dependency Function in FastMCP
DESCRIPTION: Provides an alternative method to retrieve the active `Context` object using `fastmcp.server.dependencies.get_context()`. This is useful for code that cannot easily accept context as a parameter, such as utility functions. It's crucial to note that this function only works within an active server request.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/context.mdx#_snippet_3

LANGUAGE: Python
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

TITLE: Use pathlib.Path for File System Paths in FastMCP Tools
DESCRIPTION: Demonstrates using the `pathlib.Path` type for tool parameters. FastMCP automatically converts string paths provided by clients into `Path` objects, simplifying file system interactions.
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

TITLE: Configuring FastMCP Server Instance in Python
DESCRIPTION: Details how to configure server-specific settings by passing arguments during `FastMCP` instance creation. It covers optional dependencies, tag-based component exposure, and handling duplicate registrations.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_10

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

TITLE: Toggle FastMCP Prompt State Programmatically (Python)
DESCRIPTION: This example illustrates how to programmatically enable or disable a FastMCP prompt after it has been defined, using the `.disable()` and `.enable()` methods on the prompt object itself.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_7

LANGUAGE: python
CODE:

```
@mcp.prompt
def seasonal_prompt(): return "Happy Holidays!"

# Disable and re-enable the prompt
seasonal_prompt.disable()
seasonal_prompt.enable()
```

---

TITLE: FastMCP: Mounting a Subserver for Dynamic Composition (Partial)
DESCRIPTION: This partial Python example begins to demonstrate the `FastMCP.mount()` method, which creates a live link between a main server and a subserver. It initializes a `DynamicService` subserver and defines an `initial_tool`, setting up the foundation for dynamic delegation of requests.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_1

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
```

---

TITLE: Implement ToolError for Controlled Error Messages
DESCRIPTION: Shows how to use ToolError to raise specific, client-facing error messages that are always sent, regardless of the mask_error_details setting. It also illustrates a standard TypeError which would be masked.
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

TITLE: Configure BearerAuthProvider with a Static Public Key
DESCRIPTION: This example illustrates how to configure the `BearerAuthProvider` using a static RSA public key provided directly in PEM format. This method is suitable when the public key is known and does not change frequently, offering a straightforward way to verify JWT signatures.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_1

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

TITLE: Verify FastMCP Installation
DESCRIPTION: This command executes the FastMCP version check, displaying the installed FastMCP and MCP SDK versions. It also shows Python version and platform details to confirm successful installation.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/getting-started/installation.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp version
```

---

TITLE: Define FastMCP Prompt with Required and Optional Parameters (Python)
DESCRIPTION: This snippet demonstrates how to define a FastMCP prompt function in Python, illustrating the distinction between required parameters (no default value) and optional parameters (with default values). It shows how FastMCP infers parameter requirements from the function signature.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_3

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

TITLE: Install FastMCP Server in Claude Desktop
DESCRIPTION: The `install` command sets up a FastMCP server for the Claude desktop app. It supports specifying server objects using `file.py:object` notation and managing dependencies via `--with` or `--with-editable` options. Note that Claude Desktop runs servers in an isolated environment, requiring explicit dependency specification and `uv` to be installed globally. Currently, only STDIO transport is supported for installed servers.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_10

LANGUAGE: bash
CODE:

```
fastmcp install server.py
```

LANGUAGE: bash
CODE:

```
fastmcp install server.py:my_server
```

LANGUAGE: bash
CODE:

```
fastmcp install server.py:my_server -n "My Analysis Server" --with pandas
```

---

TITLE: FastMCP Client Session Management with keep_alive
DESCRIPTION: Demonstrates how the `keep_alive` parameter affects session persistence and subprocess management when using the FastMCP client. `keep_alive=True` (default) maintains the server subprocess across context manager exits, while `keep_alive=False` starts a new subprocess for each session, ensuring isolation.
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

TITLE: Disable FastMCP Resources on Creation or Programmatically (Python)
DESCRIPTION: This section demonstrates how to control the visibility and availability of FastMCP resources. Resources can be disabled during their initial creation using the `enabled` parameter in the `@mcp.resource` decorator, or their state can be toggled programmatically after creation using `disable()` and `enable()` methods on the resource function.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_4

LANGUAGE: python
CODE:

```
@mcp.resource("data://secret", enabled=False)
def get_secret_data():
    """This resource is currently disabled."""
    return "Secret data"
```

LANGUAGE: python
CODE:

```
@mcp.resource("data://config")
def get_config(): return {"version": 1}

# Disable and re-enable the resource
get_config.disable()
get_config.enable()
```

---

TITLE: Configuring Progress Monitoring with FastMCP Client (Python)
DESCRIPTION: This example illustrates how to set up a `progress_handler` when initializing the FastMCP client. This handler is an asynchronous function designed to receive and process progress updates from the server during long-running operations, providing current progress, total expected value, and an optional status message.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/advanced-features.mdx#_snippet_1

LANGUAGE: Python
CODE:

```
from fastmcp import Client
from fastmcp.client.progress import ProgressHandler

async def my_progress_handler(
    progress: float,
    total: float | None,
    message: str | None
) -> None:
    print(f"Progress: {progress} / {total} ({message})")

client = Client(
    ...,
    progress_handler=my_progress_handler
)
```

---

TITLE: Client Authentication Error Response (401 Unauthorized)
DESCRIPTION: This snippet displays a typical error response received when an unauthenticated client attempts to access a FastMCP server. The `401 (Unauthorized)` status code indicates that the request was rejected due to missing or invalid authentication credentials.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/openai.mdx#_snippet_9

LANGUAGE: APIDOC
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

TITLE: Configuring Resource Prefix Format in FastMCP (Python)
DESCRIPTION: Demonstrates how to configure the resource prefix format globally using `fastmcp.settings` or per-server during `FastMCP` instantiation. This setting determines whether prefixes are added to the URI path or protocol.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/composition.mdx#_snippet_5

LANGUAGE: python
CODE:

```
import fastmcp
fastmcp.settings.resource_prefix_format = "protocol"

from fastmcp import FastMCP

# Create a server that uses legacy protocol format
server = FastMCP("LegacyServer", resource_prefix_format="protocol")

# Create a server that uses new path format
server = FastMCP("NewServer", resource_prefix_format="path")
```

---

TITLE: Run a Local FastMCP Server
DESCRIPTION: Executes a FastMCP server defined in a local Python file directly using the current Python environment. Users are responsible for managing dependencies.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/cli.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
fastmcp run server.py
```

---

TITLE: Set Gemini API Key Environment Variable
DESCRIPTION: This command sets the `GEMINI_API_KEY` environment variable, which is used by the Google Generative AI SDK for authentication. Replace "your-api-key" with your actual Gemini API key.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/gemini.mdx#_snippet_2

LANGUAGE: bash
CODE:

```
export GEMINI_API_KEY="your-api-key"
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

TITLE: Managing Client Sessions with `keep_alive` in FastMCP (Python)
DESCRIPTION: This snippet illustrates the `keep_alive` feature in FastMCP client session management, which is enabled by default for stdio transports. It shows how the client maintains the same subprocess session across multiple `async with` contexts, improving efficiency by avoiding repeated session initialization.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_13

LANGUAGE: python
CODE:

```
from fastmcp import Client

client = Client("my_mcp_server.py")  # keep_alive=True by default

async def example():
    async with client:
        await client.ping()

    async with client:
        await client.ping()  # Same subprocess as above
```

---

TITLE: Configuring FastMCP Global Settings via Environment Variables
DESCRIPTION: Provides examples of setting FastMCP global configurations using environment variables. It shows how to set `FASTMCP_LOG_LEVEL`, `FASTMCP_MASK_ERROR_DETAILS`, and `FASTMCP_RESOURCE_PREFIX_FORMAT` in a bash shell.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_13

LANGUAGE: bash
CODE:

```
# Global settings
export FASTMCP_LOG_LEVEL=DEBUG
export FASTMCP_MASK_ERROR_DETAILS=True
export FASTMCP_RESOURCE_PREFIX_FORMAT=protocol
```

---

TITLE: FastMCP @mcp.resource Decorator Arguments
DESCRIPTION: This API documentation outlines the configurable parameters for the `@mcp.resource` decorator, allowing developers to precisely define and describe their resources within FastMCP.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
@mcp.resource Arguments:
- uri (str): The unique identifier for the resource (required).
- name (str, optional): A human-readable name (defaults to function name).
- description (str, optional): Explanation of the resource (defaults to docstring).
- mime_type (str, optional): Specifies the content type (FastMCP often infers a default like text/plain or application/json, but explicit is better for non-text types).
- tags (set[str], optional): A set of strings for categorization, potentially used by clients for filtering.
- enabled (bool, optional): A boolean to enable or disable the resource (defaults to True). See [Disabling Resources](#disabling-resources) for more information.
```

---

TITLE: FastMCP Default Route Mapping (Python)
DESCRIPTION: Shows FastMCP's default `RouteMap` configuration, where all OpenAPI routes are mapped to `TOOL` components by default for maximum compatibility with LLM clients.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/openapi.mdx#_snippet_1

LANGUAGE: python
CODE:

```
from fastmcp.server.openapi import RouteMap, MCPType

DEFAULT_ROUTE_MAPPINGS = [
    # All routes become tools
    RouteMap(mcp_type=MCPType.TOOL),
]
```

---

TITLE: Listing Available Tools with FastMCP Client
DESCRIPTION: Demonstrates how to use the `list_tools()` method of the FastMCP client to retrieve a list of all tools available on the connected server. The method returns a list of `mcp.types.Tool` objects, providing information about each tool.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_4

LANGUAGE: Python
CODE:

```
tools = await client.list_tools()
# tools -> list[mcp.types.Tool]
```

---

TITLE: Add FastMCP Resources with Custom Storage Keys
DESCRIPTION: This snippet demonstrates how to add resources to FastMCP using `mcp.add_resource()`. It shows both the default behavior, where the resource's URI is used as the storage key, and how to specify a custom `key` parameter for alternative access. This allows for flexible resource management beyond the default URI-based keying.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_9

LANGUAGE: Python
CODE:

```
# Creating a resource with standard URI as the key
resource = TextResource(uri="resource://data")
mcp.add_resource(resource)  # Will be stored and accessed using "resource://data"

# Creating a resource with a custom key
special_resource = TextResource(uri="resource://special-data")
mcp.add_resource(special_resource, key="internal://data-v2")  # Will be stored and accessed using "internal://data-v2"
```

---

TITLE: RSAKeyPair.create_token() Method Parameters
DESCRIPTION: API documentation for the `create_token()` method, detailing its parameters such as `subject`, `issuer`, `audience`, `scopes`, `expires_in_seconds`, `additional_claims`, and `kid`, along with their types, default values, and descriptions.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/auth/bearer.mdx#_snippet_5

LANGUAGE: APIDOC
CODE:

```
Method: create_token
  Parameters:
    subject (str, default: "fastmcp-user"): JWT subject claim (usually user ID)
    issuer (str, default: "https://fastmcp.example.com"): JWT issuer claim
    audience (str, default: None): JWT audience claim
    scopes (list[str], default: None): OAuth scopes to include
    expires_in_seconds (int, default: 3600): Token expiration time
    additional_claims (dict, default: None): Extra claims to include
    kid (str, default: None): Key ID for JWKS lookup
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

TITLE: Authenticate OpenAI Client with FastMCP Server using Bearer Token
DESCRIPTION: This snippet demonstrates how to authenticate an OpenAI client when interacting with a FastMCP server that requires bearer tokens. It shows how to include the access token in the `Authorization` header within the `tools` configuration of the OpenAI client's `responses.create` method.
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
            "server_url": f"{url}/sse",
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

TITLE: FastMCP Resource Return Value Handling
DESCRIPTION: This section details how FastMCP automatically processes different Python return types from resource functions, converting them into appropriate MCP resource content formats such as `TextResourceContents` for strings and JSON, or `BlobResourceContents` for bytes.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/resources.mdx#_snippet_1

LANGUAGE: APIDOC
CODE:

```
Return Value Type Mapping:
- str: Sent as TextResourceContents (with mime_type="text/plain" by default).
- dict, list, pydantic.BaseModel: Automatically serialized to a JSON string and sent as TextResourceContents (with mime_type="application/json" by default).
- bytes: Base64 encoded and sent as BlobResourceContents. You should specify an appropriate mime_type (e.g., "image/png", "application/octet-stream").
- None: Results in an empty resource content list being returned.
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

TITLE: FastMCP Prompt Decorator Metadata Parameters (APIDOC)
DESCRIPTION: Documentation for the parameters available in the `@mcp.prompt` decorator to customize prompt metadata and behavior.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_5

LANGUAGE: APIDOC
CODE:

```
@mcp.prompt decorator parameters:
  name: Sets the explicit prompt name exposed via MCP.
  description: Provides the description exposed via MCP. If set, the function's docstring is ignored for this purpose.
  tags: A set of strings used to categorize the prompt. Clients might use tags to filter or group available prompts.
  enabled: A boolean to enable or disable the prompt (defaults to True).
```

---

TITLE: Configure FastMCP Server Tag-Based Filtering
DESCRIPTION: Demonstrates how to configure tag-based filtering when instantiating the FastMCP server. Using `include_tags` and `exclude_tags`, you can control which components are exposed to clients, with exclude tags taking precedence.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/fastmcp.mdx#_snippet_6

LANGUAGE: python
CODE:

```
# Only expose components tagged with "public"
mcp = FastMCP(include_tags={"public"})

# Hide components tagged as "internal" or "deprecated"
mcp = FastMCP(exclude_tags={"internal", "deprecated"})

# Combine both: show admin tools but hide deprecated ones
mcp = FastMCP(include_tags={"admin"}, exclude_tags={"deprecated"})
```

---

TITLE: Toggle FastMCP Tool State Programmatically
DESCRIPTION: Shows how to dynamically enable or disable a FastMCP tool after its creation using the `.disable()` and `.enable()` methods. This allows for runtime control over tool availability.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#_snippet_9

LANGUAGE: python
CODE:

```
@mcp.tool
def dynamic_tool():
    return "I am a dynamic tool."

# Disable and re-enable the tool
dynamic_tool.disable()
dynamic_tool.enable()
```

---

TITLE: mcp.prompt Duplicate Registration Behavior Options
DESCRIPTION: Describes the configurable behaviors when a prompt with an existing name is registered using `mcp.prompt`. These options determine how the system handles the conflict.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_13

LANGUAGE: APIDOC
CODE:

```
Duplicate Behavior Options:
- "warn" (default): Logs a warning, and the new prompt replaces the old one.
- "error": Raises a ValueError, preventing the duplicate registration.
- "replace": Silently replaces the existing prompt with the new one.
- "ignore": Keeps the original prompt and ignores the new registration attempt.
```

---

TITLE: Pinging FastMCP Server for Connectivity (Python)
DESCRIPTION: This example shows how to use the FastMCP client to ping the server and verify connectivity. The `ping()` method, when called within an `async with` block, confirms that the server is reachable and responsive.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/client.mdx#_snippet_12

LANGUAGE: python
CODE:

```
async with client:
    await client.ping()
    print("Server is reachable")
```

---

TITLE: ArgTransform Class Parameters
DESCRIPTION: Defines the parameters available for the `ArgTransform` object, used to modify tool arguments for LLM interaction. Includes details on name, description, default, default_factory, hide, required, and type, along with important usage constraints.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/patterns/tool-transformation.mdx#_snippet_3

LANGUAGE: APIDOC
CODE:

```
ArgTransform Class Parameters:
  name: string - The new name for the argument.
  description: string - The new description for the argument.
  default: any - The new default value for the argument.
  default_factory: function - A function that will be called to generate a default value for the argument. This is useful for arguments that need to be generated for each tool call, such as timestamps or unique IDs.
  hide: boolean - Whether to hide the argument from the LLM.
  required: boolean - Whether the argument is required, usually used to make an optional argument be required instead.
  type: type - The new type for the argument.

Constraints:
  - default_factory can only be used with hide=True, because dynamic defaults cannot be represented in a JSON schema for the client.
  - required=True can only be set for arguments that do not declare a default value.
```

---

TITLE: Configuring Dynamic Roots Callback with FastMCP Client (Python)
DESCRIPTION: This example illustrates how to configure dynamic roots for the FastMCP client using an asynchronous callback function. The `roots_callback` function is invoked by the server when it needs to query the client's roots, allowing the client to dynamically determine and return a list of accessible resources based on the provided `RequestContext`.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/clients/advanced-features.mdx#_snippet_5

LANGUAGE: Python
CODE:

```
from fastmcp import Client
from fastmcp.client.roots import RequestContext

async def roots_callback(context: RequestContext) -> list[str]:
    print(f"Server requested roots (Request ID: {context.request_id})")
    return ["/path/to/root1", "/path/to/root2"]

client = Client(
    ...,
    roots=roots_callback,
)
```

---

TITLE: Authenticate Anthropic Client with FastMCP Server
DESCRIPTION: This example demonstrates how to authenticate an Anthropic client when making requests to an authenticated FastMCP server. It shows how to pass the `authorization_token` within the `mcp_servers` configuration, allowing the client to successfully interact with the server.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_10

LANGUAGE: Python
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
            "url": f"{url}/sse",
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

TITLE: Set Anthropic API Key Environment Variable
DESCRIPTION: This command sets the `ANTHROPIC_API_KEY` environment variable, which is used by the Anthropic Python SDK for authentication when making API calls.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/integrations/anthropic.mdx#_snippet_3

LANGUAGE: bash
CODE:

```
export ANTHROPIC_API_KEY="your-api-key"
```

---

TITLE: Using Streamable HTTP Transport (Inferred)
DESCRIPTION: Demonstrates how the `Client` automatically infers and uses `StreamableHttpTransport` when provided with a standard HTTP or HTTPS URL. This simplifies client setup for web-based FastMCP servers.
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

TITLE: Disable FastMCP Prompt at Creation (Python)
DESCRIPTION: This snippet demonstrates how to disable a FastMCP prompt directly upon its creation by setting the `enabled=False` parameter in the `@mcp.prompt` decorator. Disabled prompts are not listed and cannot be called.
SOURCE: https://github.com/jlowin/fastmcp/blob/main/docs/servers/prompts.mdx#_snippet_6

LANGUAGE: python
CODE:

```
@mcp.prompt(enabled=False)
def experimental_prompt():
    """This prompt is not ready for use."""
    return "This is an experimental prompt."
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
