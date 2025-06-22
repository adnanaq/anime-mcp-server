TITLE: Starting Marqo Server with Docker (Bash)
DESCRIPTION: Provides the Bash commands required to pull the latest Marqo Docker image, remove any existing container named 'marqo', and run a new container, mapping port 8882 and configuring host networking. This is the first step to using Marqo.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_11

LANGUAGE: bash
CODE:

```
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Starting Marqo Docker Container
DESCRIPTION: Provides the necessary docker commands to pull the latest Marqo image, remove any existing container named 'marqo', and run a new container, mapping port 8882 and allowing host gateway access.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT3NewsSummary/README.md#_snippet_0

LANGUAGE: Shell
CODE:

```
docker pull marqoai/marqo:latest;
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Setting up Marqo Docker Container (Bash)
DESCRIPTION: Pulls the Marqo Docker image, removes any existing container named 'marqo', and runs a new Marqo container, mapping port 8882 and adding the host gateway for communication.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/readme.md#_snippet_1

LANGUAGE: Bash
CODE:

```
docker pull marqoai/marqo:2.0.0;
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:2.0.0
```

---

TITLE: Install Marqo Client
DESCRIPTION: Commands to create a new Conda environment named 'marqo-client' with Python 3.8, activate the environment, and install the Marqo Python client and matplotlib using pip.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_1

LANGUAGE: shell
CODE:

```
conda create -n marqo-client python=3.8
conda activate marqo-client

pip install marqo matplotlib
```

---

TITLE: Initializing Marqo Client and Creating Index (Python)
DESCRIPTION: Imports the Marqo client, initializes it, defines an index name, and creates a new Marqo index. If no specific settings are provided, the index uses the default encoder.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_3

LANGUAGE: Python
CODE:

```
from marqo import Client
mq = Client()
index_name = "iron-docs"
mq.create_index(index_name)
```

---

TITLE: Initializing Marqo Client and Creating Index (Python)
DESCRIPTION: Imports the Marqo Client, initializes a client instance, defines an index name, and creates a new index in Marqo. This sets up the environment for adding and searching documents. Requires the Marqo library.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_14

LANGUAGE: python
CODE:

```
from marqo import Client
mq = Client()
index_name = "iron-docs"
mq.create_index(index_name)
```

---

TITLE: Running Marqo Container with Docker - Bash
DESCRIPTION: This command sequence removes any existing Marqo container, pulls the latest Marqo Docker image, and runs a new container, mapping port 8882 for access. This is the first step to getting Marqo running locally and requires Docker to be installed and configured with sufficient resources.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_0

LANGUAGE: bash
CODE:

```
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
```

---

TITLE: Initialize Marqo Client
DESCRIPTION: Import the Marqo library and create a client instance connected to the Marqo server running on localhost at port 8882.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_2

LANGUAGE: python
CODE:

```
import marqo
mq = marqo.Client("http://localhost:8882")
```

---

TITLE: Adding Documents to Marqo Index (Python)
DESCRIPTION: Adds a list of documents (prepared in the previous step) to the specified Marqo index. It sets a client batch size and specifies that the 's3_http' field should be used for tensor embedding. It also allows selecting the device ('cpu' or 'cuda') for processing. Requires the Marqo Python client.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_15

LANGUAGE: python
CODE:

```
device = 'cpu' # use 'cuda' if a GPU is available

res = client.index(index_name).add_documents(documents, client_batch_size=64, tensor_fields=["s3_http"], device=device)
```

---

TITLE: Indexing Documents with Marqo Python Client
DESCRIPTION: Demonstrates how to connect to a running Marqo instance, create a new index named 'news-index', and add a list of documents (`MARQO_DOCUMENTS`) to the index, specifying 'Title' and 'Description' as tensor fields for neural embedding.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT3NewsSummary/README.md#_snippet_1

LANGUAGE: Python
CODE:

```
from news import MARQO_DOCUMENTS

DOC_INDEX_NAME = 'news-index'

print('Establishing connection to marqo client.')
mq = marqo.Client(url='http://localhost:8882')

print('creating a Marqo index')
mq.create_index(DOC_INDEX_NAME)

print('Indexing documents')
mq.index(DOC_INDEX_NAME).add_documents(MARQO_DOCUMENTS, tensor_fields= ["Title", "Description"])
```

---

TITLE: Installing Python Dependencies (Bash)
DESCRIPTION: Installs the Marqo Python client library and other project dependencies listed in the 'requirements.txt' file using pip.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/readme.md#_snippet_2

LANGUAGE: Bash
CODE:

```
pip install marqo
pip install -r requirements.txt
```

---

TITLE: Formatting Documents for Marqo Indexing (Python)
DESCRIPTION: Defines a Python dictionary representing a document with text and source fields, and creates a list of such documents for ingestion into a Marqo index. Documents must be in this dictionary format for indexing.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_2

LANGUAGE: Python
CODE:

```
document1 = {"text":"Auto-Off function: This feature automatically switches
                 off the steam iron if it has not been moved for a while.",
             "source":"page 1"}
# other document content left out for clarity
documents = [document1, document2, document3, document4, document5]
```

---

TITLE: Indexing and Searching Documents with Marqo Python Client
DESCRIPTION: Demonstrates how to initialize the Marqo client, create a new index, add a list of documents to the index (specifying fields for vector indexing and an optional document ID), and perform a vector search query against the index.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_5

LANGUAGE: python
CODE:

```
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index")

mq.index("my-first-index").add_documents([
    {
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing Polo's travels"
    },
    {
        "Title": "Extravehicular Mobility Unit (EMU)",
        "Description": "The EMU is a spacesuit that provides environmental protection, "
                       "mobility, life support, and communications for astronauts",
        "_id": "article_591"
    }],
    tensor_fields=["Description"]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)
```

---

TITLE: Basic Vector Similarity Query (Python)
DESCRIPTION: Presents a standard query based solely on vector similarity, serving as a baseline before introducing score modifications based on other signals.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_8

LANGUAGE: Python
CODE:

```
query = {"yellow handbag":1.0}
```

---

TITLE: Generate Conversational Answer using Marqo and LLM - Python
DESCRIPTION: Searches a Marqo index for relevant documents based on a query, extracts transcriptions from the results to form a context, and then uses an OpenAI language model with a predefined prompt template to generate a natural language answer based on the context and the original query.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_10

LANGUAGE: python
CODE:

```
def answer_question(
    query: str,
    limit: int,
    index: str,
    mq: marqo.Client,
) -> str:
    print("Searching...")
    results = mq.index(index).search(
        q=query,
        limit=limit,
    )
    print("Done!")

    context = ". ".join([r["transcription"] for r in results["hits"]])

    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])
    llm = OpenAI(temperature=0.9)
    chain_qa = LLMChain(llm=llm, prompt=prompt)
    llm_results = chain_qa(
        {"context": context, "question": query}, return_only_outputs=True
    )
    return llm_results["text"]
```

---

TITLE: Searching Index with Text Query - Python
DESCRIPTION: Performs a search against a Marqo index using a text query ("brocolli") and specifies the device for the search operation. Prints the first hit from the search response, which includes details like score, ID, and highlights. Requires a Marqo client and an index name.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_4

LANGUAGE: Python
CODE:

```
response = client.index(index_name).search("brocolli", device="cuda")
print(response['hits'][0])
```

---

TITLE: Performing Basic Text Search in Marqo (Python)
DESCRIPTION: Executes a search query ("green shirt") against the specified Marqo index. It uses the previously defined device and limits the number of results returned to 10. Requires the Marqo Python client and an indexed index.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_16

LANGUAGE: python
CODE:

```
query = "green shirt"
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Defining Mappings and Indexing Multimodal Text/Image Documents (Python)
DESCRIPTION: Specifies a list of fields (non_tensor_fields) that should not be tensorized. It defines a multimodal mapping (mappings) for the "multimodal" tensor field, combining the blip_large_caption (text) and s3_http (image URL) fields with specified weights. Finally, it calls add_documents to index the documents into the previously created multimodal index, applying the defined mappings.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_29

LANGUAGE: python
CODE:

```
# the fields we do not want to embed
non_tensor_fields = ['_id', 'price', 'blip_large_caption', 'aesthetic_score', 's3_http']

# define how we want to combine the fields
mappings = {"multimodal":
                         {"type": "multimodal_combination",
                          "weights":
                             {"blip_large_caption": 0.20,
                               "s3_http": 0.80,
                             }
                         }
                }

# now index
res = client.index(index_name_mm_objects).add_documents(documents, client_batch_size=64, tensor_fields=["multimodal"], device=device, mappings=mappings)
```

---

TITLE: Searching Marqo Index with Python Client
DESCRIPTION: Shows how to perform a search query against the 'news-index' using the Marqo Python client. It queries with a natural language question (`question`), applies a filter to restrict results by date (`date`), and limits the number of results returned.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT3NewsSummary/README.md#_snippet_2

LANGUAGE: Python
CODE:

```
question = 'What is happening in business today?'
date = '2022-11-09'
results = mq.index(DOC_INDEX_NAME).search(
					q=question,
					filter_string=f"date:{date}",
                    limit=5)
```

---

TITLE: Basic Marqo Indexing and Searching - Python
DESCRIPTION: Demonstrates connecting to a Marqo instance, creating an index with a specified model ('hf/e5-base-v2'), adding documents with both metadata and tensor fields ('Description'), and performing a semantic search query. This requires a running Marqo instance (e.g., via Docker) and the Marqo Python client installed.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_2

LANGUAGE: python
CODE:

```
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index", model="hf/e5-base-v2")

mq.index("my-first-index").add_documents([
    {
        "Title": "The Travels of Marco Polo",
        "Description": "A 13th-century travelogue describing Polo's travels"
    },
    {
        "Title": "Extravehicular Mobility Unit (EMU)",
        "Description": "The EMU is a spacesuit that provides environmental protection, "
                       "mobility, life support, and communications for astronauts",
        "_id": "article_591"
    }],
    tensor_fields=["Description"]
)

results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)
```

---

TITLE: Searching Marqo Index with Text Query
DESCRIPTION: Performs a search operation on the Marqo index named "hot-dogs-100k" using the text query "a face". Returns search results containing relevant documents and scores. Requires a Marqo client instance.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_7

LANGUAGE: python
CODE:

```
results = client.index("hot-dogs-100k").search("a face")
```

---

TITLE: Creating a Multimodal Query with Negation in Python
DESCRIPTION: Shows how to include negative terms in a multimodal query using a Python dictionary. A negative weight (e.g., -1.0 for "buttons") instructs the search to move away from that concept while still being drawn to positively weighted terms.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_1

LANGUAGE: python
CODE:

```
query = {"green shirt":1.0, "short sleeves":1.0, "buttons":-1.0}
```

---

TITLE: Install Marqo using Docker and Pip - Bash
DESCRIPTION: These commands demonstrate how to install the Marqo server using Docker and the Marqo Python client using pip. This is a prerequisite step before indexing documents or performing searches.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_1

LANGUAGE: Bash
CODE:

```
docker pull marqoai/marqo:2.0.0;
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:2.0.0
pip install marqo
```

---

TITLE: Using Negation to Exclude Concepts in a Multimodal Query in Python
DESCRIPTION: Illustrates using a negative weight in a multimodal query to explicitly avoid certain concepts, such as "lowres, blurry, low quality" images. This helps refine search results by excluding undesirable items based on natural language descriptions.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_2

LANGUAGE: python
CODE:

```
query = {"yellow handbag":1.0, "lowres, blurry, low quality":-1.1}
```

---

TITLE: Installing and Running Marqo
DESCRIPTION: Provides the necessary commands to install the Marqo Python library and pull and run the Marqo Docker container, exposing it on port 8882.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_0

LANGUAGE: shell
CODE:

```
pip install marqo
docker pull marqoai/marqo:latest;
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Creating and Searching Marqo Multimodal Index (Python)
DESCRIPTION: Demonstrates how to create a Marqo index with multimodal combination fields, add documents containing text and image URLs with specified weights, and perform various search queries (simple text, weighted components, negative weighting) against the index. Requires the Marqo client and a running Marqo instance.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_15

LANGUAGE: python
CODE:

```
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}

mq.create_index("my-first-multimodal-index", **settings)

mq.index("my-first-multimodal-index").add_documents(
    [
        {
            "Title": "Flying Plane",
            "caption": "An image of a passenger plane flying in front of the moon.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
        {
            "Title": "Red Bus",
            "caption": "A red double decker London bus traveling to Aldwych",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        {
            "Title": "Horse Jumping",
            "caption": "A person riding a horse over a jump in a competition.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        },
    ],
    # Create the mappings, here we define our captioned_image mapping
    # which weights the image more heavily than the caption - these pairs
    # will be represented by a single vector in the index
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    },
    # We specify which fields to create vectors for.
    # Note that captioned_image is treated as a single field.
    tensor_fields=["captioned_image"]
)

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)

print("Query 1:")
pprint.pprint(results)

# search the index with a query that uses weighted components
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0
    },
)
print("\nQuery 2:")
pprint.pprint(results)

results = mq.index("my-first-multimodal-index").search(
    q={"Animals of the Perissodactyla order": -1.0}
)
print("\nQuery 3:")
pprint.pprint(results)
```

---

TITLE: Searching Marqo Index with Images (Python)
DESCRIPTION: Illustrates how to perform a search using an image URL as the query input. The image is treated as the sole basis for the search query.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_20

LANGUAGE: python
CODE:

```
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/red_backpack.jpg"
query = {image_context_url:1.0}
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Creating an Image-Based Multimodal Query in Python
DESCRIPTION: Shows how to construct a multimodal query using an image URL as the primary component. The image URL acts as a query term with a positive weight (e.g., 1.0) to find items similar to the image.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_3

LANGUAGE: python
CODE:

```
query = {image_url:1.0}
```

---

TITLE: Install Marqo Python Library (Bash)
DESCRIPTION: Installs the Marqo Python library using pip. This is a necessary prerequisite for running the `simple_marqo_demo.py` script and interacting with the Marqo service from Python.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ClothingCLI/README.md#_snippet_2

LANGUAGE: bash
CODE:

```
pip install marqo
```

---

TITLE: Installing Marqo Library (Shell)
DESCRIPTION: Installs the Marqo Python library using the pip package manager. This makes the Marqo functionality available for the Streamlit application. Users are advised to ensure installation occurs within the correct environment, especially if using Anaconda.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ClothingStreamlit/README.md#_snippet_2

LANGUAGE: Shell
CODE:

```
pip install marqo
```

---

TITLE: Querying with Prompting for Specific Styles (Python)
DESCRIPTION: Shows how to use "prompting" by appending descriptive terms to a query string to curate search results towards specific characteristics or styles, similar to text-to-image generation models.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_6

LANGUAGE: Python
CODE:

```
query = {"handbag, bold colors, vibrant":1.0}
```

---

TITLE: Installing Marqo Python Client - Bash
DESCRIPTION: Installs the official Marqo Python client library using pip. This client is necessary to interact with a running Marqo instance from Python applications, allowing you to perform operations like creating indexes, adding documents, and searching.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_1

LANGUAGE: bash
CODE:

```
pip install marqo
```

---

TITLE: Installing Marqo Python Client (Bash)
DESCRIPTION: Shows the standard pip command to install the Marqo Python client library, which is necessary to interact with the Marqo server programmatically from Python.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_12

LANGUAGE: bash
CODE:

```
pip install marqo
```

---

TITLE: Add Formatted Documents to Marqo Index - Python
DESCRIPTION: This snippet shows how to add the prepared list of documents (containing image data) to a Marqo index using the `add_documents` function. It specifies the index name, the list of documents, the field containing the image data (`tensor_fields`), the processing device (`device`), and the batch size (`client_batch_size`). The output displays logging information about the indexing process.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_7

LANGUAGE: python
CODE:

```
mq.index(index_name).add_documents(documents, tensor_fields=["image_docker"], device="cpu", client_batch_size= 1)
```

---

TITLE: Performing Weighted Queries in Marqo (Python)
DESCRIPTION: This comprehensive example demonstrates how to use weighted queries in Marqo. It includes setting up a client, creating an index, adding documents, and performing searches using dictionaries where keys are query components and values are weights, allowing for nuanced and negated searches.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_14

LANGUAGE: python
CODE:

```
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

mq.create_index("my-weighted-query-index")

mq.index("my-weighted-query-index").add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
        {
            "Title": "Thylacine",
            "Description": "The thylacine, also commonly known as the Tasmanian tiger or Tasmanian wolf, "
            "is an extinct carnivorous marsupial."
            "The last known of its species died in 1936.",
        }
    ],
    tensor_fields=["Description"]
)

# initially we ask for a type of communications device which is popular in the 21st century
query = {
    # a weighting of 1.1 gives this query slightly more importance
    "I need to buy a communications device, what should I get?": 1.1,
    # a weighting of 1 gives this query a neutral importance
    # this will lead to 'Smartphone' being the top result
    "The device should work like an intelligent computer.": 1.0,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("Query 1:")
pprint.pprint(results)

# now we ask for a type of communications which predates the 21st century
query = {
    # a weighting of 1 gives this query a neutral importance
    "I need to buy a communications device, what should I get?": 1.0,
    # a weighting of -1 gives this query a negation effect
    # this will lead to 'Telephone' being the top result
    "The device should work like an intelligent computer.": -0.3,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("\nQuery 2:")
pprint.pprint(results)
```

---

TITLE: Indexing Transcriptions in Marqo with Python
DESCRIPTION: This function takes a list of annotated transcriptions and indexes them into a specified Marqo index using a provided Marqo client. It includes a filtering step to remove short or potentially erroneous transcriptions before adding them to the index. It allows specifying tensor fields, device, and batch size for the indexing operation.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_8

LANGUAGE: python
CODE:

```
def index_transcriptions(
    annotated_transcriptions: List[Dict[str, Any]],
    index: str,
    mq: marqo.Client,
    tensor_fields: List[str] = [],
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, str]:

    # drop short transcriptions and transcriptions that consist of duplicated repeating
    # character artifacts
    annotated_transcriptions = [
        at
        for at in annotated_transcriptions
        if len(at["transcription"]) > 5 or len({*at["transcription"]}) > 4
    ]

    response = mq.index(index).add_documents(
        annotated_transcriptions,
        tensor_fields=tensor_fields,
        device=device,
        client_batch_size=batch_size
    )

    return response
```

---

TITLE: Starting Marqo Docker Container - Shell
DESCRIPTION: Command to start the Marqo Docker container, enabling GPU support and exposing the API port. Requires Docker and optionally a CUDA-compatible GPU.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_0

LANGUAGE: Shell
CODE:

```
docker run --name marqo -it -p 8882:8882 --gpus all --add-host host.docker.internal:host-gateway -e MARQO_MODELS_TO_PRELOAD='[]' marqoai/marqo:2.0.0
```

---

TITLE: Creating Multilingual Marqo Index (Python)
DESCRIPTION: Calls the `create_index` method on the Marqo client `mq`. Creates an index named 'my-multilingual-index' and specifies the 'stsb-xlm-r-multilingual' model for tensor generation, enabling multilingual search capabilities.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_3

LANGUAGE: python
CODE:

```
mq.create_index(index_name='my-multilingual-index', model='stsb-xlm-r-multilingual')
```

---

TITLE: Setting up Marqo Docker Container (Shell)
DESCRIPTION: Removes any existing Marqo container and runs a new one named 'marqo'. It maps port 8882, enables GPU access (`--gpus all`), and sets up host gateway access. Remove `--gpus all` if no GPU is available.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_0

LANGUAGE: sh
CODE:

```
docker rm -f marqo;
docker run --name marqo -it -p 8882:8882 --gpus all --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Searching a Marqo Image Field Using Text (Python)
DESCRIPTION: This snippet demonstrates performing a text-based search against a Marqo index that contains image data. The search query 'animal' is used to find relevant documents based on the indexed image and text content.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_12

LANGUAGE: python
CODE:

```
results = mq.index("my-multimodal-index").search('animal')
```

---

TITLE: Initializing Marqo Client (Python)
DESCRIPTION: Imports the `Client` class from the `marqo` library. Creates a Marqo client instance `mq` connected to the Marqo service running on `http://localhost:8882`, which is the default address for the Docker container.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_2

LANGUAGE: python
CODE:

```
from marqo import Client
mq = Client("http://localhost:8882")
```

---

TITLE: Running Marqo Docker Container
DESCRIPTION: This command removes any existing Marqo container and then starts a new one using the latest Marqo image, mapping port 8882 and allowing host access.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SimpleWiki/README.md#_snippet_0

LANGUAGE: Shell
CODE:

```
docker rm -f marqo;docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Run Marqo in Docker
DESCRIPTION: Commands to stop any existing Marqo container, pull the specified Marqo Docker image, and run a new container mapping port 8882 and adding a host gateway.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_0

LANGUAGE: shell
CODE:

```
docker rm -f marqo
docker pull marqoai/marqo:2.0.0
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:2.0.0
```

---

TITLE: Create Marqo Index
DESCRIPTION: Define index settings including the model ('open_clip/ViT-B-32/laion2b_s34b_b79k') and enabling image treatment for URLs/pointers, then create a new Marqo index named 'image-search-guide' with these settings.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_3

LANGUAGE: python
CODE:

```
index_name = 'image-search-guide'

settings = {
        "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
        "treatUrlsAndPointersAsImages": True,
        }

mq.create_index(index_name, settings_dict=settings)
```

---

TITLE: Searching Marqo Index with Multimodal Queries (Python)
DESCRIPTION: Demonstrates how to combine both text and image inputs within a single multi-part search query. The example shows two different multimodal queries.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_21

LANGUAGE: python
CODE:

```
# skateboard
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/71iPk9lfhML._SL1500_.jpg"

query = {"backpack":1.0, image_context_url:1.0}
res = client.index(index_name).search(query, device=device, limit=10)

# trees/hiking
image_context_url = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/trees.jpg"

query = {"backpack":1.0, image_context_url:1.0}
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Retrieving Embeddings and Searching with Context (Python)
DESCRIPTION: Retrieves indexed documents by their IDs using get_documents with expose_facets=True to access tensor embeddings. It extracts the embeddings, formats them into Marqo context objects, and then performs two search queries using the search method, applying the created context objects to influence the search results.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_27

LANGUAGE: python
CODE:

```
# retrieve the embedding to use as a context for search
indexed_documents = client.index(index_name_context).get_documents([document1['_id'], document2['_id']] , expose_facets=True)

# get the embedding
context_vector1 = indexed_documents['results'][0]['_tensor_facets'][0]['_embedding']
context_vector2 = indexed_documents['results'][1]['_tensor_facets'][0]['_embedding']

# create the context for the search
context1 = {"tensor":
                [
                  {'vector':context_vector1, 'weight':0.50}
                ]
            }

# create the context for the search
context2 = {"tensor":
                [
                  {'vector':context_vector2, 'weight':0.50}
                ]
            }

# now search
query = {"backpack":1.0}
res1 = client.index(index_name).search(query, device=device, limit=10, context=context1)

res2 = client.index(index_name).search(query, device=device, limit=10, context=context2)
```

---

TITLE: Adding Documents with Images to a Marqo Index (Python)
DESCRIPTION: This code shows how to add documents containing image URLs (either from the internet or disk) to a multi-modal Marqo index. The 'tensor_fields' parameter is used to specify which fields should be processed as tensors (e.g., images).
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_11

LANGUAGE: python
CODE:

```
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

---

TITLE: Start Marqo Docker Container (Shell)
DESCRIPTION: Runs a Marqo Docker container named 'marqo', maps port 8882, and preloads the 'hf/e5-base-v2' model, making Marqo available for performance tests.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/perf_tests/README.md#_snippet_1

LANGUAGE: shell
CODE:

```
docker run --name marqo -d -p 8882:8882 -e MARQO_MODELS_TO_PRELOAD='["hf/e5-base-v2"]' marqoai/marqo
```

---

TITLE: Creating a Marqo Index for Multi-modal Search (Python)
DESCRIPTION: This snippet demonstrates how to create a Marqo index configured for multi-modal search, specifically enabling the treatment of URLs and pointers as images and specifying a CLIP model (ViT-L/14) for processing.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_10

LANGUAGE: python
CODE:

```
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

---

TITLE: Searching Marqo Index with Image Pointer - Python
DESCRIPTION: Performs a semantic search on the specified Marqo index using the image pointer from the first hit of a previous search (presumably a black image) as the query, limiting the results to 100.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_5

LANGUAGE: python
CODE:

```
query = 'a black image'

results = client.index(index_name).search(query)

# remove the blank images
results = client.index(index_name).search(results['hits'][0]['image_docker'], limit=100)
```

---

TITLE: Adding Documents to Marqo Index (Python)
DESCRIPTION: Adds the previously defined list of documents ('documents') to the specified Marqo index ('index_name'). The 'tensor_fields' parameter specifies which fields ('name' and 'text') should be vectorized and indexed for search.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_15

LANGUAGE: python
CODE:

```
results = mq.index(index_name).add_documents(documents, tensor_fields = ["name", "text"])
```

---

TITLE: Setting Up Marqo Client and Index Settings - Python
DESCRIPTION: Initializes the Marqo client and defines base settings for image indexing, including treating URLs as images, specifying the model, and setting normalization. Includes a list of patch methods to iterate through. Requires the marqo library.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_2

LANGUAGE: Python
CODE:

```
from marqo import Client
client = Client()

# setup the settings so we can comapre the different methods
patch_methods = [None, "dino-v2", "yolox"]

settings = {
    "treatUrlsAndPointersAsImages": True,
    "imagePreprocessing": {
        "patchMethod": None
    },
    "model": "ViT-B/32",
    "normalizeEmbeddings": True,
}
```

---

TITLE: Creating Marqo Index with OpenCLIP Model (Python)
DESCRIPTION: Initializes the Marqo client and defines settings for a new index named 'multimodal'. The settings specify that URLs and pointers should be treated as images, use the 'open_clip/ViT-L-14' model, and normalize embeddings. It then calls the client method to create the index. Requires the Marqo Python client.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_14

LANGUAGE: python
CODE:

```
from marqo import Client

client = Client()

index_name = 'multimodal'
settings = {
	"treatUrlsAndPointersAsImages": True,
	"model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
	"normalizeEmbeddings": True,
}

response = client.create_index(index_name, settings_dict=settings)
```

---

TITLE: Running Marqo Docker Container (Shell)
DESCRIPTION: Launches the Marqo Docker container with the name 'marqo', mapping port 8882 from the container to the host. It also configures the host gateway to allow the container to access the host machine's network, which is necessary for connecting to the local HTTP server.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ClothingStreamlit/README.md#_snippet_1

LANGUAGE: Shell
CODE:

```
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Format Image Data for Marqo Indexing - Python
DESCRIPTION: This snippet demonstrates how to convert a list of image sources (`image_docker`) into the format required by Marqo's `add_documents` function: a list of dictionaries, where each dictionary represents a document with an image field (`image_docker`) and a unique identifier (`_id`). The output shows an example of the resulting list structure.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_6

LANGUAGE: python
CODE:

```
documents = [{"image_docker" : image, "_id" : str(idx)} for idx, image in enumerate(image_docker)]

print(documents)
```

---

TITLE: Loading E-commerce Data with Pandas (Python)
DESCRIPTION: Uses the pandas library to read a specified number of rows from a remote CSV file containing e-commerce product data. It selects relevant columns, adds an '\_id' based on the 's3_http' column, and converts the DataFrame into a list of dictionaries suitable for Marqo indexing. Requires pandas.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_13

LANGUAGE: python
CODE:

```
import pandas as pd

N = 100 # samples to use, the full dataset is ~220k
filename = "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce_meta_data.csv"
data = pd.read_csv(filename, nrows=N)
data['_id'] = data['s3_http']
documents = data[['s3_http', '_id', 'price', 'blip_large_caption', 'aesthetic_score']].to_dict(orient='records')
```

---

TITLE: Searching Marqo Index and Printing Highlights (Python)
DESCRIPTION: This Python function performs a search on a Marqo index named 'my-multilingual-index' using the provided query string. It then iterates through the search results and prints the '\_highlights' attribute for each hit using the 'pprint' module for formatted output. It requires the 'mq' object (presumably a Marqo client instance) and the 'pprint' library.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_5

LANGUAGE: Python
CODE:

```
import pprint

def search(query: str):
    result = mq.index(index_name='my-multilingual-index').search(q=query)
    for res in result["hits"]:
        pprint.pprint(res["_highlights"])
```

---

TITLE: Deleting Marqo Documents by ID (Python)
DESCRIPTION: Shows how to remove specific documents from a Marqo index using their unique identifiers. Requires the Marqo client and the index name.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_16

LANGUAGE: python
CODE:

```
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

---

TITLE: Searching a Multimodal Text/Image Index (Python)
DESCRIPTION: Defines a simple text query string. It then calls the search method on the Marqo index configured for multimodal objects (index_name_mm_objects), passing the query, device, and limit parameters to retrieve relevant documents based on the combined text and image representations.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_30

LANGUAGE: python
CODE:

```
query = "red shawl"
res = client.index(index_name_mm_objects).search(query, device=device, limit=10)
```

---

TITLE: Searching Marqo Index with Text Query - Python
DESCRIPTION: Performs a semantic search on the 'hot-dogs-100k' Marqo index using the text query 'a black image' and stores the search results.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_4

LANGUAGE: python
CODE:

```
results = client.index("hot-dogs-100k").search("a black image")
```

---

TITLE: Querying with Score Modification by Field (Python)
DESCRIPTION: Illustrates how to modify the search score by adding a bias based on a document-specific field, such as an "aesthetic_score", allowing other signals to influence the ranking alongside vector similarity.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_9

LANGUAGE: Python
CODE:

```
query = {"yellow handbag":1.0}
score_modifiers = {
        "add_to_score":
            [
              {"field_name": "aesthetic_score", "weight": 0.02}]
       }
```

---

TITLE: Searching Marqo Index with Semantic Filters (Python)
DESCRIPTION: Demonstrates how to perform a search query on a Marqo index using a multi-part query object. This allows for semantic filtering by assigning weights to different terms (e.g., "green shirt", "short sleeves").
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_18

LANGUAGE: python
CODE:

```
query = {"green shirt":1.0, "short sleeves":1.0}
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Extract Bounding Box from Search Results (Python)
DESCRIPTION: Following a search performed with a reranker that provides localization, this Python code snippet shows how to access the bounding box coordinates. It retrieves the 'image_location' from the first highlight within the first hit's '\_highlights' field and prints the coordinates.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_7

LANGUAGE: python
CODE:

```
bbox = response['hits']['_highlights'][0]['image_location']
print(bbox)
```

---

TITLE: Searching Marqo Index (Python)
DESCRIPTION: Performs a search query ("sara lee") against the specified Marqo index ('index_name'). This retrieves documents from the index that are most relevant to the query based on vector similarity.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_16

LANGUAGE: python
CODE:

```
results = mq.index(index_name).search("sara lee")
```

---

TITLE: Performing Lexical Search with Marqo
DESCRIPTION: Illustrates how to perform a keyword-based (lexical) search using the `search` method by providing the query string and setting the `search_method` parameter to `marqo.SearchMethods.LEXICAL`.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_9

LANGUAGE: python
CODE:

```
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

---

TITLE: Searching Marqo Index with Score Modifiers (Python)
DESCRIPTION: Explains how to influence search result ranking by incorporating document-specific fields (like aesthetic_score) using score_modifiers. It also shows how to compare results with and without ranking modifications.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_22

LANGUAGE: python
CODE:

```
query = {"yellow handbag":1.0}

# we define the extra document specific data to use for ranking
# multiple fields can be used to multiply or add to the vector similarity score
score_modifiers = {
        "add_to_score":
            [
            {"field_name": "aesthetic_score", "weight": 0.02}]
        }

res = client.index(index_name).search(query, device=device, limit=10, score_modifiers=score_modifiers)

# now get the aggregate aesthetic score
print(sum(r['aesthetic_score'] for r in res['hits']))

# and compare to the non ranking version
res = client.index(index_name).search(query, device=device, limit=10)

print(sum(r['aesthetic_score'] for r in res['hits']))
```

---

TITLE: Preparing Context from Marqo Results (Python)
DESCRIPTION: Processes search results obtained from Marqo. It extracts text highlights, truncates them to a specified token limit, and formats them into a list of Langchain `Document` objects, adding a source reference. This prepares the context for inclusion in the LLM prompt.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_20

LANGUAGE: python
CODE:

```
highlights, texts = extract_text_from_highlights(results, token_limit=150)
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```

---

TITLE: Extracting Bounding Box from Search Highlight - Python
DESCRIPTION: Accesses the search response to extract the bounding box coordinates from the first highlight of the first hit's 'image_location' field. Prints the extracted bounding box data. Requires a Marqo search response object.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_5

LANGUAGE: Python
CODE:

```
bbox = response['hits']['_highlights'][0]['image_location']
print(bbox)
```

---

TITLE: Defining GPT Prompt Template String (Python)
DESCRIPTION: Defines a multi-line Python string containing a template for a GPT prompt. It instructs the model to answer a question based _only_ on provided source text, including placeholders for the question and sources.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_7

LANGUAGE: Python
CODE:

```
template = """
Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION"), create a final answer one paragraph long.
Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know.
QUESTION: {question}
=========
SOURCES:
{summaries}
=========
ANSWER:
"""
```

---

TITLE: Searching Marqo Index with Negation (Python)
DESCRIPTION: Shows how to include negative terms in a multi-part search query to exclude results related to specific concepts (e.g., removing "buttons" from shirt results) by assigning a negative weight.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_19

LANGUAGE: python
CODE:

```
query = {"green shirt":1.0, "short sleeves":1.0, "buttons":-1.0}
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Performing Prompt-Style Search in Marqo (Python)
DESCRIPTION: Executes a more detailed, prompt-like search query ("cozy sweater, xmas, festive, holidays") against the specified Marqo index. It uses the previously defined device and limits the number of results returned to 10, showcasing how descriptive queries can be used.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_17

LANGUAGE: python
CODE:

```
query = "cozy sweater, xmas, festive, holidays"
res = client.index(index_name).search(query, device=device, limit=10)
```

---

TITLE: Zero-Shot Classification with Marqo Labels
DESCRIPTION: Creates a new Marqo index configured for image search, adds text labels as documents to this index, and then iterates through a dataset of image documents, searching the label index with each image to obtain classification scores for each label. Requires a Marqo client, a list of image documents, and a list of label documents.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_8

LANGUAGE: python
CODE:

```
index_name = 'one_dog_two'

# the documents here are actually serving as labels. the documents (i.e. label)
# are returned in the order they most closely match and the score can be used for classification
labels = [{"label":"one hot dog"}, {"label":"two hot dogs"},
                {"label":"a hamburger"}, {"label": "a face"}]

# get a copy of the labels only
label_strings = [list(a.values())[0] for a in labels]

# we create a new index
settings = {
        "model":'open_clip/ViT-B-32/laion2b_s34b_b79k',
        "treatUrlsAndPointersAsImages": True,
        }
client.create_index(index_name, settings_dict=settings)

# add our labels to the index
responses = client.index(index_name).add_documents(labels, tensor_fields=["label"])

# loop through the documents and search against the labels to get scores
for doc in documents:

    # the url for the image is what is used as the search - an image
    # note: you will want a gpu to index the whole dataset device="cuda"
    responses = client.index(index_name).search(doc['image_docker'], device='cpu')

    # now retrieve the score for each label and add it to our document
    for lab in label_strings:
        doc[lab.replace(' ','_')] = [r['_score'] for r in responses['hits'] if r['label'] == lab][0]
```

---

TITLE: Querying with Prompting for Thematic Results (Python)
DESCRIPTION: Provides another example of using prompting in a query string to influence search results, this time using thematic descriptors like "xmas, festive, holidays" to curate results for a "cozy sweater".
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_7

LANGUAGE: Python
CODE:

```
query = {"cozy sweater, xmas, festive, holidays":1.0}
```

---

TITLE: Deleting a Marqo Index (Python)
DESCRIPTION: Illustrates how to completely remove a Marqo index. Requires the Marqo client and the index name.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_17

LANGUAGE: python
CODE:

```
results = mq.index("my-first-index").delete()
```

---

TITLE: Creating Marqo Index for Context Vectors (Python)
DESCRIPTION: Demonstrates how to create a new Marqo index specifically configured for generating context vectors from sets of items, often used for personalization or popular/liked product features.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_23

LANGUAGE: python
CODE:

```
# we create another index to create a context vector
index_name_context = 'multimodal-context'
settings = {
	"treatUrlsAndPointersAsImages": True,
	"model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
	"normalizeEmbeddings": True,
}

res = client.create_index(index_name_context, settings_dict=settings)
```

---

TITLE: Calling GPT with Langchain LLMChain (Python)
DESCRIPTION: Initializes an OpenAI LLM instance and an LLMChain. It then calls the chain with retrieved document summaries ('docs') and the user's query ('results['query']') to generate a response from the LLM. Requires Langchain and OpenAI libraries.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_11

LANGUAGE: python
CODE:

```
from langchain.chains import LLMChain
llm = OpenAI(temperature=0.9)
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": docs, "question": results['query']}, return_only_outputs=True)
```

---

TITLE: Indexing Documents with Different Patch Methods - Python
DESCRIPTION: Loops through different image preprocessing patch methods (None, DINO-v2, YOLOX), creates a new Marqo index for each, updates the settings, and adds the prepared documents to the index using a specified device (GPU or CPU). Requires a Marqo client instance and the documents list.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_3

LANGUAGE: Python
CODE:

```
for patch_method in patch_methods:

    index_name = f"visual_search-{str(patch_method).lower()}"

    settings['imagePreprocessing']['patchMethod'] = patch_method

    response = client.create_index(index_name, settings_dict=settings)

    # index the documents on the GPU
    response = client.index(index_name).add_documents(documents, tensor_fields =["image_location"], device='cuda', client_batch_size=50)
```

---

TITLE: Indexing Documents in Marqo - Python
DESCRIPTION: Configures index settings including the model and enabling image pointer handling, creates a Marqo index named 'hot-dogs-100k', and adds the prepared list of documents to the index using the specified device and batch size.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_2

LANGUAGE: python
CODE:

```
settings = {
           "model":'open_clip/ViT-B-32/laion2b_s34b_b79k',
           "treatUrlsAndPointersAsImages": True,
           }
client.create_index("hot-dogs-100k", settings_dict=settings)
responses = client.index("hot-dogs-100k").add_documents(documents, device="cuda", client_batch_size=50, tensor_fields=["image_docker"])
```

---

TITLE: Performing GPT Inference with Langchain (Python)
DESCRIPTION: Demonstrates how to use Langchain's `LLMChain` to call the OpenAI GPT model. It initializes an `OpenAI` instance, creates an `LLMChain` linking the model and the prompt, and executes the chain with prepared context (`summaries`) and conversation history.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_21

LANGUAGE: python
CODE:

```
from langchain.chains import LLMChain
lm = OpenAI(temperature=0.9)
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": docs, "conversation": "wow, what are some of your favorite things to do?", return_only_outputs=True)
```

---

TITLE: Preparing Documents from CSV - Python
DESCRIPTION: Reads image S3 URIs from a CSV file and formats them into a list of dictionaries suitable for Marqo indexing, using the filename as the document ID. Requires pandas.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_1

LANGUAGE: Python
CODE:

```
import pandas as pd
import os

df = pd.read_csv('files.csv')
documents = [{"image_location":s3_uri, '_id':os.path.basename(s3_uri)} for s3_uri in df['s3_uri']]
```

---

TITLE: Defining Multimodal Document Structure (Python)
DESCRIPTION: Shows an example Python dictionary representing a multimodal document for indexing in Marqo. It includes multiple image URLs and text fields like title and description, intended for use with a multimodal model like CLIP.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_10

LANGUAGE: python
CODE:

```
document = {"combined_text_image":
		         {
                "image1":"https://some_image1.png",
		"image2":"https://some_image2.png",
		"image3":"https://some_image3.png",
		"title": "Fresh and Versatile: The Green Cotton T-Shirt for Effortless Style"
		"description": "Crafted from high-quality cotton fabric, this t-shirt offers a soft and breathable feel, ensuring all-day comfort."
			  }
	    }
```

---

TITLE: Defining a Document with Multiple Image Fields (Python)
DESCRIPTION: Creates a Python dictionary representing a document to be indexed in Marqo. It includes an \_id and several fields (top_1 to top_4) containing URLs pointing to images. This structure is designed for multimodal indexing where multiple image fields contribute to a single vector representation.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_25

LANGUAGE: python
CODE:

```
document2 = {"_id": "2",
			 "top_1": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_1.jpg",
			 "top_2": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office2.webp",
			 "top_3": 'https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_3.jpeg',
			 "top_4": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/office_4.jpg"
			 }
```

---

TITLE: Constructing Documents for Context Vectors (Python)
DESCRIPTION: Shows how to structure a document containing multiple image URLs. This document will be used to generate a single context vector representing the collection of items.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_24

LANGUAGE: python
CODE:

```
# create the document that will be created from multiple images
document1 = {"_id": "1",
			 "top_1": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/blue_backpack.jpg",
			 "top_2": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/dark_backpack.jpeg",
			 "top_3": 'https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/green+_backpack.jpg',
			 "top_4": "https://marqo-overall-demo-assets.s3.us-west-2.amazonaws.com/ecommerce/red_backpack.jpg"

			 }
```

---

TITLE: Filtering Marqo Search Results by Language (Python)
DESCRIPTION: This Python code snippet demonstrates how to add a language filter to a Marqo search query. It calls the 'search' function on a Marqo index, passing the query string ('q=query') and a 'filter_string' parameter set to 'language:en' to restrict results to English documents. This requires a Marqo client instance ('mq') and a defined 'query' variable.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_6

LANGUAGE: Python
CODE:

```
mq.index(index_name='my-multilingual-index').search(
    q=query,
    filter_string='language:en'
)
```

---

TITLE: Defining Multimodal Mappings and Indexing Documents (Python)
DESCRIPTION: Defines two different multimodal mappings (mappings1, mappings2) using multimodal_combination with varying weights for image fields. It then indexes document1 and document2 into a Marqo index, specifying "multimodal" as the tensor field and applying the respective mappings to combine the image fields into a single vector.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_26

LANGUAGE: python
CODE:

```
# define how we want to combined
mappings1 = {"multimodal":
               {"type": "multimodal_combination",
                   "weights": {"top_1": 0.40,
                               "top_2": 0.30,
                               "top_3": 0.20,
                               "top_4": 0.10,
                            }}}

# define how we want to combined
mappings2 = {"multimodal":
               {"type": "multimodal_combination",
                   "weights": {"top_1": 0.25,
                               "top_2": 0.25,
                               "top_3": 0.25,
                               "top_4": 0.25,
                            }}}

# index the document
res = client.index(index_name_context).add_documents([document1], tensor_fields=["multimodal"], device=device, mappings=mappings1)

# index the other using a different mappings
res = client.index(index_name_context).add_documents([document2], tensor_fields=["multimodal"], device=device, mappings=mappings2)
```

---

TITLE: Embed Endpoint - Marqo API
DESCRIPTION: Describes the new POST endpoint for generating embeddings for content within a specified index. It accepts single or list content, which can be strings or weighted dictionaries.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/RELEASE.md#_snippet_0

LANGUAGE: shell
CODE:

```
POST /indexes/{index_name}/embed
```

---

TITLE: Iterative Document Sorting with Marqo Python Client
DESCRIPTION: This Python snippet demonstrates an iterative process to sort documents (likely images) based on similarity using the Marqo search client. It starts with an initial document, deletes it from the index, searches for the next most similar document using the current document's data, adds the found document's ID to an ordered list, and repeats the process until all documents are ordered. It requires an initialized Marqo client and an index.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_11

LANGUAGE: Python
CODE:

```
# create a list to store the "sorted" documents
ordered_documents = [current_document['_id']]

for i in range(len(documents)):

    # remove current document
    client.index(index_name).delete_documents([current_document['_id']])

    # now search with it to get next best
    results = client.index(index_name).search(current_document['image_docker'], filter_string="a_face:[0.58 TO 0.99]",
                            searchabel_attributes=['image_docker'], device='cuda')

    next_document = results['hits'][0]

    # now add it
    ordered_documents.append(next_document['_id'])

    current_document = next_document

ordered_images = [files_map[f] for f in ordered_documents]
```

---

TITLE: Run Interactive Chat Interface with Marqo Q&A - Python
DESCRIPTION: Initializes a Marqo client and enters an infinite loop to accept user queries. For each query, it calls the `answer_question` function to get a conversational answer based on the Marqo index and prints the result. Requires a running Marqo instance.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_11

LANGUAGE: python
CODE:

```
import marqo
from SpeechSearch.chatter import answer_question


def main():
    mq = marqo.Client(url="http://localhost:8882")

    index_name = "transcription-index"
    while True:
        query = input("Enter a query: ")
        answer = answer_question(
            query=query,
            limit=15,
            index=index_name,
            mq=mq,
        )
        print(answer)


if __name__ == "__main__":
    main()
```

---

TITLE: Performing Lexical Search with Marqo (Python)
DESCRIPTION: Executes a lexical search query against a specified Marqo index using the BM25 algorithm. The `search_method` parameter is explicitly set to 'LEXICAL' to enable this mode.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_5

LANGUAGE: Python
CODE:

```
results = mq.index(index_name).search("what is the loud clicking sound?",
                                                  search_method="LEXICAL")
```

---

TITLE: Perform Search with OWL-ViT Reranker (Python)
DESCRIPTION: This Python snippet demonstrates how to perform a search query in Marqo, explicitly specifying the 'owl/ViT-B/32' model as a reranker. This enables search-time localization without requiring index-time processing. The code executes the search and prints the first hit from the results.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchLocalization/article.md#_snippet_6

LANGUAGE: python
CODE:

```
response = client.index(index_name).search("brocolli", device="cuda",
        reranker="owl/ViT-B/32")
print(response['hits'][0])
```

---

TITLE: Defining Query and Context Vector for Conditional Search (Python)
DESCRIPTION: Demonstrates the components used in a conditional search query, including the main query term and a context vector derived from a set of items. This method steers search results based on pre-computed item vectors.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_5

LANGUAGE: Python
CODE:

```
query = {"backpack":1.0}
context_vector1 = [.1, ...,.-.8]
```

---

TITLE: Recommend Endpoint - Marqo API
DESCRIPTION: Describes the new POST endpoint for recommending similar documents based on a list of existing document IDs. It performs a search using interpolated vectors.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/RELEASE.md#_snippet_1

LANGUAGE: shell
CODE:

```
POST /indexes/{index_name}/recommend
```

---

TITLE: Run Marqo Docker on AWS EC2
DESCRIPTION: Run the latest Marqo Docker image from marqoai/marqo on an AWS EC2 instance, mapping port 8882 for access.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_18

LANGUAGE: Bash
CODE:

```
docker run --name  marqo -p 8882:8882 marqoai/marqo:latest
```

---

TITLE: Checking Marqo Index Statistics - Python
DESCRIPTION: Retrieves and prints the statistics for the 'hot-dogs-100k' Marqo index, which includes information like the number of documents indexed.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_3

LANGUAGE: python
CODE:

```
print(client.index("hot-dogs-100k").get_stats())
```

---

TITLE: Updating Marqo Documents with New Data
DESCRIPTION: Removes the 'image_docker' field from a list of documents and then updates these documents in the "hot-dogs-100k" Marqo index by adding them again. This process incorporates new data (like calculated scores) into the existing index entries. Requires a Marqo client and a list of documents to update.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_9

LANGUAGE: python
CODE:

```
documents_image_docker = [doc.pop('image_docker') for doc in documents]
responses = client.index("hot-dogs-100k").add_documents(documents, device='cpu', client_batch_size=50, tensor_fields=["image_docker"])
```

---

TITLE: Getting Marqo Index Statistics
DESCRIPTION: Shows how to call the `get_stats` method on a Marqo index object to retrieve information and statistics about the index, such as the number of documents it contains.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_8

LANGUAGE: python
CODE:

```
results = mq.index("my-first-index").get_stats()
```

---

TITLE: Deleting Duplicate Documents in Marqo
DESCRIPTION: Identifies documents with a score very close to 1 (indicating potential duplicates) from search results and deletes them from the specified Marqo index. Requires a Marqo client instance and the index name.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_6

LANGUAGE: python
CODE:

```
documents_delete = [r['_id'] for r in results['hits'] if r['_score'] > 0.99999]

client.index(index_name).delete_documents(documents_delete)
```

---

TITLE: No Model Option - Marqo Index Creation
DESCRIPTION: Describes the `no_model` option for index creation, which allows creating indexes that do not perform vectorization, useful for using custom vectors without mixing them with Marqo-generated ones.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/RELEASE.md#_snippet_6

LANGUAGE: text
CODE:

```
no_model
```

---

TITLE: IN Operator - Marqo Query DSL
DESCRIPTION: Describes the addition of the `IN` operator to the query filter string DSL for structured indexes. It allows restricting text and integer fields to be within a specified list of values.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/RELEASE.md#_snippet_5

LANGUAGE: text
CODE:

```
IN
```

---

TITLE: Optional q Parameter - Marqo Search API
DESCRIPTION: Describes the optional `q` parameter for the search endpoint, particularly useful when providing context vectors to search across documents with custom vector fields.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/RELEASE.md#_snippet_7

LANGUAGE: text
CODE:

```
q
```

---

TITLE: Implementing Conversational Loop (Python)
DESCRIPTION: Shows the core logic for an iterative chat. It appends the human question to history, searches Marqo for relevant background, prepares the context, calls the LLM chain with updated history and context, and appends the AI response to history. This loop drives the conversation.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_22

LANGUAGE: python
CODE:

```
# how many background pieces of information to use
n_background = 2
# we keep track of the human and superhero responses
history.append(f"\nHUMAN:{question}")
# search for background related to the question
results = mq.index(index_name).search(question, filter_string=f"name:({persona})", limit=20)
# optionally crop the text to the highlighted region to fit within the context window
highlights, texts = extract_text_from_highlights(results, token_limit=150)
# add the truncated/cropped text to the data structure for langchain
summaries = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts[:n_background])]
# inference with the LLM
chain_qa = LLMChain(llm=llm, prompt=prompt)
llm_results = chain_qa({"summaries": summaries, "conversation": "\n".join(history)}, return_only_outputs=False)
# add to the conversation history
history.append(llm_results['text'])
```

---

TITLE: Scoring LLM Response Relevance to Sources (Python)
DESCRIPTION: Uses a cross-encoder function ('predict_ce') to score the relevance of the LLM's generated text ('llm_results['text']') against the original source documents ('texts'). This helps identify which sources were most likely used by the LLM. Requires a pre-trained cross-encoder model.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_12

LANGUAGE: python
CODE:

```
scores = predict_ce(llm_results['text'], texts)
```

---

TITLE: Generate Docker Paths for Images
DESCRIPTION: Use glob to find all JPG files in the local data directory and generate corresponding HTTP URLs using 'http://host.docker.internal:8222/' as the base, making them addressable from within the Docker container.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ImageSearchGuide/ImageSearchGuide.md#_snippet_5

LANGUAGE: python
CODE:

```
import glob
import os

# Find all the local images
locators = glob.glob(local_dir + '*.jpg')

# Generate docker path for local images
docker_path = "http://host.docker.internal:8222/"
image_docker = [docker_path + os.path.basename(f) for f in locators]

print(image_docker)
```

---

TITLE: Transcribing Audio Segments with Hugging Face S2T Model in Python
DESCRIPTION: Describes the transcribe method which takes a list of audio data arrays and a sample rate. It batches the audio segments, pads short ones, processes them using the loaded Hugging Face S2T model and processor, and returns a list of transcribed text strings.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_6

LANGUAGE: python
CODE:

```
def transcribe(self, datas: List[np.ndarray], samplerate: int = 16000) -> List[str]:
    batches = []
    batch = []
    i = 0
    for data in datas:
        # pad short audio
        if data.shape[0] < 400:
            data = np.pad(data, [(0, 400)], mode="constant")

        batch.append(data)
        i += 1
        if i > self.transcription_batch_size:
            batches.append(batch)
            i = 0
            batch = []
    if batch:
        batches.append(batch)

    transcriptions = []
    for batch in tqdm(
        batches, desc=f"Processing with batch size {self.transcription_batch_size}"
    ):
        inputs = self.transcription_processor(
            batch, sampling_rate=samplerate, return_tensors="pt", padding=True
        )
        generated_ids = self.transcription_model.generate(
            inputs["input_features"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )
        transcription_batch = self.transcription_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        transcriptions += transcription_batch

    return transcriptions
```

---

TITLE: Adding Documents to Marqo Index (Python)
DESCRIPTION: Adds documents to the 'my-multilingual-index'. It specifies the device for processing (e.g., 'cuda' for GPU, 'cpu' for CPU). Documents are provided as a list of dictionaries, including fields like `_id`, `language`, `text`, `celex_id`, and `labels`. `tensor_fields` specifies which fields should be used for tensor generation.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiLingual/article.md#_snippet_4

LANGUAGE: python
CODE:

```
mq.index(index_name='my-multilingual-index').add_documents(
    device='cuda',
    documents=[{
                    "_id": str(doc_id),
                    "language": lang,
                    'text': sub_doc,
                    'celex_id': doc['celex_id'],
                    'labels': str(doc['labels'])
                }],
    tensor_fields=["language", "text", "labels"]
)
```

---

TITLE: Preparing Image Files for Marqo Indexing - Python
DESCRIPTION: Sets up directory paths, starts a simple HTTP server to serve images from within Docker, finds all JPG files, maps local paths to Docker-accessible paths, and creates a list of dictionaries formatted as Marqo documents for indexing.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/StableDiffusion/hot-dog-100k.md#_snippet_1

LANGUAGE: python
CODE:

```
import glob
import os
from marqo import Client

# this should be where the images are unzipped
images_directory = 'hot-dog-100k/'

# the images are accessed via docker from here - you will be able
# to access them at something like http://[::]:8000/ or http://localhost:8000/
docker_path = 'http://host.docker.internal:8222/'

# we start an image server for easier access from within docker
pid = subprocess.Popen(['python3', '-m', 'http.server', '8222', '--directory', images_directory], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# now find all the files
files = glob.glob(images_directory + "/*.jpg")

# we want to map the filename only with its docker path
files_map = {os.path.basename(f):f for f in files}

# update them to use the correct path
files_docker = [f.replace(images_directory, docker_path) for f in files]

# now we create our documents for indexing -  a list of python dicts
documents = [{"image_docker":file_docker, '_id':os.path.basename(file_docker)} for file_docker,file_local in zip(files_docker, files)]
```

---

TITLE: Performing Speaker Diarisation with pyannote.audio in Python
DESCRIPTION: Details the annotate method which uses the pre-configured annotation_pipeline to process an audio file. It extracts speaker segments, potentially splitting long segments into 30-second chunks, and returns a list of tuples containing start time, end time, and speaker labels.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_5

LANGUAGE: python
CODE:

```
def annotate(self, file: str) -> List[Tuple[float, float, Set[str]]]:
    diarization = self.annotation_pipeline(file)
    speaker_times = []
    for t in diarization.get_timeline():
        start, end = t.start, t.end
        # reduce to 30 second chunks in case of long segments
        while end - start > 0:
            speaker_times.append(
                (start, min(start + 30, end), diarization.get_labels(t))
            )
            start += 30

    return speaker_times
```

---

TITLE: Creating a Marqo Index for Multimodal Objects (Python)
DESCRIPTION: Defines a new index name and settings for a Marqo index. The settings enable treating URLs/pointers as images, specify a multimodal model (open_clip), and enable embedding normalization. It then calls client.create_index to create the new index with these configurations, preparing it for indexing documents with combined text and image fields.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_28

LANGUAGE: python
CODE:

```
# we will create a new index for the multimodal objects
index_name_mm_objects = 'multimodal-objects'
settings = {
	"treatUrlsAndPointersAsImages": True,
	"model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
	"normalizeEmbeddings": True,
}

res = client.create_index(index_name_mm_objects, settings_dict=settings)
```

---

TITLE: Starting Marqo-OS Docker Container for Unit Tests (Bash)
DESCRIPTION: This command sequence removes any existing 'marqo-os' container and then runs a new one. It maps ports 9200 and 9600, sets the discovery type to single-node, names the container 'marqo-os', and uses the 'marqoai/marqo-os:0.0.3' image. This is a prerequisite for running Marqo unit tests.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/CONTRIBUTING.md#_snippet_0

LANGUAGE: Bash
CODE:

```
docker rm -f marqo-os
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name marqo-os marqoai/marqo-os:0.0.3
```

---

TITLE: Install Nvidia Docker 2
DESCRIPTION: Install the nvidia-docker2 package using apt-get. This package is required to enable GPU support within Docker containers.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_17

LANGUAGE: Bash
CODE:

```
$ sudo apt-get install -y nvidia-docker2
```

---

TITLE: Searching Marqo Index with Filter (Python)
DESCRIPTION: Performs a search query ("what is your hobby") against the Marqo index, but applies a filter ('filter_string') to restrict the search results to documents where the 'name' field matches the specified 'persona' ("Jack Smith"). This allows searching within specific subsets of the data.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_17

LANGUAGE: python
CODE:

```
persona = "Jack Smith"
results = mq.index(index_name).search('what is your hobby', filter_string=f'name:({persona})')
```

---

TITLE: Defining Example NPC Documents (Python)
DESCRIPTION: Creates a list of dictionaries representing example documents for Non-Player Characters (NPCs). Each document contains a 'name' and 'text' field, simulating character backstories or information. This data is prepared for indexing in Marqo.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_13

LANGUAGE: python
CODE:

```
document1 = {"name":"Sara Lee", "text":"my name is Sara Lee"}
document2 = {"name":"Jack Smith", "text":"my name is Jack Smith"}
document3 = {"name":"Sara Lee", "text":"Sara worked as a research assistant for a university before becoming a park ranger."}
documents = [document1, document2, document3]
```

---

TITLE: Adding Documents to Marqo Index (Python)
DESCRIPTION: This snippet demonstrates how to fetch external data, format it into a list of documents, and add these documents to a specified Marqo index using the `add_documents` method. It specifies which fields (`name` and `text`) should be indexed as tensors.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_23

LANGUAGE: python
CODE:

```
from iron_data import get_extra_data
extra_docs = [{"text":text, "name":persona} for text in get_extra_data()]
res = mq.index(index_name).add_documents(extra_docs, tensor_fields = ["name", "text"])
```

---

TITLE: Checking NVIDIA Driver Status (Shell)
DESCRIPTION: Use this command in a terminal to check if NVIDIA drivers are installed and view their version. Lack of output may indicate a driver issue.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_19

LANGUAGE: Shell
CODE:

```
nvidia-smi
```

---

TITLE: Checking PyTorch GPU/CUDA Status (Python)
DESCRIPTION: This Python code snippet uses the PyTorch library to check if a GPU is available, retrieve the CUDA version being used by PyTorch, and count the number of available GPU devices. Requires PyTorch installed.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_20

LANGUAGE: Python
CODE:

```
import torch

torch.cuda.is_available() # is a GPU available
torch.version.cuda        # get the CUDA version
torch.cuda.device_count() # get the number of devices
```

---

TITLE: Preparing Langchain Documents from Marqo Highlights (Python)
DESCRIPTION: Imports the Langchain `Document` class, performs a Marqo search, extracts the text from the search result highlights, and formats them into a list of Langchain `Document` objects, prefixing each with a source index.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_9

LANGUAGE: Python
CODE:

```
from langchain.docstore.document import Document
results = client.index(index_name).search(question)
text = [res['_highlights'][0]['text'] for res in results['hits']]
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```

---

TITLE: Running Marqo via Uvicorn
DESCRIPTION: Sets the necessary environment variables and then starts the Marqo application using uvicorn, serving the `api:app` from the `src/marqo/tensor_search` directory on host 0.0.0.0 and port 8882 with auto-reloading enabled.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_8

LANGUAGE: bash
CODE:

```
export MARQO_ENABLE_BATCH_APIS=true
export MARQO_LOG_LEVEL=debug
export VESPA_CONFIG_URL=http://localhost:19071
export VESPA_DOCUMENT_URL=http://localhost:8080
export VESPA_QUERY_URL=http://localhost:8080
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
cd src/marqo/tensor_search
uvicorn api:app --host 0.0.0.0 --port 8882 --reload
```

---

TITLE: Defining LLM Prompt Template (Python)
DESCRIPTION: Defines the string template used for the large language model prompt. It includes placeholders for background summaries and the conversation history, setting the persona and instructions for the AI.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_18

LANGUAGE: python
CODE:

```
template = """
The following is a conversation with a fictional superhero in a movie.
BACKGROUND is provided which describes some of the history and powers of the superhero.
The conversation should always be consistent with this BACKGROUND.
Continue the conversation as the superhero in the movie.
You are very funny and talkative and **always** talk about your superhero skills in relation to your BACKGROUND.
BACKGROUND:
=========
{summaries}
=========
Conversation:
{conversation}
"""
```

---

TITLE: Install Marqo Base GPU Requirements (Bash)
DESCRIPTION: Installs the necessary dependencies for the Marqo base project specifically for AMD64 GPU environments using pip and the requirements file.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/tests/integ_tests/inference/embeddings_reference/info.txt#_snippet_0

LANGUAGE: bash
CODE:

```
pip install -r marqo-base/requirements/amd64-gpu-requirements.txt
```

---

TITLE: Preparing Langchain Documents with Token-Aware Context (Python)
DESCRIPTION: Uses a hypothetical `extract_text_from_highlights` function with a token limit to get potentially expanded text around highlights from Marqo results. It then formats this text into a list of Langchain `Document` objects.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_10

LANGUAGE: Python
CODE:

```
highlights, texts = extract_text_from_highlights(results, token_limit=150)
docs = [Document(page_content=f"Source [{ind}]:"+t) for ind,t in enumerate(texts)]
```

---

TITLE: Creating Langchain Prompt Object (Python)
DESCRIPTION: Initializes a Langchain `PromptTemplate` object using the defined template string and specifies the input variables (`summaries`, `conversation`). This object is used to format the prompt before sending it to the LLM. Requires the `langchain` library.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_19

LANGUAGE: python
CODE:

```
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=template, input_variables=["summaries", "conversation"])
```

---

TITLE: Setting Environment Variables for IDE
DESCRIPTION: Lists the environment variables required when running Marqo locally through an IDE like PyCharm. These variables configure batch APIs, logging level, preloaded models, and Vespa endpoint URLs.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_7

LANGUAGE: bash
CODE:

```
MARQO_ENABLE_BATCH_APIS=true
MARQO_LOG_LEVEL=debug
MARQO_MODELS_TO_PRELOAD=[]
VESPA_CONFIG_URL=http://localhost:19071
VESPA_DOCUMENT_URL=http://localhost:8080
VESPA_QUERY_URL=http://localhost:8080
```

---

TITLE: Run Locust Performance Tests (Shell)
DESCRIPTION: Demonstrates how to run Locust tests using the default config, overriding settings via CLI parameters, specifying index/model names via environment variables, and running against Marqo Cloud with an API key.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/perf_tests/README.md#_snippet_2

LANGUAGE: shell
CODE:

```
# this will use the default config in locust.conf file
locust

# Alternatively you can specify CLI params to override the default settings
locust -u <user_count> -r <spawn-rate> -t <duraion> -H <host> -f <test_file>

# When run locally, by default it creates an index `locust-test` with `hf/e5-base-v2` model,
# You can specify the name of the index or model by using env vars
MARQO_INDEX_NAME=<index_name> MARQO_INDEX_MODEL_NAME=<model_name> locust

# You can also run against a Marqo Cloud instance by providing the host and API key
MARQO_INDEX_NAME=<index_name> MARQO_CLOUD_API_KEY=<your api key> locust -H <host>

# After the run, a test report will be generated to report/report.html file
```

---

TITLE: Run Marqo Docker Container (Bash)
DESCRIPTION: Runs the latest Marqo Docker container, naming it 'marqo', mapping port 8882 from the container to the host, and adding a host entry for accessing the host machine's gateway. This command initializes the Marqo service required for the demo.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ClothingCLI/README.md#_snippet_1

LANGUAGE: bash
CODE:

```
docker run --name marqo -it -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest
```

---

TITLE: Running Marqo with Docker
DESCRIPTION: Provides the necessary Docker commands to remove any existing Marqo container, pull the latest Marqo image, and run a new container, mapping port 8882. This requires Docker to be installed and allocated sufficient memory and storage.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_3

LANGUAGE: bash
CODE:

```
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -p 8882:8882 marqoai/marqo:latest
```

---

TITLE: Starting Python HTTP Server (Shell)
DESCRIPTION: Sets up a simple HTTP server using Python's built-in module to serve files from the local directory on port 8222. This server is required for the Marqo Docker container to access local files, as mentioned in the linked GitHub issue.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/ClothingStreamlit/README.md#_snippet_0

LANGUAGE: Shell
CODE:

```
python3 -m http.server 8222
```

---

TITLE: Creating Langchain Prompt Template Object (Python)
DESCRIPTION: Installs the Langchain library and creates a `PromptTemplate` object using the previously defined template string. This object manages the prompt structure and input variables ('summaries', 'question').
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_8

LANGUAGE: Python
CODE:

```
pip install langchain
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=template, input_variables=["summaries", "question"])
```

---

TITLE: Run Marqo Docker Container with GPU
DESCRIPTION: Remove any existing Marqo container, build the Docker image, and run the container enabling access to all available GPUs using the --gpus all flag, mapping port 8882.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_14

LANGUAGE: Bash
CODE:

```
docker rm -f marqo &&\
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 && \
     docker run --name marqo --gpus all -p 8882:8882 marqo_docker_0
```

---

TITLE: Creating a Multimodal Text Query in Python
DESCRIPTION: Demonstrates how to construct a multi-part query using a Python dictionary. Each key represents a query component (e.g., "green shirt"), and the value is its positive weight, indicating its importance in the search. This acts as a form of query expansion.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_0

LANGUAGE: python
CODE:

```
query = {"green shirt":1.0, "short sleeves":1.0}
```

---

TITLE: Performing Vector Search with Marqo (Python)
DESCRIPTION: Executes a vector search query against a specified Marqo index using the default search method (vector search). The query string is embedded and used to find semantically similar documents.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/article/article.md#_snippet_4

LANGUAGE: Python
CODE:

```
results = mq.index(index_name).search("what is the loud clicking sound?")
```

---

TITLE: Retrieving a Document by ID with Marqo
DESCRIPTION: Demonstrates how to use the `get_document` method on a Marqo index object to retrieve a specific document from the index using its unique identifier (`document_id`). Note that adding a document with the same ID will update it.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_7

LANGUAGE: python
CODE:

```
result = mq.index("my-first-index").get_document(document_id="article_591")
```

---

TITLE: Combining Image and Text in a Multimodal Query in Python
DESCRIPTION: Demonstrates how to combine an image URL and a text term in a single multimodal query using a Python dictionary. The image URL and text term ("RED") are given positive weights to find items similar to the image that also match the text description.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/MultiModalSearch/article.md#_snippet_4

LANGUAGE: python
CODE:

```
query = {image_url:1.0, "RED":1.0}
```

---

TITLE: Starting Single-Node Vespa (Command Line)
DESCRIPTION: Initiates a single-node Vespa setup using the `vespa_local.py` script. This runs one Vespa container acting as config, api, and content node, suitable for basic local development and testing.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/scripts/vespa_local/README.md#_snippet_2

LANGUAGE: commandline
CODE:

```
python vespa_local.py start
```

---

TITLE: Installing Marqo Dependencies
DESCRIPTION: Installs the necessary Python dependencies for running Marqo locally. Different requirements files are used depending on the machine architecture (AMD or ARM) and whether development/test dependencies are needed.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_1

LANGUAGE: bash
CODE:

```
pip install -r marqo-base/requirements/amd64-gpu-requirements.txt  # For AMD machine, no matter you have a GPU or not
```

LANGUAGE: bash
CODE:

```
pip install -r marqo-base/requirements/arm64-requirements.txt  # For ARM machine
```

LANGUAGE: bash
CODE:

```
pip install -r marqo/requirements.dev.txt  # For test purposes
```

---

TITLE: Searching a Marqo Index Using an Image URL (Python)
DESCRIPTION: This code illustrates how to perform a search in Marqo by providing an image URL as the query. Marqo will use the image content to find similar documents in the index.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/README.md#_snippet_13

LANGUAGE: python
CODE:

```
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

---

TITLE: Install Latest Redis on Older Ubuntu
DESCRIPTION: Add the official Redis repository and GPG key to install the latest version of redis-server on older Ubuntu distributions using apt.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_11

LANGUAGE: Bash
CODE:

```
apt install lsb-release
curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
apt-get update
apt-get install redis-server -y
```

---

TITLE: Starting Multi-Node Vespa (Command Line)
DESCRIPTION: Starts a multi-node Vespa cluster using the `vespa_local.py` script, allowing specification of the number of shards and replicas. This command example sets up a cluster with 2 shards and 1 replica.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/scripts/vespa_local/README.md#_snippet_3

LANGUAGE: commandline
CODE:

```
python vespa_local.py start --Shards 2 --Replicas 1
```

---

TITLE: Checking Single-Node Vespa Health (Command Line)
DESCRIPTION: Checks the health status of a single-node Vespa instance by sending a curl request to the state API endpoint on localhost port 19071. A successful response indicates the node is ready.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/scripts/vespa_local/README.md#_snippet_5

LANGUAGE: commandline
CODE:

```
curl -s http://localhost:19071/state/v1/health
```

---

TITLE: Build and Run Marqo Docker Container
DESCRIPTION: Remove any existing Marqo container, build the Docker image from the current directory using BuildKit, and run the container mapping port 8882.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_13

LANGUAGE: Bash
CODE:

```
docker rm -f marqo &&\
     DOCKER_BUILDKIT=1 docker build . -t marqo_docker_0 \
     docker run --name marqo -p 8882:8882 marqo_docker_0
```

---

TITLE: Deploying Vespa Application Configuration (Command Line)
DESCRIPTION: Deploys the Vespa application configuration files located in the current directory to the running Vespa instance(s) using the `vespa_local.py` script. This step is necessary after starting Vespa.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/scripts/vespa_local/README.md#_snippet_4

LANGUAGE: commandline
CODE:

```
python vespa_local.py deploy-config
```

---

TITLE: Setting Vespa Version (Command Line)
DESCRIPTION: Sets the desired Vespa version for the setup script by exporting the VESPA_VERSION environment variable. The default is 8.431.32 if not set. Use 'latest' for the most recent version.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/scripts/vespa_local/README.md#_snippet_0

LANGUAGE: commandline
CODE:

```
export VESPA_VERSION="latest"
```

---

TITLE: Bulk Downloading Audio from File (Python)
DESCRIPTION: Provides methods for downloading multiple audio sources listed in a file. download_from_file reads URLs from a file, and multiprocess_read_url_sources uses a multiprocessing pool to process these URLs in parallel via the read_url_source method, which dispatches to either download_from_youtube or download_from_web based on the URL.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_3

LANGUAGE: python
CODE:

```
def download_from_file(self, file):
    urls = []
    with open(file, "r") as f:
        for url in f.readlines():
            urls.append(url.strip())
    self.multiprocess_read_url_sources(urls)

def multiprocess_read_url_sources(self, sources: List[str]):
    pool = Pool(os.cpu_count())
    pool.map(self.read_url_source, sources)

def read_url_source(self, source: str):
    if "www.youtube.com" in source:
        return self.download_from_youtube(source)

    return self.download_from_web(source)
```

---

TITLE: Extracting Marqo OpenAPI Spec (Shell)
DESCRIPTION: This curl command retrieves the "openapi.json" file, which contains the Swagger API specification, from a Marqo instance running locally on port 8882.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/src/marqo/README.md#_snippet_22

LANGUAGE: Shell
CODE:

```
curl http://localhost:8882/openapi.json
```

---

TITLE: Running Chat Agent Example (Bash)
DESCRIPTION: Executes the 'ironman.py' Python script, which demonstrates a chat agent with history or NPC capabilities using the configured Marqo and OpenAI environment.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/GPT-examples/readme.md#_snippet_4

LANGUAGE: Bash
CODE:

```
python ironman.py
```

---

TITLE: Initializing AudioWrangler and Helper Methods (Python)
DESCRIPTION: Defines the AudioWrangler class constructor, which sets up output and temporary directories. Includes helper methods convert_to_wav for converting audio files to WAV format using Pydub and \_move_to_output for moving processed files to the final output directory using shutil.
SOURCE: https://github.com/marqo-ai/marqo/blob/mainline/examples/SpeechProcessing/article/article.md#_snippet_0

LANGUAGE: python
CODE:

```
ABS_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))

class AudioWrangler():
    def __init__(self, output_path: str, clean_up: bool = True):

        self.output_path = output_path

        self.tmp_dir = 'downloads'

        os.makedirs(os.path.join(ABS_FILE_FOLDER, self.tmp_dir), exist_ok=True)

        if clean_up:
            self.clean_up()

		def convert_to_wav(self, fpath: str):
        sound = AudioSegment.from_file(fpath)
        wav_path = ''.join([p for p in fpath.split(".")[:-1]]) + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path

		def _move_to_output(self, file):
        target = os.path.join(self.output_path, os.path.basename(file))
        shutil.move(file, target)
        return target
```
