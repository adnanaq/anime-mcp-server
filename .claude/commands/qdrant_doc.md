TITLE: Installing Qdrant Python Client
DESCRIPTION: This command installs the official Qdrant client library for Python using pip, the standard package installer for Python. It is the first step to integrate Qdrant into a Python application, enabling interaction with Qdrant instances.
SOURCE: https://github.com/qdrant/qdrant/blob/master/README.md#_snippet_0

LANGUAGE: Bash
CODE:

```
pip install qdrant-client
```

---

TITLE: Pulling Qdrant Docker Image - Bash
DESCRIPTION: This command pulls the latest pre-built Qdrant Docker image from DockerHub, making it available on your local machine for running Qdrant instances.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_0

LANGUAGE: bash
CODE:

```
docker pull qdrant/qdrant
```

---

TITLE: Initializing Qdrant Python Client Locally
DESCRIPTION: This Python code demonstrates how to initialize the Qdrant client for local use. It provides two options: an in-memory instance for quick testing and CI/CD, or a persistent instance that saves changes to disk, suitable for fast prototyping and local development.
SOURCE: https://github.com/qdrant/qdrant/blob/master/README.md#_snippet_1

LANGUAGE: Python
CODE:

```
from qdrant_client import QdrantClient
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD
# OR
client = QdrantClient(path="path/to/db")  # Persists changes to disk, fast prototyping
```

---

TITLE: Performing Basic Vector Search with cURL
DESCRIPTION: This `curl` command sends a POST request to the Qdrant API to perform a basic vector search on 'test_collection'. It specifies a vector and requests the top 3 nearest points. This query does not include any filters.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_7

LANGUAGE: bash
CODE:

```
curl -L -X POST 'http://localhost:6333/collections/test_collection/points/search' \
    -H 'Content-Type: application/json' \
    --data-raw '{
        "vector": [0.2,0.1,0.9,0.7],
        "top": 3
    }'
```

---

TITLE: Connecting Python Client to Remote Qdrant Instance
DESCRIPTION: This Python code connects the Qdrant client to an existing Qdrant instance running on `http://localhost:6333`. This is used when Qdrant is deployed as a separate service, such as via Docker, allowing a Python application to interact with the remote Qdrant server.
SOURCE: https://github.com/qdrant/qdrant/blob/master/README.md#_snippet_3

LANGUAGE: Python
CODE:

```
qdrant = QdrantClient("http://localhost:6333") # Connect to existing Qdrant instance
```

---

TITLE: Adding Points to Qdrant Collection - Bash
DESCRIPTION: This `curl` command adds multiple vector points with associated payloads to the 'test_collection' in Qdrant. Each point has a unique ID, a vector, and optional payload data, demonstrating how to ingest data into a collection.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_6

LANGUAGE: bash
CODE:

```
curl -L -X PUT 'http://localhost:6333/collections/test_collection/points?wait=true' \
    -H 'Content-Type: application/json' \
    --data-raw '{
        "points": [
          {"id": 1, "vector": [0.05, 0.61, 0.76, 0.74], "payload": {"city": "Berlin"}},
          {"id": 2, "vector": [0.19, 0.81, 0.75, 0.11], "payload": {"city": ["Berlin", "London"] }},
          {"id": 3, "vector": [0.36, 0.55, 0.47, 0.94], "payload": {"city": ["Berlin", "Moscow"] }},
          {"id": 4, "vector": [0.18, 0.01, 0.85, 0.80], "payload": {"city": ["London", "Moscow"] }},
          {"id": 5, "vector": [0.24, 0.18, 0.22, 0.44], "payload": {"count": [0] }},
          {"id": 6, "vector": [0.35, 0.08, 0.11, 0.44]}
        ]
    }'
```

---

TITLE: Creating Qdrant Collection with Dot Product Metric - Bash
DESCRIPTION: This `curl` command creates a new collection named 'test_collection' in Qdrant, specifying that its vectors should have a size of 4 and use 'Dot' product for distance calculation. It's a PUT request to the collections API endpoint.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_4

LANGUAGE: bash
CODE:

```
curl -X PUT 'http://localhost:6333/collections/test_collection' \
    -H 'Content-Type: application/json' \
    --data-raw '{
        "vectors": {
          "size": 4,
          "distance": "Dot"
        }
    }'
```

---

TITLE: Performing Filtered Vector Search with cURL
DESCRIPTION: This `curl` command performs a vector search on 'test_collection' with an added filter. The `filter` object uses a `should` clause to match points where the 'city' key has a value of 'London', effectively narrowing down the search results based on payload data.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_9

LANGUAGE: bash
CODE:

```
curl -L -X POST 'http://localhost:6333/collections/test_collection/points/search' \
    -H 'Content-Type: application/json' \
    --data-raw '{
      "filter": {
          "should": [
              {
                  "key": "city",
                  "match": {
                      "value": "London"
                  }
              }
          ]
      },
      "vector": [0.2, 0.1, 0.9, 0.7],
      "top": 3
  }'
```

---

TITLE: Running Qdrant Docker Container (Default) - Bash
DESCRIPTION: This command runs the Qdrant Docker container with its default configuration, mapping port 6333 from the container to port 6333 on the host, allowing access to the Qdrant API.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_1

LANGUAGE: bash
CODE:

```
docker run -p 6333:6333 qdrant/qdrant
```

---

TITLE: Running Qdrant as a Docker Container
DESCRIPTION: This command runs the Qdrant vector database as a Docker container, mapping port 6333 from the container to the host. This allows clients to connect to the Qdrant instance, providing a robust way to deploy and test Qdrant locally in a client-server setup.
SOURCE: https://github.com/qdrant/qdrant/blob/master/README.md#_snippet_2

LANGUAGE: Bash
CODE:

```
docker run -p 6333:6333 qdrant/qdrant
```

---

TITLE: Running Qdrant Docker Container (Bash)
DESCRIPTION: This command runs the Qdrant Docker container, mapping port 6333 from the container to the host. This provides basic access to the Qdrant service at `localhost:6333`.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_2

LANGUAGE: bash
CODE:

```
docker run -p 6333:6333 qdrant/qdrant
```

---

TITLE: Running Qdrant Docker Container with Custom Configuration - Bash
DESCRIPTION: This command runs the Qdrant Docker container with custom volume mounts for persistent storage, snapshots, and a custom configuration file. It allows fine-grained control over data persistence and engine settings.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_3

LANGUAGE: bash
CODE:

```
docker run -p 6333:6333 \
    -v $(pwd)/path/to/data:/qdrant/storage \
    -v $(pwd)/path/to/snapshots:/qdrant/snapshots \
    -v $(pwd)/path/to/custom_config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant
```

---

TITLE: Running Qdrant Docker Container with Custom Configuration and Volumes (Bash)
DESCRIPTION: This command runs the Qdrant Docker container with persistent storage, snapshots, and a custom configuration file mounted as volumes. This allows for fine-grained control over data persistence and service behavior, ensuring data is not lost when the container is removed.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_3

LANGUAGE: bash
CODE:

```
docker run -p 6333:6333 \
    -v $(pwd)/path/to/data:/qdrant/storage \
    -v $(pwd)/path/to/snapshots:/qdrant/snapshots \
    -v $(pwd)/path/to/custom_config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant
```

---

TITLE: Retrieving Qdrant Collection Information - Bash
DESCRIPTION: This `curl` command retrieves detailed information about the 'test_collection' from Qdrant, including its status, vector count, segment details, and configuration parameters. It's a GET request to the collection's API endpoint.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_5

LANGUAGE: bash
CODE:

```
curl 'http://localhost:6333/collections/test_collection'
```

---

TITLE: Pulling Pre-built Qdrant Docker Image (Bash)
DESCRIPTION: This command pulls the latest pre-built Qdrant Docker image from DockerHub. It's used to quickly obtain a ready-to-use Qdrant container without needing to build it from source.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_1

LANGUAGE: bash
CODE:

```
docker pull qdrant/qdrant
```

---

TITLE: Expected Response for Basic Vector Search
DESCRIPTION: This JSON object represents the expected response from Qdrant for a basic vector search query. It includes a list of `result` points, each with an `id`, `score`, `payload`, and `version`, along with the `status` and `time` taken for the operation.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_8

LANGUAGE: json
CODE:

```
{
  "result": [
    { "id": 4, "score": 1.362, "payload": null, "version": 0 },
    { "id": 1, "score": 1.273, "payload": null, "version": 0 },
    { "id": 3, "score": 1.208, "payload": null, "version": 0 }
  ],
  "status": "ok",
  "time": 0.000055785
}
```

---

TITLE: Expected Response for Filtered Vector Search
DESCRIPTION: This JSON object shows the expected response for a vector search query that includes a filter. The `result` array contains only points that satisfy the specified filter criteria (e.g., city is London), demonstrating how filtering affects the returned points and their scores.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_10

LANGUAGE: json
CODE:

```
{
  "result": [
    { "id": 4, "score": 1.362 },
    { "id": 2, "score": 0.871 }
  ],
  "status": "ok",
  "time": 0.000093972
}
```

---

TITLE: Building Qdrant Docker Image from Source - Bash
DESCRIPTION: This command builds a custom Qdrant Docker image from the current directory's Dockerfile, tagging it as 'qdrant/qdrant'. This is useful for custom configurations or development.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/QUICK_START.md#_snippet_2

LANGUAGE: bash
CODE:

```
docker build . --tag=qdrant/qdrant
```

---

TITLE: Building Qdrant Docker Image from Source (Bash)
DESCRIPTION: This command builds a Docker image for Qdrant from the current source directory, tagging it as `qdrant/qdrant`. It's used for creating a custom image based on local changes or for development purposes.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_0

LANGUAGE: bash
CODE:

```
docker build . --tag=qdrant/qdrant
```

---

TITLE: Building and Running Qdrant Locally (Shell)
DESCRIPTION: This command builds the Qdrant executable in release mode using `cargo` and then runs the compiled application. This is the standard way to run Qdrant from source in a local development environment for testing or direct interaction.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_7

LANGUAGE: shell
CODE:

```
cargo build --release --bin qdrant

./target/release/qdrant
```

---

TITLE: Installing Qdrant System Dependencies on Debian/Ubuntu (Shell)
DESCRIPTION: This command updates package lists, upgrades installed packages, and installs essential build tools and libraries required for compiling Qdrant on Debian-based systems. It includes compilers, `cmake`, and `jq` among others, necessary for the build process.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_5

LANGUAGE: shell
CODE:

```
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y curl unzip gcc-multilib \
        clang cmake jq \
        g++-9-aarch64-linux-gnu \
        gcc-9-aarch64-linux-gnu
```

---

TITLE: Formatting Rust Code with rustfmt
DESCRIPTION: This command runs `rustfmt`, the Rust code formatter, across the entire project. It ensures consistent code style and adherence to the project's formatting guidelines before submitting a pull request.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/CONTRIBUTING.md#_snippet_0

LANGUAGE: Shell
CODE:

```
cargo +nightly fmt --all
```

---

TITLE: Linting Rust Code with Clippy
DESCRIPTION: This command executes `clippy`, the Rust linter, across all features and workspaces of the project. It helps identify common mistakes and improve code quality, which is a prerequisite for pull request submission.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/CONTRIBUTING.md#_snippet_1

LANGUAGE: Shell
CODE:

```
cargo clippy --workspace --all-features
```

---

TITLE: Installing Rustfmt Toolchain (Shell)
DESCRIPTION: This command installs the `rustfmt` component for the Rust toolchain, which is essential for formatting Rust code according to standard conventions. It is a prerequisite for local Qdrant development to maintain code style consistency.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_4

LANGUAGE: shell
CODE:

```
rustup component add rustfmt
```

---

TITLE: Instrumenting Rust Functions for Tracing with `tracing::instrument`
DESCRIPTION: This snippet demonstrates how to instrument Rust functions for profiling using the `tracing::instrument` macro. It shows two approaches: conditional compilation with `#[cfg_attr]` for optional `tracing` dependency, and direct application of `#[tracing::instrument]` for quick profiling. This requires enabling the `tracing` feature during compilation.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_17

LANGUAGE: Rust
CODE:

```
// `tracing` crate is an *optional* dependency in `lib/*` crates, so if you want the code to compile
// when `tracing` feature is disabled, you have to use `#[cfg_attr(...)]`...
//
// See https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute
#[cfg_attr(feature = "tracing", tracing::instrument)]
fn my_function(some_parameter: String) {
    // ...
}

// ...or if you just want to do some quick-and-dirty profiling, you can use `#[tracing::instrument]`
// directly, just don't forget to add `--features tracing` when running `cargo` (or add `tracing`
// to default features in `Cargo.toml`)
#[tracing::instrument]
fn some_other_function() {
    // ...
}
```

---

TITLE: Installing Protoc from Source (Shell)
DESCRIPTION: This sequence of commands downloads, extracts, and installs a specific version of `protoc` (Protocol Buffers compiler) from GitHub releases. It sets up the `PATH` environment variable to make `protoc` accessible and verifies the installation, which is crucial for compiling protobuf definitions.
SOURCE: https://github.com/qdrant/qdrant/blob/master/docs/DEVELOPMENT.md#_snippet_6

LANGUAGE: shell
CODE:

```
PROTOC_VERSION=22.2
PKG_NAME=$(uname -s | awk '{print ($1 == "Darwin") ? "osx-universal_binary" : (($1 == "Linux") ? "linux-x86_64" : "")}')

# curl `proto` source file
curl -LO https://github.com/protocolbuffers/protobuf/releases//download/v$PROTOC_VERSION/protoc-$PROTOC_VERSION-$PKG_NAME.zip

unzip protoc-$PROTOC_VERSION-$PKG_NAME.zip -d $HOME/.local

export PATH="$PATH:$HOME/.local/bin"

# remove source file if not needed
rm protoc-$PROTOC_VERSION-$PKG_NAME.zip

# check installed `protoc` version
protoc --version
```
