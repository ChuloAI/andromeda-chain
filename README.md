# Andromeda-Chain - mastering all the chains

This repository contains both a server and a client package.

The server is (not very creatively) called `guidance_server`.
The client is called `AndromedaChain`.

Why this package/library?
The [guidance](https://github.com/microsoft/guidance) is an awesome library, but has limited support through HTTP APIs like Oobabooga UI.
So I rolled my own HTTP server, which allows me to quickly prototype apps that use guidance templates.

I originally created Oasis with a variant of this idea: https://github.com/paolorechia/oasis


## Installation


### AndromedaChain

```bash
pip install andromeda-chain
``` 

### Guidance Server
Serving the guidance library with HuggingFace models loaded either with 4 bit quantization on GPU or on CPU (no quantization), behind a HTTP server.

#### Docker Container


Requirements:
1. docker-engine
2. docker-compose v2

If using GPU also:

3. nvidia-docker: https://github.com/NVIDIA/nvidia-docker



##### Pulling the image
You can find the images tags in [Docker Hub](https://hub.docker.com/repository/docker/paolorechia/guidance_server/general)
The easiest way to pull it is to reuse the docker-compose file.

```bash
docker-compose -f docker-compose.gpu.yaml up
```

Or use the CPU version

```bash
docker-compose -f docker-compose.cpu.yaml up
```

Note that you still need to setup the model (see in usage section).

*LICENSE NOTE*: The GPU image is based on `nvidia/cuda:12.1.1-runtime-ubuntu22.04`, which is subject to the proprietary NVIDIA license.
While the software from Andromeda repository is open source, some layers of the docker container are not.


#### Building
Just use the appropriate bash script
```bash
./build_gpu.sh
```

Or:
```bash
./build_cpu.sh
```


## Usage

1. Download a LLM model you want to use from Hugging Face.
2. Create a 'models' directory locally, and save the model in there.
3. Setup the environment variable `MODEL_PATH` in the `docker-compose.gpu` or `docker-compose.cpu` depending which one you want.
4. Start the server.
5. Use the Andromeda package to query the server.



### Using Andromeda Package

```python
from andromeda_chain import AndromedaChain, AndromedaPrompt, AndromedaResponse

chain = AndromedaChain()

prompt = AndromedaPrompt(
    name="hello",
    prompt_template="""Howdy: {{gen 'expert_names' temperature=0 max_tokens=300}}""",
    input_vars=[],
    output_vars=["expert_names"]
)

response: AndromedaResponse = chain.run_guidance_prompt(prompt)
# Use the response
print(response.expanded_generation)
print(response.result_vars)
```