#!/bin/bash

version=0.1
docker build -f Dockerfile.gpu . -t guidance_server_gpu:${version}