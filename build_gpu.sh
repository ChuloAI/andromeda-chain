#!/bin/bash

version=$(cat version.txt)
docker build -f Dockerfile.gpu . -t guidance_server:gpu-${version}