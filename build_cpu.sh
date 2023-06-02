#!/bin/bash

version=$(cat version.txt)
docker build -f Dockerfile.cpu . -t guidance_server:cpu-${version}