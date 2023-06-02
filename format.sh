#!/bin/bash
isort guidance_server
isort src/andromeda_chain
black guidance_server
black src/andromeda_chain