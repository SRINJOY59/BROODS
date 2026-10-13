#!/usr/bin/env bash
# Build one virtual environment per system under .venvs/.
# Run from the repository root:  bash scripts/setup_envs.sh
# Environments are separate because they pin incompatible
# versions of openai, pydantic and langchain-core.
set -e

# venv puts the interpreter in Scripts/ on Windows and bin/
# elsewhere; resolve it rather than hardcoding either.
vpy() { [ -x ".venvs/$1/Scripts/python.exe" ] \
        && echo ".venvs/$1/Scripts/python.exe" \
        || echo ".venvs/$1/bin/python"; }

python -m venv .venvs/baseline
"$(vpy baseline)" -m pip install -q -r requirements/baseline.txt

python -m venv .venvs/langgraph
"$(vpy langgraph)" -m pip install -q -r requirements/langgraph.txt

python -m venv .venvs/crewai
"$(vpy crewai)" -m pip install -q -r requirements/crewai.txt

python -m venv .venvs/autogen
"$(vpy autogen)" -m pip install -q -r requirements/autogen.txt

python -m venv .venvs/pydantic_ai
"$(vpy pydantic_ai)" -m pip install -q -r requirements/pydantic_ai.txt

python -m venv .venvs/openai_agents
"$(vpy openai_agents)" -m pip install -q -r requirements/openai_agents.txt

python -m venv .venvs/llamaindex
"$(vpy llamaindex)" -m pip install -q -r requirements/llamaindex.txt

python -m venv .venvs/smolagents
"$(vpy smolagents)" -m pip install -q -r requirements/smolagents.txt
