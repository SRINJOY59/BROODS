#!/usr/bin/env bash
# Regenerate every measurement behind the paper. Run from the repository root
# after scripts/setup_envs.sh. Nothing here calls a model provider, and a full
# run takes a few hours, most of it the depth-3 spawn series and CrewAI.
set -e

vpy() { [ -x ".venvs/$1/Scripts/python.exe" ] \
        && echo ".venvs/$1/Scripts/python.exe" \
        || echo ".venvs/$1/bin/python"; }

echo "== Tables 2 and 3: the full matrix, three trials"
python -m broods.run_all --trials 3

echo "== Table 3: the depth-3 growth series"
python -m broods.runner --probe spawn --depths 3 --trials 3

echo "== Table 4: native delegation constructs"
"$(vpy openai_agents)" broods/native/openai_handoff.py
"$(vpy llamaindex)"    broods/native/llamaindex_workflow.py
"$(vpy langgraph)"     broods/native/langgraph_subgraph_loops.py
"$(vpy crewai)"        broods/native/crewai_hier.py
"$(vpy smolagents)"    broods/native/smolagents_managed.py
"$(vpy autogen)"       broods/native/autogen_team.py
"$(vpy pydantic_ai)"   broods/native/pydantic_scope.py

echo "== Table 6: capability audit behind every L0"
for fw in langgraph crewai autogen pydantic_ai openai_agents smolagents llamaindex; do
    "$(vpy "$fw")" broods/audit_caps.py "$fw"
done

echo "== Table 8: TaskScope on all eight systems"
python -m broods.experiments.scope_eval --trials 3

echo "== Growth-law check and the matrix from the newest bundle"
python -m broods.analyze --bundle latest --holdout
python -m broods.table $(ls results/full/matrix_*.json | tail -1)
