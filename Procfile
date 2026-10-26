# For platforms that read a Procfile instead of a blueprint. The wildcard bind
# is a deployment decision stated here, not an application default -- see
# deploy/render.yaml for why.
web: uvicorn rse.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
