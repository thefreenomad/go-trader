# go-trader paper instance (containerized stable deployment).
# Multi-stage: build the Go scheduler, then a slim Python runtime with the
# check-script deps. The container = the always-running paper version; the
# working tree stays for development.

# ---- build the Go binary (pure-Go sqlite -> CGO off, static) ----
FROM golang:1.26.2-bookworm AS build
WORKDIR /src/scheduler
COPY scheduler/go.mod scheduler/go.sum ./
RUN go mod download
COPY scheduler/ ./
RUN CGO_ENABLED=0 go build -ldflags "-X main.Version=docker-paper" -o /go-trader .

# ---- runtime ----
FROM python:3.12-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# go-trader invokes ".venv/bin/python3"; HL check scripts need ccxt + HL SDK + numpy/pandas
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir ccxt hyperliquid-python-sdk numpy pandas
# Pin the HL SDK to the repo-locked version (0.23.x changed asset_to_sz_decimals
# keying + meta structure and breaks the adapter). Separate layer keeps the big
# pip layer above cached.
RUN /app/.venv/bin/pip install --no-cache-dir "hyperliquid-python-sdk==0.22.0"
# repo dirs (scripts self-resolve imports relative to their own location)
COPY platforms/ ./platforms/
COPY shared_scripts/ ./shared_scripts/
COPY shared_strategies/ ./shared_strategies/
COPY shared_tools/ ./shared_tools/
COPY scheduler/static/ ./scheduler/static/
COPY research/ ./research/
COPY --from=build /go-trader ./go-trader
COPY scheduler/config.docker.json ./scheduler/config.json
RUN mkdir -p /app/data/logs
ENV GO_TRADER_ALLOW_MISSING_STATE=1
EXPOSE 8099
CMD ["./go-trader", "--config", "scheduler/config.json"]
