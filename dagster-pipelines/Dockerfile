FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY feast/ ./feast/
COPY contracts/ ./contracts/
COPY checks/ ./checks/

# Dagster user code gRPC server
EXPOSE 3030
CMD ["/app/.venv/bin/dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "3030", "-m", "src.definitions"]
