# Distil — distroless-style, stdlib-only runtime image.
# The core has zero runtime dependencies, so the image is tiny and the build
# is reproducible. `[live]` extras (Anthropic SDK) are opt-in at run time.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY distil ./distil
COPY corpus ./corpus
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim
LABEL org.opencontainers.image.title="distil" \
      org.opencontainers.image.description="Compression with a quality contract — cache-aware, causally-pruned context compression for agentic runtimes." \
      org.opencontainers.image.source="https://github.com/dshakes/distil" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /app
COPY --from=build /dist/*.whl /tmp/
COPY corpus ./corpus
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl
ENV DISTIL_CORPUS=/app/corpus
ENTRYPOINT ["distil"]
CMD ["bench"]
