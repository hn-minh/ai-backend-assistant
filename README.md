
# AI Backend Assistant

An end-to-end LLM-powered development assistant designed to support backend engineering with code and
terminal command suggestions, custom model fine-tuning and monitoring.

## Key Features

*   **Custom Fine-Tuned Models:** Includes pipelines for Supervised Fine-Tuning (SFT) using Unsloth and LoRA adapters, specifically tailored for code generation (`code-suggester`) and terminal commands (`bash-suggester`).
*   **Automated Data Pipeline:** Scripts to crawl GitHub repositories and generate Open-Source Software (OSS) instruction datasets for model training.
*   **High-Performance Inference:** Utilizes `vLLM` for fast, optimized LLM serving with support for dynamic LoRA adapters.
*   **Comprehensive Benchmarking:** Built-in benchmarking tools to evaluate serving performance, prefix caching efficiency, and response quality (LLM Judge).
*   **Production-Ready Observability:** A fully containerized monitoring stack integrating Prometheus, Loki, Promtail, and Grafana for real-time metrics and log analysis.

## Project Architecture

The repository is structured into distinct, modular components:

*   **`code-suggester/` & `bash-suggester/`**: Contains data crawling scripts, SFT notebooks (Baseline vs. Best Params), model configurations, and evaluation reports.
*   **`src/backend/`**: The core API application (FastAPI-based) handling chat routing, LLM integration, caching, and prompts.
*   **`src/vllm_api/`**: Dockerized vLLM serving infrastructure with adapter management scripts.
*   **`src/monitor/`**: The observability stack containing configurations for Grafana dashboards, Prometheus, and Loki.
*   **`src/benchmarks/`**: Scripts to load-test the API and evaluate LLM request functions.

## Getting Started

### Prerequisites
*   [Docker](https://www.docker.com/) and Docker Compose
*   [uv](https://docs.astral.sh/uv/) (Python package manager, recommended for `uv.lock`)

1. Go to `src` directory:
	```bash
	cd src
	```
2. Copy the example env files and fill in secrets where needed:

   ```bash
   cp backend/.env.example backend/.env
   cp vllm_api/.env.example vllm_api/.env
   cp monitor/.env.example monitor/.env
   ```

3. The system is fully containerized using Docker Compose. To spin up the entire ecosystem (Backend API, vLLM Engine, and Monitoring Stack), a unified runner script is provided:

```bash
# Execute the main run script
bash src/run.sh up
```
## Accessing Services

- **vLLM API**: `http://$VLLM_HOST:$VLLM_PORT`
- **FastAPI Gateway**: `http://$BACKEND_HOST:$BACKEND_PORT`
- **Grafana**: `http://$BACKEND_HOST:3000`
- **Prometheus**: `http://$BACKEND_HOST:9090`
- **Loki**: `http://$BACKEND_HOST:3100`


## Benchmarking

Benchmarks measure serving throughput and latency using a ShareGPT conversation dataset.

### Download the dataset

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
     -P benchmarks/
```

### Online serving throughput

Runs concurrent requests against the gateway (or any OpenAI-compatible endpoint) and reports request throughput, output throughput, TTFT, TPOT, and ITL percentiles.

```bash
export OPENAI_API_KEY=<your gateway api key>
make bench_serving
```

### Prefix caching

Measures the KV cache hit rate and latency improvement when vLLM prefix caching is enabled.

```bash
make benchmark_prefix_caching
```

Results are written to `benchmarks/results/` as JSON files.
