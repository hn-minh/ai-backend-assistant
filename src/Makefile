CONCURRENCIES ?= 1 2 4 8 16 32 64 128

bench_serving:
	@for concurrency in $(CONCURRENCIES); do \
		echo "=== concurrency = $$concurrency ==="; \
		python benchmarks/benchmark_serving.py \
			--model routing-classifier \
			--tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
			--endpoint /v1/completions \
			--dataset-name sharegpt \
			--dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
			--request-rate inf \
			--max-concurrency $$concurrency \
			--lora-modules backend-lora bash-lora \
			--percentile-metrics ttft,tpot,e2el \
			--metric-percentiles 95,99 \
			--result-dir ./benchmarks/results \
			--save-result \
			--save-detailed \
			--metadata concurrency=$$concurrency || exit 1; \
	done


bench_prefix_caching:
	python benchmarks/benchmark_prefix_caching.py \
		--model Qwen/Qwen2.5-Coder-7B-Instruct \
		--tokenizer Qwen/Qwen2.5-Coder-7B-Instruct \
		--dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
		--enable-prefix-caching \
		--num-prompts 20 \
		--repeat-count 5 \
		--input-length-range 128:256 | tee benchmarks/results/prefix_caching_result.txt
