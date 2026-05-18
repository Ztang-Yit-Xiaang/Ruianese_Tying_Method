# VLM Backend Comparison

| Backend | Status | Runtime sec | Rows | Hanzi rows | Review rows | Tone resolved | Tone unresolved | Invalid tone | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| paddleocr_vl | cached_existing |  | 22 | 19 | 22 | 0 | 22 | 0 | loaded_existing_rows_tsv |
| qwen_vl | blocked | 235.71 | 0 | 0 | 0 | 0 | 0 | 0 | CUDA out of memory. Tried to allocate 250.84 GiB. GPU 0 has a total capacity of 4.00 GiB of which 0 bytes is free. Of the allocated memory 4.51 GiB is allocated by PyTorch, and 934.13 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) See raw.json for traceback. |

## Validity Policy

- Tone number must be `1-8` or blank.
- Tone base must be `平`, `上`, `去`, `入`, or blank.
- Tone register must be `yin`, `yang`, `unknown`, or blank.
- Review rows are acceptable; confident invalid tone values are not.