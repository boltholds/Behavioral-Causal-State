# Frozen L1 public-text preparation

`experiments/l1/preparation.py` produces the existing `embedding-cache-v1` layout under train, validation, iid_test, and challenge. It reads only each split's public JSONL. Label fields and unsupported public metadata are rejected by the cache reader; no training labels or evaluator files are opened.

The CLI verifies the original SONAR model and tokenizer hashes through `SonarEncoder`, uses rus_Cyrl float32 CPU embeddings with no normalization or fitted preprocessing, and checks the pinned encoder identity. Exact utterance vectors are memoized globally while the per-split cache retains first-occurrence text order and the original record/message index mapping. Frozen feature extraction across public splits learns no parameters or statistics.

Run from the repository checkout in the isolated SONAR environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/scratch/ccdcdfc626df/sonar-env/bin/python experiments/l1/preparation.py --dataset artifacts/l1-v0.1-final --assets artifacts/sonar-assets --output artifacts/l1-embeddings-v0.1 --threads 2 --batch-size 16
```

All existing split caches are verified against public hashes, binary/record checksums, encoder identity, and public ordering before any new split is encoded. Verified completed splits seed exact-text memoization. A `.SPLIT.partial` directory is never silently reused: inspect and remove an abandoned partial directory explicitly before retrying. New splits are written to staging, loaded through the verifier, then renamed into their completed destinations. Corrupt completed caches fail closed.

Progress is emitted at most 20 seconds apart between batches. Batch prediction is synchronous; a single unusually slow batch can extend that interval. The completion summary includes split/source/cache hashes, records, unique sentence counts, actual newly encoded counts, library versions, batch size, threads, and elapsed time. Compact manifests are copied to `experiments/results/l1-preparation-v0.1`; binary caches and checkpoint assets remain excluded from git.

The separate `experiments/reviews/l1-catalog-review-request.json` and `.md` request renders 48 controlled-language forms and binds the pending human-review decision to the exact frozen catalog SHA256. Preparation does not invent or approve that review.
