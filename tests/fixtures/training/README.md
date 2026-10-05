# Training test fixture

`test_sequences.parquet` (a small real protein-sequence parquet with
`sequence_id` / `sequence` columns) is **not committed** — it is large data and
is git-ignored. Place it here to run the data and pilot-training tests; they
`pytest.skip` automatically when it is absent.

`eval-25k_v2026-09-29.parquet` (25,000 real paired antibody rows: `sequence_aa:0/1`,
`cdr_mask_aa:0/1`, `nongermline_mask_aa:0/1`, `v_mutation_count_aa:0/1`, `donor`) is the
within-donor eval split of the lab's `v2026-09-29` release
(`s3://brineylab-eu/datasets/ablm-training-data/v2026-09-29/within-donor/`). Also not
committed; the collator real-data tests skip without it.
