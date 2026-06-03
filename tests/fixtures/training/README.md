# Training test fixture

`test_sequences.parquet` (a small real protein-sequence parquet with
`sequence_id` / `sequence` columns) is **not committed** — it is large data and
is git-ignored. Place it here to run the data and pilot-training tests; they
`pytest.skip` automatically when it is absent.
