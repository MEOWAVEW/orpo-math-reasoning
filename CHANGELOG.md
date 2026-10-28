# Changelog

## 0.1.0-evidence — 2026-10-28

- Correct the completed training claim to the recovered 250-step BF16 LoRA run; distinguish optional QLoRA code from observed experiments.
- Archive raw predictions, evaluator snapshots, training metadata, logs, dataset revisions, and SHA256 manifests.
- Document historical contamination, missing checkpoint attribution, and absent baseline comparisons.
- Add a deterministic text-deduplicated split and frozen question/gold evaluation inputs for new runs.
- Add a separate strict numeric evaluation protocol with a true base-model branch, explicit parse outcomes, and per-run provenance.
- Add read-only archive verification, 12 offline regression tests, and GitHub Actions checks.

No new training or model inference results are claimed in this version. Recovered adapter ZIPs, checksums, and provenance cards are now provided as Release assets. The About text describes evidence and tooling without unverified performance claims.
