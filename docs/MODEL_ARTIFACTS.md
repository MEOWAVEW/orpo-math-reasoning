# Recovered adapter downloads

Release: [v0.1.0-evidence](https://github.com/MEOWAVEW/orpo-math-reasoning/releases/tag/v0.1.0-evidence).

| File | Contents | ZIP SHA256 |
|---|---|---|
| [checkpoint-200](https://github.com/MEOWAVEW/orpo-math-reasoning/releases/download/v0.1.0-evidence/recovered-checkpoint-200.zip) | Recovered adapter, tokenizer export, provenance card and per-file hashes | `e4251c0451f8c94d89fce3c0921fb5dc7e259794390437d845308878ef2fab38` |
| [checkpoint-250](https://github.com/MEOWAVEW/orpo-math-reasoning/releases/download/v0.1.0-evidence/recovered-checkpoint-250.zip) | Recovered adapter, tokenizer export, provenance card and per-file hashes | `049e2ab54c2ebab6678a500bd1d2a00d0a5d6c0da4a3e5254228d2b7659913b8` |

The base model is not included. The archived Kaggle base reference is `arthurchariyasathian/qwen3-5-9b/pytorch/fp16/1`; its exact upstream identity/revision remains unverified. The recovered run used BF16 LoRA with 4-bit weight quantization disabled.

Neither adapter is claimed to reproduce the archived benchmark scores. Historical evaluator outputs did not record adapter hashes. These files were recovered and checked for byte integrity, not independently validated for model quality or compatibility.

Download both ZIPs, `artifact-manifest.json`, and `SHA256SUMS.txt` from the release into one directory. Verify before extraction:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Cross-platform Python verification:

```python
from pathlib import Path
import hashlib
for line in Path("SHA256SUMS.txt").read_text().splitlines():
    expected, name = line.split("  ", 1)
    h = hashlib.sha256()
    with Path(name).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {name}")
    print(f"OK: {name}")
```

Each ZIP contains `FILES_SHA256.json` covering its original exported files and `RECOVERED_MODEL_CARD.md` explaining provenance. The original adapter hashes are also in [the training weight manifest](../evidence/training/run250/weight_manifest.json). ZIP checksums differ from raw adapter checksums because ZIPs also contain tokenizer and documentation files.

The source tag remains fixed at commit `ac2a329475683be55de7a480a31358f91e2a0f3a`; current main includes these download instructions. Repository MIT licensing covers code; upstream model/tokenizer terms apply to their respective artifacts.
