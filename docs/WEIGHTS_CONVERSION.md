# Weight Conversion

This project supports converting legacy TensorFlow checkpoints (`.ckpt`) into Torch checkpoints (`.ot`).

## Prerequisites

- Python 3.12 venv with `tensorflow-macos` and `numpy`
- OCaml build with `convert_legacy_ckpt.exe`

## One checkpoint

```bash
.venv-convert312/bin/python scripts/tf_ckpt_to_npz.py \
  --ckpt models/legacy_gcs/large.5000.ckpt \
  --out  models/legacy_gcs/large.5000.npz

opam exec -- dune exec ./convert_legacy_ckpt.exe -- \
  -npz models/legacy_gcs/large.5000.npz \
  -out models/converted/large.5000.ot \
  -activation Relu
```

## Batch conversion

```bash
for ckpt in models/legacy_gcs/*.ckpt; do
  npz="${ckpt%.ckpt}.npz"
  base="$(basename "${ckpt%.ckpt}")"

  .venv-convert312/bin/python scripts/tf_ckpt_to_npz.py --ckpt "$ckpt" --out "$npz"

  act="Sigmoid"
  case "$base" in
    large.*) act="Relu" ;;
    medium.*) act="Sigmoid" ;;
  esac

  opam exec -- dune exec ./convert_legacy_ckpt.exe -- \
    -npz "$npz" \
    -out "models/converted/${base}.ot" \
    -activation "$act"
done
```

## Recommended activation by family

- `large.*` -> `Relu`
- `medium.*` -> `Sigmoid`
- `small.*`, `self.*`, `hybrid.*`, `pcr.*` -> `Sigmoid`

Always pass `-activation` explicitly.
