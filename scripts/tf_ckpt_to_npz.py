#!/usr/bin/env python3
import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export legacy TensorFlow checkpoint to NPZ"
    )
    parser.add_argument("--ckpt", required=True, help="Input TF checkpoint path")
    parser.add_argument("--out", required=True, help="Output NPZ path")
    args = parser.parse_args()

    import tensorflow as tf

    reader = tf.compat.v1.train.NewCheckpointReader(args.ckpt)
    variable_map = reader.get_variable_to_shape_map()
    tensors = {name: reader.get_tensor(name) for name in sorted(variable_map.keys())}

    np.savez(args.out, **tensors)
    print(f"exported {len(tensors)} tensors -> {args.out}")
    for name in sorted(variable_map.keys()):
        print(name, variable_map[name])


if __name__ == "__main__":
    main()
