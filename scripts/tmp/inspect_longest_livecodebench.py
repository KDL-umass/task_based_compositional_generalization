import argparse
import os
import pickle

from init import ROOT_DIR
from src.models.pretrained import load_granite_2b


def load_samples(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_example_text(sample):
    code = sample.get("code", "").rstrip()
    inp = sample.get("input", "")
    out = sample.get("output", "")
    return f"{code}\nassert {inp} == {out}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(
            ROOT_DIR,
            "data",
            "livecodebench",
            "holdout_96",
            "seed_0",
            "test.pkl",
        ),
    )
    args = parser.parse_args()

    print(f"Loading samples from: {args.data_path}")
    samples = load_samples(args.data_path)
    model, tokenizer, _ = load_granite_2b()
    _ = model  # model loaded to match eval env, not used further

    longest = None
    longest_tokens = -1
    for sample in samples:
        text = build_example_text(sample)
        token_count = len(tokenizer.encode(text))
        if token_count > longest_tokens:
            longest_tokens = token_count
            longest = sample

    if longest is None:
        print("No samples found.")
        return

    example_text = build_example_text(longest)
    print(f"Longest token count: {longest_tokens}")
    print("Example input:")
    print(longest.get("input", ""))
    print("Example output:")
    print(longest.get("output", ""))
    print("Full code + assert:")
    print(example_text)


if __name__ == "__main__":
    main()
