"""Sample a small batch of vessels from the GenerationConfig defaults."""

from pathlib import Path

from vesselgen.config import GenerationConfig
from vesselgen.library import generate_dataset

OUT = Path("out/example_batch")


def main() -> None:
    cfg = GenerationConfig(
        side_branch_probability=0.5,
    )
    written = generate_dataset(n=8, out_dir=OUT, config=cfg, base_seed=42)
    print(f"Wrote {len(written)} vessels to {OUT}")


if __name__ == "__main__":
    main()
