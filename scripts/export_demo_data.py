from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.quant.market import generate_demo_market  # noqa: E402


def main() -> None:
    output = ROOT / "data" / "demo_market.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_market(seed=20260825)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()

