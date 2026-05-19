"""产出 mid.csv，供 step2 消费。"""

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/raw.csv")
    df.to_csv("data/mid.csv", index=False)
