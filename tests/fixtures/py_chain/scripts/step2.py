"""输入为 step1 的输出 data/mid.csv。"""

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/mid.csv")
    df["z"] = 1
    df.to_csv("data/final.csv", index=False)
