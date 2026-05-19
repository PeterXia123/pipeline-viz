"""扇出：一路 staging → 两路 out（摘要 + 明细）。"""

import pandas as pd


def split_outputs() -> None:
    df = pd.read_csv("data/staging/merged_orders.csv")
    df.head(8).to_csv("data/out/summary.csv", index=False)
    df.to_csv("data/out/detail_wide.csv", index=False)
