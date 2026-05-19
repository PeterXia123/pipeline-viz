"""多输入、多输出（由解析器从源码识别）。"""

import pandas as pd


def run() -> None:
    a = pd.read_csv("data/in_a.csv")
    b = pd.read_csv("data/in_b.csv")
    m = pd.concat([a, b], ignore_index=True)
    m.to_csv("data/out_1.csv", index=False)
    m.to_csv("data/out_2.csv", index=False)
