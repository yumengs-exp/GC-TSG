import pandas as pd

df = pd.read_parquet("./Dataset/Detection.parquet")

print("detections:", len(df))
print("time range:", df["date_no"].min(), "to", df["date_no"].max())
print("columns:", df.columns)
