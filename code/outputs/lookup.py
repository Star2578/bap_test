import pandas as pd

# แก้ชื่อไฟล์ตามจริง
df = pd.read_csv("bap_ollama_run_20251005T192015Z.csv")

# แปลง duration เป็นตัวเลข (กันกรณีเป็นสตริง/มีค่าว่าง)
df["total_duration_ms"] = pd.to_numeric(df["total_duration_ms"], errors="coerce").fillna(0)

# group by example_id
out = (
    df[["prompt", "total_duration_ms"]]
)

out.to_csv("data.grouped.csv", index=False)
print(out.head())

