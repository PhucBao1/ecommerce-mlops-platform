import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ==========================================
# LOAD DATA
# ==========================================

reference_df = pd.read_parquet("monitoring/evidently/reference/reference.parquet")

current_df = pd.read_parquet("monitoring/evidently/reference/valid_df.parquet")

# ==========================================
# REPORT
# ==========================================

report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])

my_eval = report.run(reference_data=reference_df, current_data=current_df)

# ==========================================
# SAVE HTML
# ==========================================

my_eval.save_html("monitoring/evidently/reports/drift_report.html")

print("Report generated!")
