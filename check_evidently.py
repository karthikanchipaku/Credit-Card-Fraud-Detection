from evidently.core.report import Report
from evidently.presets import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
# Print every single attribute/method that doesn't start with underscore
available_methods = [m for m in dir(report) if not m.startswith('_')]
print(f"Available methods on your Report object: {available_methods}")