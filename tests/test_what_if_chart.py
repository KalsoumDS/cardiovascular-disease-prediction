import ast
from pathlib import Path


def test_what_if_risk_gauge_has_unique_streamlit_key():
    app_tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))

    for node in ast.walk(app_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "plotly_chart" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "st" or not node.args:
            continue

        chart = node.args[0]
        if (
            isinstance(chart, ast.Call)
            and isinstance(chart.func, ast.Name)
            and chart.func.id == "plot_risk_gauge"
            and chart.args
            and isinstance(chart.args[0], ast.Name)
            and chart.args[0].id == "new_prob"
        ):
            key = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
            assert isinstance(key, ast.Constant)
            assert key.value == "what_if_risk_gauge"
            return

    raise AssertionError("What-If risk gauge chart call was not found")
