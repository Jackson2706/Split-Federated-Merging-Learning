"""
Minimal booktabs LaTeX table generation for the camera-ready paper.

table(rows, columns, ...) -> str. Each `column` is (key, header). Cell values are
taken verbatim from each row dict (already formatted, e.g. via io_utils.fmt_mean_std).
"""


def _escape(s):
    s = str(s)
    # Leave math ($...$) and already-escaped sequences mostly alone; escape % and _.
    return s.replace("%", r"\%").replace("_", r"\_")


def table(rows, columns, caption="", label="", float_fmt="{:.4f}",
          escape=True, column_spec=None):
    """Build a booktabs LaTeX table string.

    Args:
        rows: list of dicts.
        columns: list of (key, header) tuples.
        caption, label: LaTeX caption/label text.
        float_fmt: format applied to float cell values.
        escape: escape %/_ in non-math text cells.
        column_spec: e.g. "lcccc". Defaults to first col 'l', rest 'c'.
    """
    keys = [k for k, _ in columns]
    headers = [h for _, h in columns]
    if column_spec is None:
        column_spec = "l" + "c" * (len(columns) - 1)

    def cell(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return _escape(v) if escape else str(v)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if caption:
        lines.append(r"\caption{%s}" % caption)
    if label:
        lines.append(r"\label{%s}" % label)
    lines.append(r"\begin{tabular}{%s}" % column_spec)
    lines.append(r"\toprule")
    lines.append(" & ".join(_escape(h) if escape else str(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(cell(r.get(k, "")) for k in keys) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_table(rows, columns, path, **kwargs):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    s = table(rows, columns, **kwargs)
    with open(path, "w") as f:
        f.write(s + "\n")
    return path
