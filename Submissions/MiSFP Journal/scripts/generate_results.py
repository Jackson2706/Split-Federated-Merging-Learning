#!/usr/bin/env python3
"""Validate long-format MiSFP records and emit an auditable completion report."""
from pathlib import Path
import csv, json, math, statistics, sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"results/manifest.json").read_text())
rows=list(csv.DictReader((ROOT/"results/measurements.csv").open()))
required=manifest["measurement_columns"]; errors=[]; seen=set(); groups={}
for line,r in enumerate(rows,2):
    missing=[k for k in required if k not in r]
    if missing: errors.append(f"line {line}: missing columns {missing}"); continue
    key=tuple(r[k] for k in manifest["unique_key"])
    if key in seen: errors.append(f"line {line}: duplicate key {key}")
    seen.add(key)
    if r["seed"] and int(r["seed"]) not in manifest["seeds"]: errors.append(f"line {line}: unregistered seed")
    if r["value"]:
        try: value=float(r["value"]); assert math.isfinite(value)
        except Exception: errors.append(f"line {line}: value is not finite"); continue
        g=tuple(r[k] for k in manifest["aggregate_key"])
        groups.setdefault(g,[]).append((int(r["seed"]),value,r["unit"],r["config_hash"]))
summary=[]
for g,vals in groups.items():
    units={v[2] for v in vals}; cfgs={v[3] for v in vals}; seeds={v[0] for v in vals}
    if len(units)!=1: errors.append(f"group {g}: mixed units {units}")
    if len(cfgs)!=1: errors.append(f"group {g}: incomparable config hashes {cfgs}")
    numbers=[v[1] for v in vals]
    summary.append({**dict(zip(manifest["aggregate_key"],g)),"mean":statistics.mean(numbers),
      "sd":statistics.stdev(numbers) if len(numbers)>1 else None,"completed":len(seeds),"expected":len(manifest["seeds"]),"unit":next(iter(units))})
out={"valid":not errors,"errors":errors,"rows":len(rows),"aggregates":summary}
(ROOT/"generated"/"validation_report.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"valid":not errors,"rows":len(rows),"aggregates":len(summary),"errors":errors},indent=2))
sys.exit(1 if errors else 0)
