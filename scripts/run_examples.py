"""Run every educational script in a disposable copy, with a headless backend.

No generated plots or caches are written into the checkout. --output-dir optionally
keeps logs/results outside the temporary directory (do not commit local raw logs).
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
from pathlib import Path
import contextlib,io,json,runpy,sys
import numpy as np
p=Path(sys.argv[1]);sys.path.insert(0,str(p.parent))
# Importing any example must not start experiments or print results.
with contextlib.redirect_stdout(io.StringIO()) as capture:
    runpy.run_path(str(p),run_name="review_import")
if capture.getvalue():raise AssertionError("Import triggered example output")
fit_inputs=[];main_locals={}
fit_names={"fit_linear_regression","fit_logistic_regression","fit_random_forest",
           "fit_bagging","fit_adaboost","fit_gradient_boosting",
           "fit_gaussian_naive_bayes_multifeature","fit_linear_svm","fit_pca"}
def profile(frame,event,arg):
    name=frame.f_code.co_name
    if event=="call" and (name in fit_names or (name=="build_tree" and frame.f_back.f_code.co_name=="main")
                          or (name=="predict" and p.stem=="knn_classification")):
        loc=frame.f_locals
        value=loc.get("X",loc.get("x",loc.get("X_train")))
        if isinstance(value,np.ndarray):fit_inputs.append((name,value.copy()))
    if event=="return" and name=="main" and frame.f_globals.get("__file__")==str(p):
        main_locals.update(frame.f_locals)
sys.setprofile(profile)
try:runpy.run_path(str(p),run_name="__main__")
finally:sys.setprofile(None)
audit={"import_quiet":True,"split_check":"not applicable","train_only_fit_calls":0}
if all(k in main_locals for k in ("train_indices","validation_indices","test_indices")):
    arrays=[main_locals[k] for k in ("train_indices","validation_indices","test_indices")]
    all_indices=np.concatenate(arrays)
    assert len(np.unique(all_indices))==len(all_indices),"Overlapping split indices"
    np.testing.assert_array_equal(np.sort(all_indices),np.arange(len(all_indices)))
    audit["split_check"]=[len(a) for a in arrays]
    # Main experiments must fit only their training arrays. Learning demos have no split.
    allowed=[main_locals[k] for k in ("X_train","x_train","x_reg_train","X_train_scaled") if k in main_locals]
    for name,value in fit_inputs:
        assert any(value.shape==a.shape and np.array_equal(value,a) for a in allowed),name+" fitted non-Train input"
    audit["train_only_fit_calls"]=len(fit_inputs)
    if "dtrain" in main_locals:
        np.testing.assert_array_equal(main_locals["dtrain"].get_label(),np.asarray(main_locals["y_train"],dtype=np.float32))
        np.testing.assert_allclose(main_locals["dtrain"].get_data().toarray(),main_locals["X_train_xgb"],rtol=1e-6)
        audit["xgboost_train_matrix_check"]=True
print("REVIEW_AUDIT "+json.dumps(audit))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="ml-examples-") as temporary:
        base = Path(temporary)
        snapshot = base / "repository"
        shutil.copytree(ROOT, snapshot, ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", "*.bak", ".DS_Store",
            ".env", ".env.*", ".pytest_cache", ".review-local",
        ))
        env = os.environ.copy()
        env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(base / "mpl"),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for script in sorted(snapshot.glob("[0-9][0-9]_*/*.py")):
            if script.name == "xgboost_lightgbm_catboost.py" and not args.include_xgboost:
                continue
            relative = str(script.relative_to(snapshot))
            start = time.monotonic()
            try:
                process = subprocess.run(
                    [sys.executable, "-c", WORKER, str(script)], cwd=base,
                    env=env, text=True, capture_output=True, timeout=240,
                )
                output = process.stdout + process.stderr
                record = {"script": relative, "exit": process.returncode,
                          "seconds": round(time.monotonic() - start, 2)}
                audits = [line for line in process.stdout.splitlines() if line.startswith("REVIEW_AUDIT ")]
                if audits:
                    record["audit"] = json.loads(audits[-1][len("REVIEW_AUDIT "):])
            except subprocess.TimeoutExpired:
                record = {"script": relative, "exit": "timeout"}
                output = "Timed out after 240 seconds"
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.output_dir:
                (args.output_dir / (script.stem + ".log")).write_text(output)
            if record["exit"] != 0:
                print(output[-6000:], file=sys.stderr)
        if args.output_dir:
            (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
    print(f"Passed {sum(r['exit'] == 0 for r in records)}/{len(records)} scripts")
    return int(any(r["exit"] != 0 for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
