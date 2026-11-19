"""Fit separate binomial models for favorable and unfavorable positioning."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from descriptive_analysis import read_measurement, TARGETS


def wald_test(model, interaction_only=False):
    names = list(model.params.index)
    if interaction_only:
        tested = [i for i, name in enumerate(names) if "C(core_family" in name and ":C(outlet" in name]
    else:
        tested = [i for i, name in enumerate(names) if "C(outlet" in name]
    restriction = np.eye(len(names))[tested]
    test = model.wald_test(restriction, scalar=True)
    return {"n_coefficients_tested": len(tested), "chi2": float(test.statistic),
            "df": int(test.df_denom), "p_value": float(test.pvalue)}


def run(measurement, output_dir):
    df = read_measurement(measurement)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["occurrence_id", "video_id", "outlet", "month", "core_family", "final_orientation"]
    analysis = df.loc[df["core_family"].isin(TARGETS), columns].dropna().copy()
    analysis["month"] = analysis["month"].astype(str)
    print(f"Instances: {len(analysis)}; videos: {analysis['video_id'].nunique()}; months: {analysis['month'].nunique()}")
    pd.crosstab(analysis["core_family"], analysis["outlet"]).to_csv(output_dir / "model_sample.csv")

    formula = ("{outcome} ~ C(core_family, Treatment(reference='Israel')) * "
               "C(outlet, Treatment(reference='bbcnews')) + C(month)")
    omnibus, interactions = {}, {}
    for outcome in ["unfavorable", "favorable"]:
        analysis[outcome] = analysis["final_orientation"].eq(outcome.capitalize()).astype(int)
        model = smf.glm(formula.format(outcome=outcome), data=analysis, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": analysis["video_id"], "use_correction": True})
        print(f"{outcome}: converged={model.converged}, n={int(model.nobs)}")
        pd.DataFrame({"coefficient": model.params, "clustered_se": model.bse}).to_csv(
            output_dir / f"{outcome}_coefficients.csv")
        omnibus[outcome] = wald_test(model)
        interactions[outcome] = wald_test(model, interaction_only=True)
    for name, results in [("outlet_omnibus_wald_tests", omnibus), ("outlet_target_interaction_tests", interactions)]:
        table = pd.DataFrame(results).T.rename_axis("outcome")
        table.to_csv(output_dir / f"{name}.csv")
        print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.measurement, args.output_dir)


if __name__ == "__main__":
    main()
