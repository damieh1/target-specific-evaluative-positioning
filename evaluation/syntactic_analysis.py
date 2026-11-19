"""Link parsed targets to final orientations and count governing forms."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from descriptive_analysis import LABELS, TARGETS, read_measurement

ROLES = ["subject", "passive_subject", "object", "oblique"]
# Keep the notebook's exact stopword list, including by.
PREDICATE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "by", "as", "is", "are", "was", "were", "be", "been", "being", "am", "have", "has",
    "had", "do", "does", "did", "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "we", "you", "i", "root",
}


def map_syntactic_role(dep):
    if pd.isna(dep):
        return "unknown"
    dep = str(dep).strip().lower()
    if dep in {"nsubj:pass", "nsubjpass", "csubj:pass", "csubjpass"}:
        return "passive_subject"
    if dep in {"nsubj", "csubj"}:
        return "subject"
    if dep in {"obj", "dobj", "iobj"}:
        return "object"
    if dep in {"obl", "pobj"} or dep.startswith("obl:"):
        return "oblique"
    if dep in {"poss", "nmod:poss"}:
        return "possessor"
    if dep in {"amod", "compound", "nmod", "appos"} or dep.startswith("nmod:"):
        return "modifier"
    return "other"


def plot_predicates(counts, output_dir):
    positions = ["Unfavorable", "Neutral / Descriptive", "Favorable"]
    colors = ["#D55E00", "#999999", "#0072B2"]
    fig, axes = plt.subplots(len(TARGETS), len(ROLES), figsize=(14, 6), sharex=False)
    for i, target in enumerate(TARGETS):
        for j, role in enumerate(ROLES):
            ax = axes[i, j]
            data = counts.loc[counts["core_family"].eq(target) & counts["syntactic_role"].eq(role)]
            if data.empty:
                ax.axis("off")
                continue
            # Select ten forms by total frequency, then show their positioning distributions.
            top = data.groupby("predicate_clean")["n"].sum().nlargest(10).index
            pivot = data.loc[data["predicate_clean"].isin(top)].pivot_table(
                index="predicate_clean", columns="positioning", values="n", fill_value=0)
            pivot = pivot.reindex(columns=positions, fill_value=0)
            pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
            pivot.plot(kind="barh", stacked=True, ax=ax, color=colors, legend=False, width=0.65)
            ax.set_title(role.replace("_", " ").title() if i == 0 else "", fontsize=10, fontweight="bold")
            ax.set_ylabel(target if j == 0 else "", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            sns.despine(ax=ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], positions,
               loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    fig.tight_layout(pad=1.0, h_pad=1.2, w_pad=1.0)
    fig.savefig(output_dir / "plot.pdf", bbox_inches="tight")
    plt.close(fig)


def run(measurement, parses, output_dir):
    df = read_measurement(measurement)
    parsed = pd.read_csv(parses, dtype={"occurrence_id": "string"}, low_memory=False)
    target_parse = parsed.loc[parsed["core_family"].isin(TARGETS)].copy()
    target_parse["syntactic_role"] = target_parse["dep_rel"].apply(map_syntactic_role)
    # Use a lemma if present; otherwise count surface forms without lemmatizing.
    predicate_source = next((c for c in ["head_lemma", "predicate_lemma", "governor_lemma"] if c in target_parse), "head_form")
    target_parse["predicate_lemma"] = target_parse[predicate_source]
    print("Governing form column:", predicate_source)
    eval_keep = df[["occurrence_id", "final_orientation"]]
    merged = target_parse.merge(eval_keep, on="occurrence_id", how="left", validate="one_to_one", indicator=True)
    print("Merge status:", merged["_merge"].value_counts().to_dict())
    matched = merged.loc[merged["final_orientation"].isin(LABELS)].copy()
    matched["positioning"] = matched["final_orientation"]
    columns = ["occurrence_id", "core_family", "syntactic_role", "predicate_lemma", "positioning",
               "dep_rel", "head_form", "annotation_target", "focal_sentence_original"]
    core = matched[columns].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    core.to_csv(output_dir / "pred_freq_parsed.csv", index=False)

    groups = ["core_family", "positioning", "syntactic_role"]
    overview = core.groupby(groups, dropna=False).size().reset_index(name="n")
    overview["group_total"] = overview.groupby(groups[:2])["n"].transform("sum")
    overview["pct_within_family_position"] = (100 * overview["n"] / overview["group_total"]).round(2)
    overview.sort_values([*groups[:2], "n"], ascending=[True, True, False]).to_csv(
        output_dir / "core_analysis_freqs.csv", index=False)

    lexical = core.copy()
    forms = lexical["predicate_lemma"].astype("string").str.lower().str.strip()
    valid = forms.str.contains(r"[a-zA-Z]", regex=True, na=False)
    stopword = forms.isin(PREDICATE_STOPWORDS)
    # Report exclusions so we can compare this filter with the manuscript's retention count.
    forms[valid & stopword].value_counts().to_csv(output_dir / "excluded_stopwords.csv", header=["n"])
    lexical["predicate_clean"] = forms
    lexical = lexical.loc[valid & ~stopword].copy()
    counts = lexical.groupby(groups + ["predicate_clean"], dropna=False).size().reset_index(name="n")
    counts["role_total"] = counts.groupby(groups)["n"].transform("sum")
    counts["pct_within_role"] = (100 * counts["n"] / counts["role_total"]).round(2)
    counts = counts.sort_values(groups + ["n", "predicate_clean"], ascending=[True, True, True, False, True])
    counts.to_csv(output_dir / "governing_form_counts.csv", index=False)
    counts.groupby(groups, group_keys=False).head(10).to_csv(output_dir / "top_governing_forms.csv", index=False)
    # Preserve the rows used to select qualitative examples, locally only.
    lexical.to_csv(output_dir / "filtered_occurrences.csv", index=False)
    summary = {"predicate_source": predicate_source, "matched_n": len(core), "retained_n": len(lexical),
               "excluded_stopwords_n": int((valid & stopword).sum()), "excluded_nonlexical_n": int((~valid).sum()),
               "excluded_by_n": int((forms.eq("by").fillna(False) & valid).sum())}
    (output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    plot_predicates(counts, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--parses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.measurement, args.parses, args.output_dir)


if __name__ == "__main__":
    main()
