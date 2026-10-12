# Llama-3.1-8B-Instruct temperature comparison

Folders use actual temperatures: t0 (0), t0.7 (0.7), t1 (1). Each contains aut, dat, cdat, drat, with response scores, plots and creative_vs_controls.csv.

The t0 results consolidate the original greedy runs listed in temperature_zero_sources.json, including unchanged response CSVs, calibration files and scoring manifests. Each task also contains residual/, concept_mean/, concept_contrast/, concept_head/, plots/, effects.csv and determinism_audit.csv. The original greedy analysis folders were removed after verifying every migrated file against its SHA-256 hash; see greedy_results_migration.json. Behavioral files remain at the task folder root. They contain 60 identical repetitions per item/condition; these are determinism checks, not independent samples. Greedy contrasts are descriptive and have no sampling confidence intervals.

The t0.7 and t1 results contain behavioral generation only, with 60 sampled repetitions per item/condition. Residual and concept intervention runs were not performed at these temperatures. CDAT novelty must be interpreted alongside cdat_gates.csv. Plot error bars show +/- one standard deviation across responses. AUT relative novelty ICF uses clusters fitted within each original scoring run.

Raw sampled generation remains in outputs/temperature_comparison_8b_20261007T094311Z/t0 (temperature 0.7) and t1 (temperature 1), as originally submitted. This analysis layout uses explicit temperature names.

Mechanistic reruns: previous t0 intervention results, plots and effects are preserved under each task/previous_mechanistic_center/. New residual and concept jobs target the usual candidate directories at t0, t0.7 and t1. Composite suppression uses the corrected full Standard post-block residual center. Behavioral scores are retained.
