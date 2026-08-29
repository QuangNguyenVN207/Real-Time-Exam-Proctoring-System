# Stage 6 grouped-OOF 0.8469 technical note

## Status

- Highest verified result: grouped-OOF actor macro-F1 `0.8469938154473259`.
- Preserved 30 FPS reference: `0.826448522100696`; absolute gain `0.0205452933466299`.
- Human dropped target `0.85`; Stage 6 acceptance remains `>0.8`.
- Final mixed-profile Stage 6 bundle: `tmp/causal_8fps_stage6_mixed_084699_final_20260827`.
- Gate 3 evidence is complete. Gate 4 runtime compatibility is explicitly deferred.
- Locked test was not scored or tuned.
- Best evidence: `tmp/causal_8fps_stage6_mix_multistart_20260827.json`, SHA256 `20348D22CB6A9FC12621687D2C9F268872C61B1405D980F7F65CE24D4E3F6F3F`.

## Code changes

- `behavior_subset_stage2.py:1551-1554`: retain `sample_index`, `split`, and `split_group` in causal prefix rows.
- `behavior_subset_stage2.py:2120`: group replay snapshots by `sample_index`, never provenance-only `source_frame_index`.
- `causal_stream.py:274-278`: reject C2 pair propagation when either endpoint lacks current-frame scores; prevents absent-peer `NaN` evidence.
- `stage6_bundle.py:1-113`: new strict bundle loader, SHA256 validation, schema/policy mismatch rejection, and saved-fold OOF reproduction CLI.
- `stage6_retrain.py:28-69`: Stage 6 format, labels, specialist set, fixed model-profile candidates.
- `stage6_retrain.py:70-161`: JSON/split hashing, matrix validation, regenerated class/actor weights, specialist fitting.
- `stage6_retrain.py:162-230`: rebuild Stage 6 rows from 8 FPS features and enforce schemas C2 `65`, C3 `90`, suspicious `50`.
- `stage6_retrain.py:231-266`: causal gates and 1%-quantile train-OOF gate candidates.
- `stage6_retrain.py:267-423`: vectorized OOF evaluator mirroring C2 priority, pair propagation, history score/frame/timestamp reduction.
- `stage6_retrain.py:424-660`: actor metrics, score candidates, deterministic global search, coordinate refinement, exact causal parity before/after selection.
- `stage6_retrain.py:661-765`: actor/pair prediction export, pair metrics, leave-one-`split_group`-out fold training.
- `stage6_retrain.py:773-1033`: Stage 5 validation, mixed-profile selection materialization, final models, hashes, provenance, environment, CLI, acceptance gate.
- `stage6_profile_mix_search.py:1-120`: new per-specialist profile mixer; validates identical OOF row keys and searches deterministic seeds without locked test.

## Techniques producing 0.8469

1. Use only clean 8 FPS Stage 5 input and policy `8/4/8/24/450`; preserve ordered schemas `65/90/50`.
2. Use leave-one-`split_group`-out OOF across nine training groups; fit thresholds and gates only from held-out train-group scores.
3. Keep causal replay unchanged: 24 sampled frames, four-valid-observation strict-past warmup, timestamp derivatives, persistent evidence history.
4. Fix replay identity to `sample_index` and block pair evidence for absent endpoints.
5. Regenerate weights from 8 FPS rows; compare balanced, unweighted, legacy, depth, conservative, rich, and actor-balanced profiles.
6. Select specialist profiles independently: C2 `balanced`, C3 `unweighted`, suspicious_activity `balanced_depth4`.
7. Search score thresholds and existing gate thresholds jointly because history chooses highest qualified evidence across specialists.
8. Use 1%-quantile gate candidates, deterministic multi-start seeds, then coordinate refinement.
9. Require vectorized evaluator to match exact causal replay before and after every selected calibration.
10. Keep locked-test predictions absent during model selection and calibration.

Best calibration (`search_seed=20260823`):

```text
C2 threshold                    0.31567803025245667
C3 threshold                    0.22652508318424225
suspicious_activity threshold   0.6603246927261353
C3 side floor                   0.23424867951982659
C3 down ceiling                 0.04676047707994139
suspicious down floor           0.02284153357560299
suspicious motion floor         0.27528379193801616
suspicious lower floor         -0.10919798641680552
```

## Evidence artifacts

- `tmp/causal_8fps_stage6_profile_mix_search_20260827.json`: 27 profile combinations; SHA256 `C4A36645E3E12594BE9C99C711876C91E52EF462EB125739C51B644414F10A98`.
- `tmp/causal_8fps_stage6_profile_mix_fine_20260827.json`: fine-grid search; SHA256 `CCE16752AE85B6ED5C29C942D531C7E674CD8B9A50E32B630C97FAC3DED39663`.
- `tmp/causal_8fps_stage6_mix_actor_c2_axis_20260827.json`: C2-axis search; SHA256 `2115FC0071C31DF00CABDCC7DB8A690638B0B070CDB92E6DB6BDCAA2A1167F20`.
- `tmp/causal_8fps_stage6_mix_actor_c3_axis_20260827.json`: C3-axis search; SHA256 `79F64B4C0A168A0B230E786DB645AC50007D39A909A1C40337938047E41DD97A`.
- `tmp/causal_8fps_stage6_mix_actor_suspicious_axis_20260827.json`: suspicious-axis search; SHA256 `248F672F41604AC12FB91BA636A5F7C2D6C02EE4EAEF22A29BFBD222C0A40FF6`.
- `tmp/causal_8fps_stage6_mix_multistart_20260827.json`: ten-seed final search; SHA256 `20348D22CB6A9FC12621687D2C9F268872C61B1405D980F7F65CE24D4E3F6F3F`.
- `tmp/causal_8fps_stage6_mixed_084699_final_20260827/bundle_manifest.json`: final mixed bundle manifest; SHA256 `4B4A647A70D7DADAFFC2D89464C092F5DC8DEA0E639F8D6316794A7416ED5F17`.
- Final metrics SHA256: `F147D2CEFA16D12DBF47E2686870352810F2949013BF959D1843AFD7B982FDF1`.
- Final calibration SHA256: `6A5D8380B89457351D34896B92B1104D2D80D67F6C10AF664AEEAB5CE2865FD4`.
- Final provenance SHA256: `0692A38102D1E345C47260DCC7F9E35EF3B8EBCEA9D80AC315D3DB7210B07C14`.
- Final bundle: 41 hashed files, 27 fold models, three final models, OOF reproduction `20,511` rows/specialist, schemas `65/90/50`, policy `8/4/8/24/450`.
- `tmp/causal_8fps_stage6_20260827_run4`: earlier single-profile bundle and loader/reproduction proof; its `0.821701` predates absent-peer causal correction and is not current acceptance evidence.

## Tried but below target

- Independent specialist thresholds: `0.7553`; ignored cross-specialist history competition.
- Joint coordinate calibration: `0.7780`; converged to local optimum.
- Coarse 5%-quantile gates: `0.8217`; insufficient gate resolution.
- Legacy regenerated weighting: `0.81047`; overweighted positives.
- Fully unweighted models: `0.82773`; improved separation, still below target.
- Balanced depth 2: `0.80859`; underfit grouped actors.
- Balanced depth 4: `0.82295`; extra depth did not generalize.
- Conservative trees: `0.81647`; excessive regularization.
- Rich trees: `0.82271`; extra capacity did not improve OOF.
- Actor-balanced depth 3: `0.82217`; equal actor contribution remained insufficient.
- Actor-balanced depth 4: `0.81963`; deeper actor weighting regressed.
- First 27-profile mix: `0.83758`; specialist mixing helped but missed target.
- Fine 100k calibration: `0.84103`; single search basin remained suboptimal.
- Ten-seed multi-start: `0.84699`; best result, still `0.00301` below `0.85`.

## Reproduction boundary

No commit, merge, push, plan edit, locked-test tuning, or Step 7 evaluation occurred. Stage 6 mixed bundle is complete and Gate 3 passes. Promotion remains blocked until deferred Gate 4 passes for this exact bundle and hashes.
