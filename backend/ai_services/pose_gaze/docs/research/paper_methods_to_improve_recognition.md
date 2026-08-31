# What the two supplied papers actually used to improve recognition

Sources reviewed: `Exploring_the_potential_of_skeleton_and_machine_le (3).pdf` (Tran et al., 2023) and `R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2 (3).pdf` (Arinaldi and Fanany, 2017). Page numbers below are PDF pages.

## Implemented methods with a reported improvement

| Paper | Method | Evidence | What improved | Applicability here |
|---|---|---|---|---|
| Arinaldi and Fanany, p. 4-5 | Replace 3D-CNN softmax classifier with **XGBoost** on the 512-D 3D-CNN feature vector. | Table 2: accuracy 0.700 -> 0.811; Cohen's kappa 0.622 -> 0.760. | +11.1 pp accuracy and +0.138 kappa, not F1. | Worth considering only after a valid video/group-held-out 3D-CNN feature extractor exists. It is not a change to skeleton `2d_world` features. |
| Tran et al., p. 8-9 | Compare SVM, DT, RF, XGBoost, and Ac-LSTM on 51 pose features. | Table 3 reports XGBoost highest F1: 0.92; Ac-LSTM 0.90. | XGBoost > alternatives in their split. | We already use XGBoost. The score is not a target: their task is binary and their reported 80/20 split is at frame/sample level. |

## Implemented, but no isolated proof of metric gain

| Paper | Method | Evidence | Correct interpretation |
|---|---|---|---|
| Arinaldi and Fanany, p. 4 | Augment only `Looking Friend` and `Give Code` train chunks by horizontal flip. | Authors explicitly say it increases training data; no ablation. | Do not flip multi-class pose data blindly. A valid implementation must swap left/right landmarks and be allowed only for classes invariant to side. |
| Arinaldi and Fanany, p. 4 | 30-frame chunks, 10-frame overlap. | Used for gesture chunks/smoother frequent descriptions; no accuracy/F1 ablation. | Test as a controlled window ablation, not as a promised gain. |
| Arinaldi and Fanany, p. 4 | Dropout 0.3 at input and 0.5 later in 3D-CNN; XGBoost L1 alpha 0.1 and L2 lambda 15. | Architecture/hyperparameter description; no ablation. | Anti-overfit candidates only. No evidence they solve our held-out-group failure. |
| Tran et al., p. 6-7 | IoU tracking at threshold 0.5; sequence input to 4-layer, 50-unit Ac-LSTM with dropout 0.2. | Tracking is used to avoid mixing people across consecutive frames; authors say dropout helps avoid memorization. | Tracking integrity is already required. Ac-LSTM is a future model experiment, not evidence it beats our XGBoost on our split. |

## Author-proposed future improvements

- **Remove occluded lower-body keypoints**: Tran et al. say classroom cameras mainly see upper body, lower body is often occluded, and removing those keypoints could reduce model weight/noise (p. 10). This directly supports the running `2d_world_upper` ablation.
- **Expand the dataset**: Tran et al. identify ambiguous normal looking-left/right vs copying-answer behaviour and say they will investigate it; they explicitly propose expanding the dataset (p. 10). This supports collecting front hard negatives and class/domain coverage, not duplicating windows.
- **GCN for skeleton data**: Tran et al. propose graph convolutional networks as future work (p. 10). It is not evaluated in the paper.

## Methods that do not answer the current F1 problem

- Arinaldi and Fanany's bidirectional-input MIMO LSTM generates text descriptions (p. 4-5). Its result is word accuracy, not gesture F1; it is outside the current classifier target.
- Their MOG foreground crop, 80x80 grayscale input, and 3D-CNN are a pixel-video pipeline (p. 3-4), not a skeleton feature engineering technique.
- Tran et al.'s multi-camera data collection (left/right/front) is a dataset design detail, not a demonstrated cross-camera generalization method (p. 3-4).

## Why paper scores must not be compared to our primary video macro-F1

- Tran et al. have binary cheating/non-cheating and split 45,535 frame/sample rows 80/20 (p. 8); the paper does not state a held-out-video, actor, group, or scene split. Their 0.92 F1 can contain adjacent-frame/video leakage and is not comparable to multi-class video macro-F1.
- Arinaldi and Fanany split gesture **chunks** 80/20 after segmenting 71 videos (p. 4). Their chunks overlap and the paper reports no held-out-video/actor protocol, macro-F1, precision/recall, seeds, or augmentation ablation. Their 0.811 is accuracy, not F1.

## Evidence-based next experiments for `2d_world`

1. Finish the front-v4 `2d_world_upper` ablation. Accept it only if primary video macro-F1 and every class's F1 do not regress materially; specifically inspect former lower-body-dependent classes.
2. If it wins, add wrist-head, wrist-torso, elbow angle, and velocity/acceleration features in a separate ablation. Neither paper evaluates those features, so label this project evidence, not paper-backed gain.
3. Test 30/20 against 16/4 with the unchanged front-v4 split and model. It is a cadence/context experiment derived from Arinaldi and Fanany, not a claim of F1 improvement.
4. Only then consider left/right-safe augmentation, with a video-level leakage-safe split and a documented landmark swap. It must be training-only.
5. Treat 3D-CNN+XGBoost or skeleton GCN as separate model families after the feature/domain ablations; both papers lack a valid comparison against our protocol.
