# Pose-Gaze glossary

- **IoU tracking**: Associates person detections across consecutive frames into a stable `track_id`. It protects landmark continuity; it does not classify actions or improve class separability by itself.
- **Temporal window**: Consecutive, contiguous frames from one video and one `track_id`, with one class label. A 30-frame window at the observed 30 FPS spans about one second.
- **Primary unit**: One actor decision keyed by `(video, actor_id)`. Video-level metrics are not acceptance metrics.
- **Live causal feed**: At frame `t`, inference may use only frame `t` and state derived from frames `< t`; future frames are forbidden.
- **Offline reference**: Full-video actor aggregation is retained only for historical comparison and must not be called live or realtime performance.
- **Causal replay benchmark**: A recorded stream is replayed in timestamp order with the same stateful rules intended for camera input. Its acceptance metric is actor-level macro-F1 and its audit includes first-flag latency.
- **Official benchmark (`benchmark gốc`)**: The original-label, actor-level causal replay used as the reproducibility reference. A sensitivity run with altered truth is never promoted implicitly.
- **Causal prefix state**: Actor state after frame `t`, computed only from frames `0..t`. Appending later frames may not change any earlier output.
- **C1/C4 family candidate**: Valid pose evidence of lower-body concealment/manipulation or opposite-sleeve manipulation. It does not by itself identify phone or paper.
- **`c1_c4_family_unknown`**: The required subtype result when pose qualifies the C1/C4 family but object evidence cannot reliably distinguish C1 from C4-v1.
- **Lower-body proxy**: A pose-relative wrist/arm state near the actor's hip or lap. It must not be described as confirmed under-table, pocket, phone, or paper evidence.
- **Direct object override**: An actor-owned `phone` detection assigns C1 and an actor-owned `cheating_paper` detection assigns C4 at that observed frame, regardless of competing pose or behavior cues.
- **Baseline paper**: An allowed exam sheet already placed on the desk. `baseline_paper` may support scene context but can never flag an actor or assign a behavior class.
- **C3 orientation proxy**: Actor-level evidence that the actor's head and/or upper body is persistently oriented toward the explicit peer. It is not true gaze or intent. The primary contract is signed, baseline-relative nose/head and torso geometry from pose landmarks; face mesh, hands, and missingness are not primary C3 evidence.
- **Training domain**: Front-camera groups remain the current training domain; rear-camera data is excluded from the current actor benchmark.
- **Flip augmentation**: A training-only geometric transform valid only when a class is invariant under left/right reversal and every paired landmark is swapped with its counterpart.
