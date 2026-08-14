# Object-detection glossary

- **Baseline paper**: An allowed exam sheet already placed on the desk. Detecting it never flags cheating.
- **Cheating paper**: An additional unauthorized paper introduced from a pocket, bag, sleeve, or another concealed source. When partially occluded, its box covers only the visible paper edge or fragment; annotators must not infer the hidden full sheet extent.
- **In-domain classroom frame**: A frame captured with the target classroom camera layout, scale, lighting, desks, and actor-object geometry.
- **External object image**: An image outside the target classroom capture domain. It may support an experiment but cannot replace in-domain validation.
- **Full-frame inference**: Detection from the current complete frame without future frames or temporal look-ahead.
- **SAHI training view**: A spatial crop generated from a current training frame with labels remapped into crop coordinates. It is a training sample, not future temporal context.
- **SAHI inference**: Current-frame sliced inference whose detections are remapped to full-frame coordinates and deduplicated before evaluation.
- **Inference-equivalent evaluation**: Detector evaluation that uses the same current-frame color ordering, resize/slicing, coordinate remapping, deduplication, and confidence handling as the production inference path.
- **Qualified positive frame**: A current frame containing a correctly classified, sufficiently reliable, actor-owned `phone` or `cheating_paper` detection. One qualified positive frame is sufficient to flag that actor for C1/C4; failure to detect the object in other frames is not, by itself, a failure for this objective.
- **Pose-gated detection**: In realtime, the pose/suspicious-activity module activates object detection on the current frame. The detector is not an unconditional always-on flag source.
- **C1/C4 priority**: When pose has produced `suspicious_activity` and the activated detector finds a qualified `phone` or `cheating_paper`, the object-backed C1/C4 decision takes priority over the generic suspicious-activity result.
- **Detector acceptance metric**: Per-class precision, recall, F1, and mAP computed on held-out frames grouped by source video; aggregate F1 cannot hide a failed target class.
- **Promotion holdout**: In-domain videos not used to choose data, augmentation, thresholds, checkpoints, or stopping decisions.
- **Training box imbalance**: Unequal numbers of annotated object instances per class in the training split. It is distinct from frame imbalance because one frame may contain several boxes of the same class.
- **Closed test split**: A held-out split used only for the final evaluation of a fixed training recipe. Its metrics must not select sampling ratios, loss functions, epochs, checkpoints, thresholds, or augmentations.
