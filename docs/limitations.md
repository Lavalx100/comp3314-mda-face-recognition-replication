# Limitations and interpretation rules

1. **Dataset fidelity:** current experiment uses scikit-learn's 64×64 Olivetti copy, not the original-resolution ORL images.
2. **Protocol fidelity:** five repeated random splits and fixed feature settings differ from the paper's reported best recognition rates across configurations. We do not have the exact original random partitions.
3. **Few repeats:** five split means and SDs are descriptive; they do not establish significance or stable method rankings.
4. **Scale sensitivity:** generalized eigenvectors have scaling conventions. Euclidean 1-NN is not invariant to arbitrary feature-axis rescaling; the sensitivity run materially changes accuracy.
5. **Fixed dimensions:** PCA and MDA dimensions are not selected by nested validation. A future study should tune using training/validation data only.
6. **Implementation:** this is a group implementation, not official author code. Regularization and numerical handling are implementation choices that must be disclosed.
7. **Dataset scope:** ORL is small, controlled, grayscale, and largely frontal. It does not establish modern unconstrained face-recognition performance.
8. **Training objective:** an increasing MDA Fisher-ratio trace is an optimization diagnostic, not evidence of improved held-out classification.

## Safe language

Say: “Under our fixed 64×64 protocol and unit-L2 axes, MDA/2-1 led only at the smallest gallery size, while Fisherfaces led at larger gallery sizes; MDA/2-2 improved with more examples but did not lead in this configuration.”

Do not say: “We disproved the paper,” “MDA is worse,” or “we reproduced the paper's exact results.”
