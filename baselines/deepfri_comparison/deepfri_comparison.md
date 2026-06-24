# Comparison with DeepFRI

## Addressing Reviewer 2, Comment 1
> "The comparison with DeepFRI is not sufficiently clear. The manuscript states that DeepFRI is a general-purpose model trained across multiple taxa, but the Discussion suggests that DeepFRI was trained on the same Viridiplantae dataset as DeepGreenGO. These two descriptions lead to different interpretations. Please clearly state whether the comparison was performed against the original pretrained DeepFRI model, a fine-tuned DeepFRI model, or a DeepFRI architecture retrained from scratch on the same plant-specific dataset."

**Clarification for the Manuscript:**
For our baseline comparison against DeepFRI, we utilized the **original pretrained DeepFRI model**. We did not retrain or fine-tune DeepFRI from scratch on our plant-specific (Viridiplantae) dataset. 

The rationale for this comparison is to demonstrate the limitations of general-purpose, multi-taxa models (like the original DeepFRI) when applied specifically to plant proteins, which often possess unique functional adaptations and taxonomic biases not fully captured by general models. 

By comparing our plant-specific DeepGreenGO model to the original pretrained DeepFRI, we highlight the performance gains achieved by domain-specific training on a targeted taxonomic group. The confusion in the Discussion section regarding DeepFRI being trained on the same dataset will be corrected in the revised manuscript to clearly reflect that the original pretrained weights were used for inference on our test set.

## Ensuring Fair Evaluation
To ensure a fair evaluation:
1. We apply the original pretrained DeepFRI model to our newly constructed, homology-reduced test set (`_test.txt`).
2. We compute the exact same metrics (Fmax, Smin, AUROC, AUPRC) on the DeepFRI predictions as we do for DeepGreenGO.
3. This allows for a direct, unbiased comparison of a state-of-the-art general-purpose model versus our specialized plant-centric model.
