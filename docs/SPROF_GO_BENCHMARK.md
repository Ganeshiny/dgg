# SPROF-GO external benchmark

SPROF-GO is evaluated as an external pretrained method, not retrained on the
DeepGreenGO split. Its official inference averages five models per ontology
and then applies DIAMOND-based label diffusion against SPROF-GO's bundled
training and validation proteins. The manuscript must state this distinction.

On ARC, first run `bash "arc slurms/setup_sprof_go_arc.sh"` from the repository
root. This downloads roughly 1.4 GB of SPROF-GO files and a 5.3 GB ProtT5
archive outside the Git repository under `/home/ganeshiny.sridharan/dgg/external`.

Then submit `sbatch "arc slurms/run_sprof_go_benchmark.slurm"`.

Scores are aligned to the exact DeepGreenGO test protein IDs and target GO-term
vocabulary. Target terms absent from SPROF-GO's vocabulary receive score zero;
SPROF-only terms are excluded. Reports record target, SPROF, and common term
counts. DeepGOPlus and DeepGO-SE are not included in the new benchmark plot.

For publication, add an overlap audit against SPROF-GO's packaged train+valid
sequences and disclose the 10% identity threshold used by its label diffusion.
