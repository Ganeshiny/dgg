import sys
sys.path.append('.')
from plot_extended_ablations import load_input_ablations
df = load_input_ablations()
summary = df.groupby(['Ontology', 'Modality'])[['Macro_Fmax', 'Micro_Fmax']].mean().reset_index()
print(summary.to_string(index=False))
