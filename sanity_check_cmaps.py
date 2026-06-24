import numpy as np, os, glob

npz_dir = 'preprocessing/data/structure_files/tmp_cmap_files'
files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
print(f'Total .npz files found: {len(files)}')
issues = []
for f in files[:50]:
    prot = os.path.basename(f).replace('.npz','')
    d = np.load(f)
    ca = d['C_alpha']
    cb = d['C_beta']
    plddt = d['plddt']
    sym_ok = np.allclose(ca, ca.T, atol=1e-4, equal_nan=True)
    diag_ok = np.allclose(np.diag(ca), 0, atol=1e-4)
    neg = bool(np.nanmin(ca) < -0.01)
    shape_ok = ca.shape[0] == len(plddt) and ca.shape[0] == cb.shape[0]
    all_nan = bool(np.all(np.isnan(ca)))
    if not sym_ok or not diag_ok or neg or not shape_ok or all_nan:
        issues.append(f'{prot}: sym={sym_ok} diag={diag_ok} neg={neg} shape={shape_ok} all_nan={all_nan}')

if issues:
    print('Issues found:')
    for i in issues:
        print(' ', i)
else:
    print('All 50 checked files passed sanity checks.')

# Print stats for one file
f = files[5]
d = np.load(f)
ca = d['C_alpha']
print()
print(f'Sample: {os.path.basename(f)}, shape={ca.shape}, min={np.nanmin(ca):.2f}, max={np.nanmax(ca):.2f}, mean={np.nanmean(ca):.2f}')
print(f'pLDDT range: {d["plddt"].min():.2f} - {d["plddt"].max():.2f}')
print(f'NaN count in C_alpha: {np.sum(np.isnan(ca))}')
print(f'Contact pairs (< 10A): {np.sum(ca < 10)}')
