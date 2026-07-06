import gemmi
st = gemmi.Structure()
m = gemmi.Model("1")
m.add_chain(gemmi.Chain("A"))
m.add_chain(gemmi.Chain("B"))
print([c.name for c in m])
# remove chain B
chains_to_keep = []
for c in m:
    if c.name == "A":
        chains_to_keep.append(c)

# try to replace chains:
try:
    m.chains = chains_to_keep
except Exception as e:
    print(f"Error: {e}")

# What if we delete from the end?
for i in range(len(m)-1, -1, -1):
    if m[i].name != "A":
        del m[i]

print([c.name for c in m])
