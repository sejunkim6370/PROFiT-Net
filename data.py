import pandas as pd
from pymatgen.core.structure import Structure
from matminer.featurizers.structure import OrbitalFieldMatrix

structure = [Structure.from_file('LiAg3O2.cif')]

df = pd.DataFrame({'structure':structure})

ofm = OrbitalFieldMatrix(period_tag=False)
df = ofm.featurize_dataframe(df,'structure')
df.to_csv('data.csv', index=False)
