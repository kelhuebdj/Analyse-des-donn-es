import pandas as pd
from fanalysis.ca import CA
import matplotlib.pyplot as plt
plt.ion()

# Load data
df = pd.read_excel(r'C:\Users\soumi\Downloads\TP.xlsx', index_col=0)
print(df)

# Remove row and column labels
X = df.values

# Fit correspondence analysis
afc = CA(row_labels=df.index, col_labels=df.columns, stats=True)
afc.fit(X)

# Print eigenvalues
print(afc.eig_)

# Plot eigenvalues
afc.plot_eigenvalues()
plt.pause(1)

# Get row coordinates
info = afc.row_topandas()
print(info.columns)

# Plot row coordinates
afc.mapping_row(num_x_axis=1, num_y_axis=2)
coord_lig = afc.row_coord_[:, :2]
print(coord_lig)
plt.pause(1)

# Plot row and column associations
afc.mapping(num_x_axis=1, num_y_axis=2)
plt.pause(1)
# Show all plots
plt.show(block=True)