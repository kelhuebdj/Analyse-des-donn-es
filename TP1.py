import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import openpyxl
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


if __name__=="__main__":
    data = pd.read_excel(r'C:\Users\info\Desktop\ACP\Classeur1.xlsx')
    print(data)
    print(data.shape)
    matrix=data.values
    #print(matrix)
    #La matrice centrée réduite:
    col_means = np.mean(matrix, axis=0)
    centered_matrix = matrix - col_means
    col_stds = np.std(centered_matrix, axis=0)
    centered_reduced_matrix = centered_matrix / col_stds
    #print(centered_reduced_matrix)
    #La matrice de corrélation:
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    #print(correlation_matrix)
    #Les valeurs propres et le vecteurs propres de la matrice de corrélation:
    values, vectors = np.linalg.eig(correlation_matrix)
   # print("Les valeurs propres de la matrice de corrélation sont :", values)
    #print("Les vecteurs propres de la matrice de corrélation sont :", vectors)
    #La projection sue les axes principaux:
    X_projected = np.dot( centered_reduced_matrix , vectors)#Le produit de la matrice centrée réduite avec les vecteurs propres 
    print(X_projected)
    sns.scatterplot(data=data, x="curb-weight", y="engine-size", hue="price" )
    #plt.show()
    #Le cercle de corrélation:
       # Calculer les coordonnées des variables sur le cercle de corrélation
    coords_variable = np.sqrt(values)[:, np.newaxis] * vectors
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor='black', linestyle='dashed'))

      # Tracer les flèches représentant les variables
    for i in range(len(coords_variable)):
         x = coords_variable[i][0]
         y = coords_variable[i][1]
         ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='black', ec='black')
         ax.text(x, y, f'Var{i+1}', ha='center', va='center')

      # Ajouter les étiquettes des axes
    ax.set_xlabel('Axe 1 (PC1)')
    ax.set_ylabel('Axe 2 (PC2)')

      # Afficher le cercle de corrélation
    plt.show()
   # Nombre de points aléatoires
    n_points = 10

   # Générer des points aléatoires sur le cercle de corrélation
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.vstack((x, y)).T

   # Calculer les quantités de projection
    for i in range(n_points):
          for j in range(coords_variable.shape[1]):
               cos_theta = np.dot(points[i], coords_variable[:, j]) / np.linalg.norm(points[i])
               quantite = cos_theta**2
               print(f"Quantité(M{i+1}, F{j+1}) = {quantite}")
    #Interpretation



