
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_geometry_problem():
    # 1. Definiamo i punti del triangolo ABC
    A = np.array([0, 0])
    B = np.array([5, 0])
    C = np.array([1, 4])
    
    # 2. Calcoliamo il punto medio M di AB
    M = (A + B) / 2
    
    # 3. La "Mossa di Davide": Estendere la mediana CM fino a E
    E = 2 * M - C
    
    # --- VISUALIZZAZIONE ---
    plt.figure(figsize=(10, 8))
    
    # Triangolo originale ABC
    plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'b-', label='Triangolo Originale ABC', linewidth=2)
    
    # Mediana CM
    plt.plot([C[0], M[0]], [C[1], M[1]], 'g--', label='Mediana CM', linewidth=2)
    
    # Estensione Davide
    plt.plot([M[0], E[0]], [M[1], E[1]], 'r--', label='Estensione Davide (CM=ME)', linewidth=2)
    plt.plot([A[0], E[0]], [A[1], E[1]], 'r:', alpha=0.6)
    plt.plot([B[0], E[0]], [B[1], E[1]], 'r:', alpha=0.6)
    
    # Punti
    points = {'A': A, 'B': B, 'C': C, 'M': M, 'E': E}
    for name, p in points.items():
        color = 'black' if name != 'E' else 'red'
        plt.plot(p[0], p[1], 'o', color=color)
        plt.text(p[0]+0.1, p[1]+0.1, name, fontsize=12, fontweight='bold', color=color)

    plt.title("L'Intuizione di Davide: Il Parallelogramma Ausiliario", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    # SALVATAGGIO INVECE DI SHOW (per ambiente headless)
    output_path = "median_solution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_geometry_problem()
