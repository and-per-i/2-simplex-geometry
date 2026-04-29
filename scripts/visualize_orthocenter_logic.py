
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_orthocenter_proof_logic():
    # Vertici
    A = np.array([2, 5])
    B = np.array([0, 0])
    C = np.array([6, 0])
    
    # Piedi altezze
    D = np.array([2, 0])
    m_ac = (C[1]-A[1])/(C[0]-A[0])
    m_perp_b = -1/m_ac
    x_e = (5 - 2*m_ac) / (m_perp_b - m_ac)
    y_e = m_perp_b * x_e
    E = np.array([x_e, y_e])
    
    # H (Ortocentro)
    H = np.array([2, m_perp_b * 2])
    
    plt.figure(figsize=(10, 10))
    
    # Triangolo base
    plt.plot([B[0], C[0], A[0], B[0]], [B[1], C[1], A[1], B[1]], 'k-', alpha=0.2)
    
    # Altezze iniziali
    plt.plot([A[0], D[0]], [A[1], D[1]], 'g--', label='Altezza AD')
    plt.plot([B[0], E[0]], [B[1], E[1]], 'g--', label='Altezza BE')
    
    # IL CERCHIO DI DAVIDE (Quadrilatero CDHE)
    # CDHE è ciclico perché angoli in D e E sono 90°. 
    # Il diametro è CH.
    diameter_center = (C + H) / 2
    radius = np.linalg.norm(C - H) / 2
    circle = plt.Circle(diameter_center, radius, color='orange', fill=True, alpha=0.1, label='Area di Conoscenza (Cerchio Ciclico)')
    plt.gca().add_patch(circle)
    plt.gca().add_patch(plt.Circle(diameter_center, radius, color='orange', fill=False, linewidth=3))
    
    # Linea CH (Il collegamento trovato grazie a Davide)
    plt.plot([C[0], H[0]], [C[1], H[1]], 'r-', linewidth=3, label='Linea CH (Nuova Connessione)')
    
    # Punti
    points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'H': H}
    for name, p in points.items():
        color = 'red' if name == 'H' else 'black'
        plt.plot(p[0], p[1], 'o', color=color, markersize=8)
        plt.text(p[0]+0.15, p[1]+0.15, name, fontsize=14, fontweight='bold', color=color)

    plt.title("La Costruzione della Dimostrazione: Il Cerchio Ciclico", fontsize=16)
    plt.legend(loc='upper left')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.4)
    
    output_path = "orthocenter_logic_proof.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_orthocenter_proof_logic()
