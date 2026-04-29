
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_orthocenter_double_circles():
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
    
    plt.figure(figsize=(12, 12))
    plt.plot([B[0], C[0], A[0], B[0]], [B[1], C[1], A[1], B[1]], 'k-', alpha=0.3)
    
    # 1. CERCHIO 1 (CDHE) - Diametro CH
    c1_center = (C + H) / 2
    c1_radius = np.linalg.norm(C - H) / 2
    circle1 = plt.Circle(c1_center, c1_radius, color='orange', fill=True, alpha=0.1, label='Cerchio 1 (CDHE)')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(plt.Circle(c1_center, c1_radius, color='orange', fill=False, linewidth=3))
    
    # 2. CERCHIO 2 (ABDE) - Diametro AB
    # Poiché ADB e AEB sono 90°, A, B, D, E stanno sul cerchio con diametro AB
    c2_center = (A + B) / 2
    c2_radius = np.linalg.norm(A - B) / 2
    circle2 = plt.Circle(c2_center, c2_radius, color='cyan', fill=True, alpha=0.05, label='Cerchio 2 (ABDE)')
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(plt.Circle(c2_center, c2_radius, color='cyan', fill=False, linewidth=3, linestyle='--'))
    
    # Altezze
    plt.plot([A[0], D[0]], [A[1], D[1]], 'g--', alpha=0.6, label='Altezza AD')
    plt.plot([B[0], E[0]], [B[1], E[1]], 'g--', alpha=0.6, label='Altezza BE')
    plt.plot([C[0], H[0]], [C[1], H[1]], 'r-', linewidth=3, label='Linea CH (Strategia Davide)')
    
    # Punti
    points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'H': H}
    for name, p in points.items():
        color = 'red' if name == 'H' else 'black'
        plt.plot(p[0], p[1], 'o', color=color, markersize=10)
        plt.text(p[0]+0.15, p[1]+0.15, name, fontsize=16, fontweight='bold', color=color)

    plt.title("La Rete Logica: Il Doppio Rimbalzo degli Angoli", fontsize=18)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    
    output_path = "orthocenter_double_circles.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_orthocenter_double_circles()
