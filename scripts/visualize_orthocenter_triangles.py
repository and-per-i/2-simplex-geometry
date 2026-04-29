
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_orthocenter_triangles():
    # Vertici reali
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
    
    # L (Punto su AB)
    m_ab = (A[1]-B[1])/(A[0]-B[0])
    m_perp_c = -1/m_ab
    x_l = (m_perp_c * 6) / (m_perp_c - m_ab)
    y_l = m_ab * x_l
    L = np.array([x_l, y_l])
    
    plt.figure(figsize=(12, 10))
    
    # 1. Triangolo Base ABC
    plt.fill([B[0], C[0], A[0]], [B[1], C[1], A[1]], color='gray', alpha=0.05)
    plt.plot([B[0], C[0], A[0], B[0]], [B[1], C[1], A[1], B[1]], 'k-', alpha=0.3)
    
    # 2. Triangolo Rettangolo ADC (Verde)
    plt.fill([A[0], D[0], C[0]], [A[1], D[1], C[1]], color='green', alpha=0.15, label='Triangolo ADC (Altezza AD)')
    plt.plot([A[0], D[0], C[0], A[0]], [A[1], D[1], C[1], A[1]], 'g--', alpha=0.4)
    
    # 3. Triangolo Rettangolo BEC (Blu)
    plt.fill([B[0], E[0], C[0]], [B[1], E[1], C[1]], color='blue', alpha=0.15, label='Triangolo BEC (Altezza BE)')
    plt.plot([B[0], E[0], C[0], B[0]], [B[1], E[1], C[1], B[1]], 'b--', alpha=0.4)
    
    # 4. Triangolo Finale ALC (Rosso - La Tesi)
    plt.fill([A[0], L[0], C[0]], [A[1], L[1], C[1]], color='red', alpha=0.1, label='Triangolo ALC (Obiettivo: 90° in L)')
    plt.plot([A[0], L[0], C[0], A[0]], [A[1], L[1], C[1], A[1]], 'r-', linewidth=3)
    
    # Linee di connessione interne
    plt.plot([C[0], H[0]], [C[1], H[1]], 'r:', linewidth=2, label='Connessione Davide (CH)')
    
    # Punti
    points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'H': H, 'L': L}
    for name, p in points.items():
        color = 'red' if name in ['H', 'L'] else 'black'
        plt.plot(p[0], p[1], 'o', color=color, markersize=10)
        plt.text(p[0]+0.15, p[1]+0.15, name, fontsize=16, fontweight='bold', color=color)

    plt.title("I Triangoli della Dimostrazione: Dalle Ipotesi alla Tesi", fontsize=18)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    
    output_path = "orthocenter_triangles.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_orthocenter_triangles()
