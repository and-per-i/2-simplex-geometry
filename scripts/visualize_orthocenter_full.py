
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_orthocenter_full():
    # Vertici del triangolo ABC
    A = np.array([2, 5])
    B = np.array([0, 0])
    C = np.array([6, 0])
    
    # Piedi delle altezze
    # D: Piede altezza da A su BC (y=0)
    D = np.array([A[0], 0]) 
    
    # E: Piede altezza da B su AC
    m_ac = (C[1]-A[1])/(C[0]-A[0])
    m_perp_b = -1/m_ac
    # y = m_perp_b * x
    x_e = (5 - 2*m_ac) / (m_perp_b - m_ac)
    y_e = m_perp_b * x_e
    E = np.array([x_e, y_e])
    
    # H: Ortocentro (Intersezione delle prime due altezze)
    H = np.array([2, m_perp_b * 2])
    
    # L: Piede altezza da C su AB
    m_ab = (A[1]-B[1])/(A[0]-B[0])
    m_perp_c = -1/m_ab
    # y = m_perp_c * (x - 6)
    x_l = (m_perp_c * 6) / (m_perp_c - m_ab)
    y_l = m_ab * x_l
    L = np.array([x_l, y_l])
    
    plt.figure(figsize=(10, 8))
    
    # Triangolo
    plt.plot([B[0], C[0], A[0], B[0]], [B[1], C[1], A[1], B[1]], 'b-', linewidth=2, label='Triangolo ABC')
    
    # Altezze AD e BE (Verdi)
    plt.plot([A[0], D[0]], [A[1], D[1]], 'g--', alpha=0.5, label='Altezze AD e BE')
    plt.plot([B[0], E[0]], [B[1], E[1]], 'g--', alpha=0.5)
    
    # Terza altezza CHL (Gialla)
    plt.plot([C[0], L[0]], [C[1], L[1]], 'y:', linewidth=3, label='Altezza CHL (Tesi)')
    
    # Punti e Etichette
    points = {
        'A': A, 'B': B, 'C': C, 
        'D': D, 'E': E, 'L': L, 
        'H': H
    }
    
    for name, p in points.items():
        color = 'red' if name == 'H' else 'black'
        plt.plot(p[0], p[1], 'o', color=color, markersize=8)
        # Offset dinamico per le etichette
        offset = 0.15
        plt.text(p[0]+offset, p[1]+offset, name, fontsize=14, fontweight='bold', color=color)

    plt.title("Dimostrazione Completa dell'Ortocentro: Concorrenza delle Altezze", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    
    output_path = "orthocenter_full.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_orthocenter_full()
