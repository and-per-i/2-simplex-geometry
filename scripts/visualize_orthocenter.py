
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_orthocenter_problem():
    # 1. Punti del triangolo ABC
    A = np.array([2, 5])
    B = np.array([0, 0])
    C = np.array([6, 0])
    
    # 2. Calcoliamo le altezze
    # Altezza da A su BC (base orizzontale y=0) -> x = 2
    H_val = np.array([A[0], 0]) 
    
    # Altezza da B su AC
    m_ac = (C[1]-A[1])/(C[0]-A[0])
    m_perp_b = -1/m_ac
    # y = m_perp_b * x
    # Intersezione con AC: y - 5 = m_ac * (x - 2)
    x_k = (5 - 2*m_ac) / (m_perp_b - m_ac)
    y_k = m_perp_b * x_k
    K_val = np.array([x_k, y_k])
    
    # 3. Ortocentro (Intersezione delle altezze)
    # Sappiamo che x=2 per la prima altezza, quindi y = m_perp_b * 2
    H_point = np.array([2, m_perp_b * 2])
    
    # --- VISUALIZZAZIONE ---
    plt.figure(figsize=(10, 8))
    
    # Triangolo
    plt.plot([B[0], C[0], A[0], B[0]], [B[1], C[1], A[1], B[1]], 'b-', label='Triangolo ABC', linewidth=2)
    
    # Altezze (Verde)
    plt.plot([A[0], H_val[0]], [A[1], H_val[1]], 'g--', label='Altezza da A')
    plt.plot([B[0], K_val[0]], [B[1], K_val[1]], 'g--', label='Altezza da B')
    
    # Punto suggerito da Davide (Ortocentro)
    plt.plot(H_point[0], H_point[1], 'ro', markersize=10, label='Punto "i" (Davide)')
    plt.text(H_point[0]+0.1, H_point[1]+0.1, "H (Ortocentro)", fontsize=12, fontweight='bold', color='red')

    # Terza altezza (Gialla) per dimostrare che Davide ha ragione
    m_ab = (A[1]-B[1])/(A[0]-B[0])
    m_perp_c = -1/m_ab
    # y = m_perp_c * (x - 6)
    x_l = (m_perp_c * 6 - 0) / (m_perp_c - m_ab) # Intersezione con AB
    y_l = m_perp_c * (x_l - 6)
    L_val = np.array([x_l, y_l])
    plt.plot([C[0], L_val[0]], [C[1], L_val[1]], 'y:', alpha=0.8, label='Terza altezza (coincidente!)')

    plt.title("L'Intuizione di Davide: L'Intersezione Strategica", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    output_path = "orthocenter_solution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Immagine salvata in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    draw_orthocenter_problem()
