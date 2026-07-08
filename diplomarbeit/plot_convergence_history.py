import matplotlib.pyplot as plt
import os

def plot_convergence_history():
    iterations = [1, 2, 3]

    # Die Daten laut Vorgabe (ermittelt als maximales Delta über alle 4 Investoren pro Iteration)
    # Model Run with aFRR
    # Iteration 1: max(23.491, 23.298, 23.332, 23.456) = 23.491
    # Iteration 2: max(0.722, 0.611, 0.599, 0.584) = 0.722
    # Iteration 3: max(0.170, 0.013, 0.030, 0.052) = 0.170
    max_delta_with_afrr = [23.491, 0.722, 0.170]

    # Model Run without aFRR
    # Iteration 1: max(23.999, 23.935, 23.347, 23.386) = 23.999
    # Iteration 2: max(1.659, 1.539, 1.572, 1.597) = 1.659
    # Iteration 3: max(0.169, 0.038, 0.099, 0.342) = 0.342
    max_delta_without_afrr = [23.999, 1.659, 0.342]

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, max_delta_with_afrr, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8, label='Multi-Service Scenario')
    plt.plot(iterations, max_delta_without_afrr, marker='s', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='Energy-Only Scenario')

    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Tolerance (0.5 MW)')

    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Max Delta (MW)', fontsize=12)
    plt.title('Convergence History Comparison', fontsize=14)
    plt.xticks(iterations)
    plt.grid(True, which='both', linestyle=':', color='gray', alpha=0.7)
    
    plt.legend(fontsize=12)
    
    plt.tight_layout()

    # Bild speichern
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'convergence_history_comparison.pdf')
    plt.savefig(output_path)
    print(f"Plot erfolgreich als '{output_path}' gespeichert.")
    
    # Bild rendern
    plt.show()

if __name__ == '__main__':
    plot_convergence_history()
