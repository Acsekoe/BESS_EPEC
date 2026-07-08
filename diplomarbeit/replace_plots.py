import os

with open('c:/vscode/diplomarbeit/bess_epec_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ("plt.tight_layout()\n    plt.show()\n    \n    cols_res = [c for c in df_res.columns if df_res[c].sum() > 0]",
     "plt.tight_layout()\n    plt.savefig('output/Inputs_Load_Profiles.pdf')\n    plt.close()\n    \n    cols_res = [c for c in df_res.columns if df_res[c].sum() > 0]"),
    
    ("        plt.tight_layout()\n        plt.show()\n\ndef plot_grid_enhanced():",
     "        plt.tight_layout()\n        plt.savefig('output/Inputs_RES_Profiles.pdf')\n        plt.close()\n\ndef plot_grid_enhanced():"),
    
    ("    plt.title(\"Grid topology: Node and line limits\")\n    plt.axis('off')\n    plt.tight_layout()\n    plt.show()\n\n\ndef plot_bess_detail",
     "    plt.title(\"Grid topology: Node and line limits\")\n    plt.axis('off')\n    plt.tight_layout()\n    plt.savefig('output/Grid_Topology.pdf')\n    plt.close()\n\n\ndef plot_bess_detail"),
    
    ("    ax3.set_ylim(-5, 105)\n    ax3.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.show()\n\ndef plot_system_aggregation",
     "    ax3.set_ylim(-5, 105)\n    ax3.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig(f'output/{node}_{investor}_Detail.pdf')\n    plt.close()\n\ndef plot_system_aggregation"),
    
    ("    plt.legend()\n    plt.grid(True, alpha=0.3)\n    plt.tight_layout()\n    plt.show()\n\ndef print_scenario_overview",
     "    plt.legend()\n    plt.grid(True, alpha=0.3)\n    plt.tight_layout()\n    plt.savefig('output/System_Balance.pdf')\n    plt.close()\n\ndef print_scenario_overview"),
    
    ("        ax2.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1), fontsize='small')\n        ax2.axhline(0, color='black', linewidth=0.5)\n        \n        plt.tight_layout()\n        plt.show()\n\ndef plot_afrr_provision",
     "        ax2.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1), fontsize='small')\n        ax2.axhline(0, color='black', linewidth=0.5)\n        \n        plt.tight_layout()\n        plt.savefig(f'output/{n}_Balance.pdf')\n        plt.close()\n\ndef plot_afrr_provision"),
    
    ("        ax2.set_xlabel(\"Hour\")\n        ax2.set_xticks(range(0, 25, 4))\n        ax2.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.show()\n\ndef plot_convergence",
     "        ax2.set_xlabel(\"Hour\")\n        ax2.set_xticks(range(0, 25, 4))\n        ax2.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.savefig(f'output/{n}_aFRR.pdf')\n        plt.close()\n\ndef plot_convergence"),
    
    ("    plt.xlabel(\"Iteration\")\n    plt.grid(True, alpha=0.3, which='both')\n    plt.legend()\n    plt.tight_layout()\n    plt.show()\n\ndef plot_investment_bar_chart",
     "    plt.xlabel(\"Iteration\")\n    plt.grid(True, alpha=0.3, which='both')\n    plt.legend()\n    plt.tight_layout()\n    plt.savefig('output/Convergence.pdf')\n    plt.close()\n\ndef plot_investment_bar_chart"),
    
    ("    plt.xlabel(\"Node\")\n    plt.legend(title=\"Investor\", loc='upper left', bbox_to_anchor=(1.05, 1))\n    plt.grid(axis='y', alpha=0.3)\n    plt.tight_layout()\n    plt.show()\n\ndef plot_line_congestion",
     "    plt.xlabel(\"Node\")\n    plt.legend(title=\"Investor\", loc='upper left', bbox_to_anchor=(1.05, 1))\n    plt.grid(axis='y', alpha=0.3)\n    plt.tight_layout()\n    plt.savefig('output/Investment_Capacities.pdf')\n    plt.close()\n\ndef plot_line_congestion"),
    
    ("    plt.ylabel(\"Line\")\n    plt.xlabel(\"Hour\")\n    plt.tight_layout()\n    plt.show()\n\ndef plot_overall_afrr_provision",
     "    plt.ylabel(\"Line\")\n    plt.xlabel(\"Hour\")\n    plt.tight_layout()\n    plt.savefig('output/Line_Congestion.pdf')\n    plt.close()\n\ndef plot_overall_afrr_provision"),
    
    ("    ax2.set_xticks(range(0, 25, 4))\n    ax2.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.show()\n\ndef print_revenue_and_ep_ratio",
     "    ax2.set_xticks(range(0, 25, 4))\n    ax2.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig('output/Total_aFRR.pdf')\n    plt.close()\n\ndef print_revenue_and_ep_ratio")
]

for tgt, repl in replacements:
    num_replacements = content.count(tgt)
    if num_replacements == 0:
        print(f"Warning: Chunk not found:\n{tgt[:50]}...")
    content = content.replace(tgt, repl)

main_block_tgt = """if __name__ == "__main__":
    try:
        # # --- Latex Printer ---"""

main_block_repl = """if __name__ == "__main__":
    import os
    import sys
    os.makedirs('output', exist_ok=True)
    
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger("output/console_log.txt")
    
    with open(__file__, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print("--- Parameter Konfiguration ---")
        for i in range(28, 51):
            print(lines[i].strip())
        print("-------------------------------")

    try:
        # # --- Latex Printer ---"""

content = content.replace(main_block_tgt, main_block_repl)

with open('c:/vscode/diplomarbeit/bess_epec_model.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Changes applied!")
