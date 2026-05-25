import matplotlib.pyplot as plt
import os

# One color + marker per combination — consistent across all plots
COMBO_STYLES = {
    'none':                     {'color': 'black',  'marker': 'o', 'linestyle': '-'},
    #______________________________________________________________________
    #one transformations
    #______________________________________________________________________
    'flip':                     {'color': 'blue',   'marker': 's', 'linestyle': '-'},  #1
    'none_cutmix':              {'color': 'green',  'marker': '^', 'linestyle': '-'},  #2
    'rotation_deg_15':          {'color': 'red',    'marker': 'D', 'linestyle': '-'},  #3
    'rotation_deg_30':          {'color': 'purple', 'marker': 'v', 'linestyle': '-'},  #4
    'rotation_deg_90':          {'color': 'orange', 'marker': '*', 'linestyle': '-'},  #5
    'color_0.2':                {'color': 'green',  'marker': 'x', 'linestyle': '-'},  #6
    'color_0.3':                {'color': 'cyan',   'marker': 'p', 'linestyle': '-'},  #7
    'crop':                     {'color': 'magenta', 'marker': 'd', 'linestyle': '-'}, #8

    #_______________________________________________________________________________
    # Two transformations
    # best degree rotation: 15
    # best color_brightness_contrast: 0.3
    #_______________________________________________________________________________
    'crop_cutmix':              {'color': 'tab:blue',   'marker': 's', 'linestyle': '-'},
    'flip_rotation':            {'color': 'tab:green',  'marker': 'D', 'linestyle': '-'},
    'flip_crop':                {'color': 'tab:red',    'marker': 'v', 'linestyle': '-'},
    'flip_cutmix':              {'color': 'tab:purple', 'marker': '*', 'linestyle': '-'},
    'flip_color':               {'color': 'tab:brown',  'marker': 'X', 'linestyle': '-'},
    'rotation_color':           {'color': 'tab:orange', 'marker': '^', 'linestyle': '-'},
    'rotation_crop':            {'color': 'tab:pink',   'marker': 'P', 'linestyle': '-'},
    'rotation_deg_15_cutmix':   {'color': 'tab:gray',   'marker': '<', 'linestyle': '-'},
    'color_0.3_cutmix':         {'color': 'tab:olive',  'marker': '>', 'linestyle': '-'},
    'color_crop':               {'color': 'tab:cyan',   'marker': 'o', 'linestyle': '-'},


    #_______________________________________________________________________________
    # Three transformations
    #_______________________________________________________________________________
    #write your code here:
    'rotation_crop_cutmix':     {'color': 'tab:blue',   'marker': 's', 'linestyle': '-'},
    'flip_color_crop':          {'color': 'tab:brown',  'marker': 'X', 'linestyle': '-'},
    'rotation_color_cutmix':    {'color': 'tab:orange', 'marker': '^', 'linestyle': '-'},
    'flip_rotation_crop':       {'color': 'tab:green',  'marker': 'D', 'linestyle': '-'},
    'flip_color_cutmix':        {'color': 'tab:purple', 'marker': '*', 'linestyle': '-'},
    'rotation_color_crop':      {'color': 'tab:pink',   'marker': 'P', 'linestyle': '-'},
    'flip_rotation_color':      {'color': 'tab:red',    'marker': 'v', 'linestyle': '-'},
    'flip_rotation_cutmix':     {'color': 'tab:gray',   'marker': '<', 'linestyle': '-'},
    'flip_crop_cutmix':         {'color': 'tab:olive',  'marker': '>', 'linestyle': '-'},
    'color_crop_cutmix':        {'color': 'tab:cyan',   'marker': 'o', 'linestyle': '-'},

    # _______________________________________________________________________________
    # Four transformations
    # _______________________________________________________________________________
    #write your code here:
    'flip_rotation_crop_cutmix':    {'color': 'tab:blue',   'marker': 's', 'linestyle': '-'},
    'flip_color_crop_cutmix':       {'color': 'tab:brown',  'marker': 'X', 'linestyle': '-'},
    'rotation_color_crop_cutmix':   {'color': 'tab:orange', 'marker': '^', 'linestyle': '-'},
    'flip_rotation_color_crop':     {'color': 'tab:green',  'marker': 'D', 'linestyle': '-'},
    'flip_rotation_color_cutmix':   {'color': 'tab:purple', 'marker': '*', 'linestyle': '-'},
    

    # _______________________________________________________________________________
    # Five transformations
    # _______________________________________________________________________________
    #write your code here:
    'flip_rotation_color_crop_cutmix': {'color': 'tab:pink',   'marker': 'P', 'linestyle': '-'},

    # _______________________________________________________________________________
    # Smart transformations
    # _______________________________________________________________________________
    #write your code here:
    'auto_augment':  {'color': 'tab:cyan',   'marker': 'o', 'linestyle': '-'},
}


def plot_experiment(metrics, percentage, combo_name):
    """
    Plots training loss, validation loss, and validation accuracy
    for ONE experiment (3 side-by-side graphs).

    Args:
        metrics:    [train_losses, val_losses, val_accuracies]
        percentage: 0.10, 0.25, 0.50, or 1.0
        combo_name: e.g. 'none', 'flip', 'full'
    """
    train_losses, val_losses, val_accuracies = metrics
    epochs = range(1, len(train_losses) + 1)
    title  = f"{int(percentage*100)}% Training Data — Combo: {combo_name}"
    style  = COMBO_STYLES[combo_name]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: Training Loss
    axes[0].plot(epochs, train_losses,
                 color=style['color'], linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Plot 2: Validation Loss
    axes[1].plot(epochs, val_losses,
                 color=style['color'], linewidth=2)
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    # Plot 3: Validation Accuracy
    axes[2].plot(epochs, val_accuracies,
                 color=style['color'], linewidth=2)
    axes[2].set_title('Validation Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].grid(True)

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    filename = f"plots/{int(percentage*100)}pct_{combo_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()


def plot_all_combos_per_percentage(all_metrics, all_results,
                                    percentages, combinations):
    """
    For each data percentage, plots ALL 6 combinations
    on the same accuracy curve — so you can compare them.

    Produces 4 graphs (one per percentage).

    Args:
        all_metrics:  dict of metrics e.g. all_metrics['10%_none']
        all_results:  dict of best accuracy e.g. all_results['10%_none']
        percentages:  [0.10, 0.25, 0.50, 1.0]
        combinations: list of combo names
    """
    for percentage in percentages:
        p = int(percentage * 100)

        fig, ax = plt.subplots(figsize=(10, 5))

        for combo in combinations:
            key   = f"{p}%_{combo}"
            style = COMBO_STYLES[combo]
            _, _, val_accuracies = all_metrics[key]
            epochs = range(1, len(val_accuracies) + 1)
            best   = all_results[key]

            ax.plot(epochs, val_accuracies,
                    color=style['color'],
                    linewidth=2,
                    label=f"{combo} (best={best:.1f}%)")

        ax.set_title(
            f'All Combinations — {p}% Training Data',
            fontsize=13, fontweight='bold'
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        filename = f"plots/all_combos_{p}pct.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()


def plot_final_summary(all_results, percentages, combinations):
    """
    THE main result graph for your report!
    Shows accuracy vs data percentage for ALL 6 combinations.

    Args:
        all_results:  dict e.g. {'10%_none': 45.2, '10%_flip': 47.5 ...}
        percentages:  [0.10, 0.25, 0.50, 1.0]
        combinations: list of combo names
    """
    pct_labels = [int(p * 100) for p in percentages]

    plt.figure(figsize=(10, 6))

    for combo in combinations:
        style      = COMBO_STYLES[combo]
        accuracies = [
            all_results[f"{int(p*100)}%_{combo}"]
            for p in percentages
        ]
        plt.plot(pct_labels, accuracies,
                 color=style['color'],
                 linewidth=2,
                 marker=style['marker'],
                 markersize=8,
                 label=combo)

    plt.title(
        'Accuracy vs Training Data Size\n'
        'Effect of Different Augmentation Combinations',
        fontsize=13, fontweight='bold'
    )
    plt.xlabel('Training Data Percentage (%)', fontsize=11)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=11)
    plt.xticks(pct_labels, ['10%', '25%', '50%', '100%'])
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/final_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/final_summary.png")
    plt.show()


def plot_bar_per_percentage(all_results, percentages, combinations):
    """
    Bar chart for each percentage showing all 6 combinations.
    Easy to compare which combination wins at each data size.

    Produces 4 bar charts (one per percentage).

    Args:
        all_results:  dict of best accuracies
        percentages:  [0.10, 0.25, 0.50, 1.0]
        combinations: list of combo names
    """
    colors = [COMBO_STYLES[c]['color'] for c in combinations]

    for percentage in percentages:
        p          = int(percentage * 100)
        accuracies = [all_results[f"{p}%_{c}"] for c in combinations]

        plt.figure(figsize=(9, 5))
        bars = plt.bar(combinations, accuracies,
                       color=colors, alpha=0.85,
                       edgecolor='black')

        # Accuracy label on top of each bar
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{acc:.1f}%',
                ha='center', fontsize=9, fontweight='bold'
            )

        plt.title(
            f'Accuracy per Combination — {p}% Training Data',
            fontsize=13, fontweight='bold'
        )
        plt.xlabel('Augmentation Combination', fontsize=11)
        plt.ylabel('Best Validation Accuracy (%)', fontsize=11)
        plt.xticks(rotation=20, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        os.makedirs('plots', exist_ok=True)
        filename = f"plots/bar_{p}pct.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()


"""
After experiment 1  → saves plots/10pct_none.png
After experiment 2  → saves plots/10pct_flip.png
After experiment 3  → saves plots/10pct_flip_rotation.png
After experiment 4  → saves plots/10pct_flip_rotation_color.png
After experiment 5  → saves plots/10pct_flip_rotation_color_crop.png
After experiment 6  → saves plots/10pct_full.png

After experiment 7  → saves plots/25pct_none.png
After experiment 8  → saves plots/25pct_flip.png
...and so on

Each plot shows 3 graphs side by side:
┌─────────────────┬─────────────────┬─────────────────┐
│   Train Loss    │   Val Loss      │   Val Accuracy  │
│  (goes down ↓)  │  (goes down ↓)  │  (goes up ↑)    │
└─────────────────┴─────────────────┴─────────────────┘

PLUS after all 24 experiments finish:
plot_all_combos_per_percentage() → 4 more graphs
plot_bar_per_percentage()        → 4 more graphs
plot_final_summary()             → 1 main graph
─────────────────────────────────────────────────
Total plots: 24 + 4 + 4 + 1 = 33 plots! 
"""