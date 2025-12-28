
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.synthesis.synth_os import generate_all_patterns, normalize_density, gen_uniform, gen_random, gen_clustered, gen_linear, gen_radial, gen_multi_nuclear
from src.analysis.lacunarity import LacunarityAnalyzer
from src.analysis.multifractal import MultifractalAnalyzer
from src.analysis.os_indicators import OSIndicatorCalculator

# Set font for Japanese support
plt.rcParams['font.family'] = ['sans-serif']
# Try common Japanese fonts
for font in ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'TakaoGothic', 'IPAGothic']:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        break
    except:
        continue

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_pattern_from_config(cfg, H, W, density, rng):
    ptype = cfg["type"]
    if ptype == "uniform":
        return gen_uniform(H, W, density, rng)
    elif ptype == "random":
        return gen_random(H, W, density, rng)
    elif ptype == "clustered":
        return gen_clustered(
            H, W, density, rng,
            n_clusters=cfg.get("n_clusters", 4),
            cluster_size=cfg.get("cluster_size", 0.15)
        )
    elif ptype == "linear":
        return gen_linear(
            H, W, density, rng,
            n_lines=cfg.get("n_lines", 3),
            line_width=cfg.get("line_width", 0.1)
        )
    elif ptype == "radial":
        return gen_radial(
            H, W, density, rng,
            n_rings=cfg.get("n_rings", 3)
        )
    elif ptype == "multi_nuclear":
        return gen_multi_nuclear(
            H, W, density, rng,
            n_nuclei=cfg.get("n_nuclei", 3),
            nucleus_size=cfg.get("nucleus_size", 0.12)
        )
    else:
        raise ValueError(f"Unknown pattern type: {ptype}")

def make_chart_a(config, output_dir, timestamp):
    print("Generating Chart A...")
    rows = config["chart_a"]["rows"]
    H = config["parameters"]["resolution"]
    W = config["parameters"]["resolution"]
    density = config["parameters"]["density"]
    
    # Setup layout: 3 rows, 5 cols (4 images + 1 plot)
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 1.5])
    
    rng = np.random.default_rng(42)
    analyzer = LacunarityAnalyzer(r_min=2, r_max=128, r_steps=30, full_scan=False)
    
    for row_idx, row_cfg in enumerate(rows):
        variations = row_cfg["variations"]
        curves = []
        
        for col_idx, var in enumerate(variations):
            # Generate image
            img = generate_pattern_from_config(var["params"], H, W, density, rng)
            
            # Analyze
            df, _ = analyzer.analyze(img)
            curves.append({"name": var["name"], "data": df})
            
            # Plot Image
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img, cmap="Greys", interpolation="nearest")
            ax.set_title(f"{var['name']}\nL={df.iloc[10]['lambda']:.2f} (r={df.iloc[10]['r']})", fontsize=10)
            ax.axis("off")
            
        # Plot Curve
        ax_curve = fig.add_subplot(gs[row_idx, 4])
        for curve in curves:
            df = curve["data"]
            ax_curve.plot(np.log(df["r"]), np.log(df["lambda"]), label=curve["name"], linewidth=2)
        
        ax_curve.set_xlabel("log(r)")
        ax_curve.set_ylabel("log($\\Lambda(r)$)")
        ax_curve.set_title(f"{row_cfg['label']} - Lacunarity Curves")
        ax_curve.legend(fontsize=8)
        ax_curve.grid(True, alpha=0.3)
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"ChartA_Lacunarity_{timestamp}.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Chart A saved to {outfile}")

def make_chart_b(config, output_dir, timestamp):
    print("Generating Chart B...")
    variations = config["chart_b"]["variations"]
    H = config["parameters"]["resolution"]
    W = config["parameters"]["resolution"]
    density = config["parameters"]["density"]
    
    # Layout: 2 rows of 3 images (left) + 1 large plot (right)
    # Actually user said: 2x3 Grid + Right Dq spectrum
    # Let's do: Left side (2x3 grid), Right side (1 column with Dq and f(alpha))
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.5])
    
    rng = np.random.default_rng(42)
    mfa_analyzer = MultifractalAnalyzer(r_min=2, r_max=64, q_min=-5, q_max=5, q_steps=21)
    
    results = []
    
    for i, var in enumerate(variations):
        row = i // 3
        col = i % 3
        
        # Generate
        img = generate_pattern_from_config(var["params"], H, W, density, rng)
        
        # Analyze
        df, _ = mfa_analyzer.analyze(img)
        results.append({"name": var["name"], "data": df})
        
        # Plot Image
        if row < 2 and col < 3: # Safety check
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap="Greys", interpolation="nearest")
            ax.set_title(var["name"])
            ax.axis("off")
            
    # Plot Dq Spectrum
    ax_dq = fig.add_subplot(gs[0, 3])
    for res in results:
        df = res["data"]
        # Dq = tau(q) / (q-1) usually, but code returns tau.
        # Dq = tau / (q-1). For q=1, D1 = alpha(1) = entropy dimension.
        # Let's check multifractal.py implementation.
        # It returns q, alpha, f_alpha, tau.
        # Dq is not explicitly returned but can be calculated.
        # Generalized dimensions: D_q = \tau(q) / (q-1)
        q = df["q"]
        tau = df["tau"]
        # Handle q=1 singularity
        Dq = tau / (q - 1)
        # Fix q=1 case (limit)
        idx_1 = np.argmin(np.abs(q - 1.0))
        if abs(q[idx_1] - 1.0) < 1e-5:
             Dq[idx_1] = df["alpha"][idx_1] # D1 = alpha(1) = f(alpha(1))
             
        ax_dq.plot(q, Dq, label=res["name"])
        
    ax_dq.set_xlabel("q (moment)")
    ax_dq.set_ylabel("$D_q$")
    ax_dq.set_title("Generalized Dimensions spectrum")
    ax_dq.legend(fontsize=8)
    ax_dq.grid(True)
    
    # Plot f(alpha)
    ax_fa = fig.add_subplot(gs[1, 3])
    for res in results:
        df = res["data"]
        ax_fa.plot(df["alpha"], df["f_alpha"], label=res["name"])
        
    ax_fa.set_xlabel("$\\alpha$ (singularity strength)")
    ax_fa.set_ylabel("$f(\\alpha)$ (fractal dimension)")
    ax_fa.set_title("Multifractal Spectrum")
    ax_fa.grid(True)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"ChartB_MFA_{timestamp}.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Chart B saved to {outfile}")

def make_chart_c(config, output_dir, timestamp):
    print("Generating Chart C...")
    n_samples = config["chart_c"]["n_samples"]
    H = config["parameters"]["resolution"]
    W = config["parameters"]["resolution"]
    density = config["parameters"]["density"]
    ranges = config["chart_c"]["parameter_ranges"]
    
    rng = np.random.default_rng(123)
    
    data = []
    
    # Create indicators calculator
    # Reduce sampling for speed in batch
    calc = OSIndicatorCalculator(
        lac_r_max=64, lac_r_steps=10, lac_r_extract=16,
        mfa_r_max=32, mfa_q_min=-3, mfa_q_max=3, # Narrower for speed
        perc_d_max=100, perc_d_steps=10
    )
    
    print(f"Generating {n_samples} samples...")
    for i in tqdm(range(n_samples)):
        # Randomly choose a pattern type and parameters
        ptype = rng.choice(["clustered", "linear", "multi_nuclear", "random"])
        
        params = {"type": ptype}
        if ptype == "clustered":
            params["n_clusters"] = rng.integers(ranges["n_clusters"][0], ranges["n_clusters"][1])
            params["cluster_size"] = rng.uniform(ranges["cluster_size"][0], ranges["cluster_size"][1])
        elif ptype == "linear":
            params["n_lines"] = rng.integers(ranges["n_lines"][0], ranges["n_lines"][1])
            params["line_width"] = rng.uniform(ranges["line_width"][0], ranges["line_width"][1])
        elif ptype == "multi_nuclear":
            params["n_nuclei"] = rng.integers(ranges["n_nuclei"][0], ranges["n_nuclei"][1])
            params["nucleus_size"] = rng.uniform(ranges["nucleus_size"][0], ranges["nucleus_size"][1])
        # random type has no params to vary in current implementation other than seed
            
        try:
            img = generate_pattern_from_config(params, H, W, density, rng)
            
            # Compute indicators
            inds = calc.compute_all_os_indicators(img)
            
            entry = {
                "id": i,
                "type": ptype,
                "image": img,
                "lacunarity": inds["lacunarity_r16"],
                "alpha_width": inds["alpha_width"],
                "r_crit": inds["r_crit_px"]
            }
            data.append(entry)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue
            
    df = pd.DataFrame(data)
    
    # Plot 3D scatter projected to 2D
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    
    ax_scatter = fig.add_subplot(gs[0])
    
    # Scatter plot
    scatter = ax_scatter.scatter(
        df["lacunarity"], df["alpha_width"], 
        c=df["r_crit"], cmap="viridis", s=50, alpha=0.7
    )
    plt.colorbar(scatter, ax=ax_scatter, label="Critical Radius (Percolation)")
    
    ax_scatter.set_xlabel("Lacunarity (r=16)")
    ax_scatter.set_ylabel("MFA Alpha Width")
    ax_scatter.set_title("Pattern Space: Lacunarity vs MFA vs Percolation")
    
    # Annotate extremes
    # Find points with max/min of each metric
    extremes = []
    for metric in ["lacunarity", "alpha_width", "r_crit"]:
        extremes.append(df.iloc[df[metric].idxmax()])
        extremes.append(df.iloc[df[metric].idxmin()])
        
    # Remove duplicates
    seen = set()
    unique_extremes = []
    for e in extremes:
        if e["id"] not in seen:
            unique_extremes.append(e)
            seen.add(e["id"])
            
    # Plot representative images on the right
    # 3x2 grid on the right side? No, simpler: just select 4-5 key points
    
    ax_scatter.grid(True, linestyle="--", alpha=0.5)
    
    # Label points on scatter
    for i, e in enumerate(unique_extremes):
        ax_scatter.text(e["lacunarity"], e["alpha_width"], str(i+1), fontsize=12, fontweight="bold")
        
    # Representative Images
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1])
    for i, e in enumerate(unique_extremes[:6]): # Limit to 6
        ax_img = fig.add_subplot(gs_right[i // 2, i % 2])
        ax_img.imshow(e["image"], cmap="Greys", interpolation="nearest")
        ax_img.set_title(f"#{i+1} {e['type']}\nL={e['lacunarity']:.1f}, W={e['alpha_width']:.2f}", fontsize=8)
        ax_img.axis("off")
        
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"ChartC_Space_{timestamp}.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Chart C saved to {outfile}")

def main():
    config_path = "251227/OS/configs/paper_refactored.yaml"
    config = load_config(config_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["project"]["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting analysis... Output: {output_dir}")
    
    make_chart_a(config, output_dir, timestamp)
    make_chart_b(config, output_dir, timestamp)
    # make_chart_c(config, output_dir, timestamp) # Chart C takes time, run if needed
    
    # Run Chart C with fewer samples for demo if user wants "all"
    make_chart_c(config, output_dir, timestamp)

if __name__ == "__main__":
    main()

