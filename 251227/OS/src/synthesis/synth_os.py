"""Synthetic pattern generation for JRSI Figure 1.

集落構造の原型となる6つのパターンを生成:
1. Uniform (均一分布) - グリッド状街区
2. Random (ランダム分布) - ポアソン分布
3. Clustered (クラスター分布) - 複数のクラスター
4. Linear (線状分布) - 道路沿い
5. Radial (放射状・同心円分布) - 中心から放射状
6. Multi-Nuclear (多核分布) - 複数の中心
"""

import numpy as np
from tqdm import tqdm


def normalize_density(pattern: np.ndarray, target_density: float, rng: np.random.Generator) -> np.ndarray:
    """密度を目標値に調整する。
    
    Args:
        pattern: 2D binary array (0/1)
        target_density: 目標密度 (0-1)
        rng: Random number generator
        
    Returns:
        密度調整後のパターン
    """
    pattern = pattern.copy().astype(float)
    current_density = pattern.sum() / pattern.size
    
    if abs(current_density - target_density) < 1e-6:
        return pattern
    
    if current_density > target_density:
        # ランダムに削除
        indices = np.where(pattern == 1)
        n_remove = int((current_density - target_density) * pattern.size)
        if n_remove > 0 and len(indices[0]) > 0:
            remove_idx = rng.choice(len(indices[0]), min(n_remove, len(indices[0])), replace=False)
            pattern[indices[0][remove_idx], indices[1][remove_idx]] = 0
    else:
        # ランダムに追加
        indices = np.where(pattern == 0)
        n_add = int((target_density - current_density) * pattern.size)
        if n_add > 0 and len(indices[0]) > 0:
            add_idx = rng.choice(len(indices[0]), min(n_add, len(indices[0])), replace=False)
            pattern[indices[0][add_idx], indices[1][add_idx]] = 1
    
    return pattern


def gen_uniform(H: int, W: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform (均一分布) パターンを生成 - グリッド状街区。
    
    計画都市のグリッド状街区を模倣。規則的な配置。
    OS指標で識別: 非常に低いラクナリティ（均一性が高い）
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        
    Returns:
        2D binary array
    """
    # 完全に規則的なグリッド（点配置）
    grid_spacing = int(np.sqrt(1.0 / density))
    pattern = np.zeros((H, W))
    
    # 規則的なグリッド配置（各セルに1点）
    for i in range(grid_spacing // 2, H, grid_spacing):
        for j in range(grid_spacing // 2, W, grid_spacing):
            if i < H and j < W:
                pattern[i, j] = 1
    
    # 密度調整
    return normalize_density(pattern, density, rng)


def gen_random(H: int, W: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Random (ランダム分布) パターンを生成 - ポアソン分布。
    
    自然発生的な村落や歴史的な小規模集落の家屋配置を模倣。
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        
    Returns:
        2D binary array
    """
    pattern = (rng.random((H, W)) < density).astype(float)
    return pattern


def gen_clustered(
    H: int, W: int, density: float, rng: np.random.Generator,
    n_clusters: int = 3, cluster_size: float = 0.12
) -> np.ndarray:
    """Clustered (クラスター分布) パターンを生成 - 複数のクラスター。
    
    商業中心地（CBD）周辺のビル群や住宅団地を模倣。
    複数の小さな群が散在するパターン。
    OS指標で識別: 高いラクナリティ（クラスター間の大きな空隙）
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        n_clusters: クラスタ数
        cluster_size: クラスタの相対サイズ（画像サイズに対する割合）
        
    Returns:
        2D binary array
    """
    pattern = np.zeros((H, W))
    max_r = min(H, W) * cluster_size
    
    # クラスタ中心を生成（境界から離す）
    margin = max_r * 2.5
    # マージンが大きすぎる場合は調整
    if margin * 2 >= min(H, W):
        margin = min(H, W) / 2 - 1
        
    centers = []
    for _ in range(n_clusters):
        attempts = 0
        while attempts < 100:
            cx = rng.uniform(margin, W - margin)
            cy = rng.uniform(margin, H - margin)
            # 既存の中心から十分離れているか確認
            too_close = False
            for ocx, ocy in centers:
                if np.sqrt((cx - ocx)**2 + (cy - ocy)**2) < margin:
                    too_close = True
                    break
            if not too_close:
                centers.append((cx, cy))
                break
            attempts += 1
    
    # 各クラスタにポイントを配置（密にクラスター化）
    n_points_per_cluster = int((H * W * density) / len(centers))
    
    for cx, cy in centers:
        for _ in range(n_points_per_cluster):
            # ガウシアン分布でクラスタ中心周辺に配置
            angle = rng.uniform(0, 2 * np.pi)
            r = abs(rng.normal(0, max_r / 2.5))
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            
            if 0 <= y < H and 0 <= x < W:
                pattern[y, x] = 1
    
    # 密度を正確に調整
    return normalize_density(pattern, density, rng)


def gen_linear(
    H: int, W: int, density: float, rng: np.random.Generator,
    n_lines: int = 2, line_width: float = 0.05
) -> np.ndarray:
    """Linear (線状分布) パターンを生成 - 道路沿い。
    
    河川沿いの街や高速道路沿いの商業施設を模倣。
    リボン状（帯状）の街区形成。
    OS指標で識別: 非常に高いラクナリティ（線状構造による大きな異質性）
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        n_lines: 線の数
        line_width: 線の幅（画像サイズに対する割合）
        
    Returns:
        2D binary array
    """
    pattern = np.zeros((H, W))
    line_pixels = max(2, int(min(H, W) * line_width))
    
    # 明確な線状構造（縦横の組み合わせ）
    # 縦線
    for i in range(n_lines):
        x = int(W * (i + 1) / (n_lines + 1))
        x_start = max(0, x - line_pixels // 2)
        x_end = min(W, x + line_pixels // 2)
        # 線に沿って点を配置
        n_line_points = int(H * W * density * 0.4 / n_lines)
        for _ in range(n_line_points):
            y = rng.integers(0, H)
            pattern[y, x_start:x_end] = 1
    
    # 横線
    for i in range(n_lines):
        y = int(H * (i + 1) / (n_lines + 1))
        y_start = max(0, y - line_pixels // 2)
        y_end = min(H, y + line_pixels // 2)
        n_line_points = int(H * W * density * 0.3 / n_lines)
        for _ in range(n_line_points):
            x = rng.integers(0, W)
            pattern[y_start:y_end, x] = 1
    
    # 密度を正確に調整
    return normalize_density(pattern, density, rng)


def gen_radial(
    H: int, W: int, density: float, rng: np.random.Generator,
    n_rings: int = 3
) -> np.ndarray:
    """Radial (放射状・同心円分布) パターンを生成 - 中心から放射状。
    
    パリやモスクワのような歴史都市のリングロード構造を模倣。
    中心（CBD）→商業→住宅→郊外の層状構造。
    OS指標で識別: 中程度のラクナリティ（同心円構造）
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        n_rings: 同心円の数
        
    Returns:
        2D binary array
    """
    pattern = np.zeros((H, W))
    cy, cx = H / 2, W / 2
    max_r = np.sqrt(H**2 + W**2) / 2
    
    # 明確な同心円構造
    ring_radii = np.linspace(max_r * 0.2, max_r * 0.8, n_rings)
    
    for ring_r in ring_radii:
        # 各リングにポイントを配置
        n_points_in_ring = int((H * W * density) / (n_rings + 1))
        for _ in range(n_points_in_ring):
            angle = rng.uniform(0, 2 * np.pi)
            r = ring_r + rng.normal(0, ring_r * 0.1)
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            
            if 0 <= y < H and 0 <= x < W:
                pattern[y, x] = 1
    
    # 中心部にも配置
    center_points = int((H * W * density) / (n_rings + 1))
    for _ in range(center_points):
        r = abs(rng.normal(0, max_r * 0.15))
        angle = rng.uniform(0, 2 * np.pi)
        y = int(cy + r * np.sin(angle))
        x = int(cx + r * np.cos(angle))
        if 0 <= y < H and 0 <= x < W:
            pattern[y, x] = 1
    
    # 密度を正確に調整
    return normalize_density(pattern, density, rng)


def gen_multi_nuclear(
    H: int, W: int, density: float, rng: np.random.Generator,
    n_nuclei: int = 3, nucleus_size: float = 0.12
) -> np.ndarray:
    """Multi-Nuclear (多核分布) パターンを生成 - 複数の中心。
    
    ロサンゼルスや東京のような大都市圏を模倣。
    複数のサブセンター（新宿、渋谷など）が存在するポリセントリック都市。
    OS指標で識別: 中程度のラクナリティ（複数核による構造）
    
    Args:
        H: 高さ
        W: 幅
        density: 密度
        rng: Random number generator
        n_nuclei: 核の数
        nucleus_size: 核の相対サイズ
        
    Returns:
        2D binary array
    """
    pattern = np.zeros((H, W))
    max_r = min(H, W) * nucleus_size
    margin = max_r * 2.5
    
    # 複数の核（中心）を生成
    nuclei = []
    
    # マージンが大きすぎる場合は調整
    if margin * 2 >= min(H, W):
        margin = min(H, W) / 2 - 1

    for _ in range(n_nuclei):
        attempts = 0
        while attempts < 100:
            cx = rng.uniform(margin, W - margin)
            cy = rng.uniform(margin, H - margin)
            
            # 既存の核から十分離れているか確認
            too_close = False
            for nx, ny in nuclei:
                if np.sqrt((cx - nx)**2 + (cy - ny)**2) < margin:
                    too_close = True
                    break
            
            if not too_close:
                nuclei.append((cx, cy))
                break
            attempts += 1
    
    # 各核にポイントを配置
    n_points_per_nucleus = int((H * W * density) / len(nuclei))
    
    for cx, cy in nuclei:
        for _ in range(n_points_per_nucleus):
            # ガウシアン分布で核中心周辺に配置
            angle = rng.uniform(0, 2 * np.pi)
            r = abs(rng.normal(0, max_r / 2.5))
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            
            if 0 <= y < H and 0 <= x < W:
                pattern[y, x] = 1
    
    # 密度を正確に調整
    return normalize_density(pattern, density, rng)


def generate_all_patterns(
    H: int, W: int, target_density: float, random_seed: int = 42,
    pattern_configs: list[dict] | None = None
) -> dict[str, np.ndarray]:
    """全パターンを生成。
    
    Args:
        H: 高さ
        W: 幅
        target_density: 目標密度
        random_seed: 乱数シード
        pattern_configs: パターン設定のリスト
        
    Returns:
        パターン名をキー、パターン配列を値とする辞書
    """
    rng = np.random.default_rng(random_seed)
    
    if pattern_configs is None:
        pattern_configs = [
            {"name": "uniform", "type": "uniform"},
            {"name": "random", "type": "random"},
            {"name": "clustered", "type": "clustered", "n_clusters": 4, "cluster_size": 0.15},
            {"name": "linear", "type": "linear", "n_lines": 3, "line_width": 0.1},
            {"name": "radial", "type": "radial", "n_rings": 3},
            {"name": "multi_nuclear", "type": "multi_nuclear", "n_nuclei": 3, "nucleus_size": 0.12},
        ]
    
    patterns = {}
    
    for cfg in tqdm(pattern_configs, desc="Generating patterns"):
        ptype = cfg["type"]
        name = cfg.get("name", ptype)
        
        if ptype == "uniform":
            pattern = gen_uniform(H, W, target_density, rng)
        elif ptype == "random":
            pattern = gen_random(H, W, target_density, rng)
        elif ptype == "clustered":
            pattern = gen_clustered(
                H, W, target_density, rng,
                n_clusters=cfg.get("n_clusters", 4),
                cluster_size=cfg.get("cluster_size", 0.15)
            )
        elif ptype == "linear":
            pattern = gen_linear(
                H, W, target_density, rng,
                n_lines=cfg.get("n_lines", 3),
                line_width=cfg.get("line_width", 0.1)
            )
        elif ptype == "radial":
            pattern = gen_radial(
                H, W, target_density, rng,
                n_rings=cfg.get("n_rings", 3)
            )
        elif ptype == "multi_nuclear":
            pattern = gen_multi_nuclear(
                H, W, target_density, rng,
                n_nuclei=cfg.get("n_nuclei", 3),
                nucleus_size=cfg.get("nucleus_size", 0.12)
            )
        else:
            raise ValueError(f"Unknown pattern type: {ptype}")
        
        patterns[name] = pattern
    
    return patterns


