import os
import json
import math
import random
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


# ============================================================
# 1. 설정
# ============================================================

CONFIG = {
    "image_size": 512,
    "num_images": 300,
    "output_dir": "synthetic_sem_dataset_v3",

    "make_zip": True,
    "zip_name": "synthetic_sem_dataset_v3.zip",

    # particle 설정
    "num_particles_range": (150, 260),
    "particle_radius_range": (3, 20),
    "large_particle_prob": 0.08,

    # clustering 설정
    "cluster_prob": 0.55,
    "num_clusters_range": (4, 10),
    "cluster_spread_range": (18, 55),

    # anisotropy 설정
    "anisotropy_prob": 0.45,
    "anisotropy_strength_range": (0.4, 0.85),

    # agglomerate 설정
    "agglomerate_prob": 0.75,
    "num_agglomerates_range": (3, 8),
    "particles_per_agglomerate_range": (12, 40),
    "agglomerate_spread_range": (8, 22),

    # 기존 random network
    "network_prob": 0.75,
    "num_networks_range": (1, 3),
    "network_radius_range": (4, 13),

    # percolation mode 설정
    "percolation_modes": ["low", "medium", "high"],
    "percolation_mode_probs": [0.33, 0.34, 0.33],

    # mode별 좌우 관통 path 확률
    "force_percolation_prob_low": 0.05,
    "force_percolation_prob_medium": 0.45,
    "force_percolation_prob_high": 0.90,

    # 좌우 관통 filler path 설정
    "percolation_path_particles_range": (45, 90),
    "percolation_path_radius_range": (6, 14),
    "percolation_branch_prob": 0.35,

    # void 설정
    "void_prob": 0.75,
    "num_voids_range": (3, 12),
    "void_radius_range": (8, 30),

    # noise 설정
    "gaussian_noise_std": 10,
    "blur_radius": 0.7,
    "texture_strength": 35,
}


# ============================================================
# 2. 유틸 함수
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clamp(v, a, b):
    return max(a, min(v, b))


def random_radius(r_min, r_max, large_prob):
    if random.random() < large_prob:
        r = random.uniform(r_max * 0.7, r_max * 1.6)
    else:
        r = np.random.lognormal(
            mean=math.log((r_min + r_max) / 4),
            sigma=0.45
        )

    return float(clamp(r, r_min, r_max * 1.6))


# ============================================================
# 3. Percolation mode 선택
# ============================================================

def choose_percolation_mode(cfg):
    return random.choices(
        cfg["percolation_modes"],
        weights=cfg["percolation_mode_probs"],
        k=1
    )[0]


# ============================================================
# 4. 기본 particle 생성
# random + clustering + anisotropy
# ============================================================

def generate_base_particles(cfg):
    W = cfg["image_size"]
    particles = []

    n = random.randint(*cfg["num_particles_range"])

    use_cluster = random.random() < cfg["cluster_prob"]
    use_aniso = random.random() < cfg["anisotropy_prob"]

    theta = random.uniform(0, math.pi)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    strength = random.uniform(*cfg["anisotropy_strength_range"])

    clusters = []
    if use_cluster:
        for _ in range(random.randint(*cfg["num_clusters_range"])):
            clusters.append((
                random.uniform(0.1 * W, 0.9 * W),
                random.uniform(0.1 * W, 0.9 * W)
            ))

    for _ in range(n):
        r = random_radius(
            cfg["particle_radius_range"][0],
            cfg["particle_radius_range"][1],
            cfg["large_particle_prob"]
        )

        if use_cluster and random.random() < 0.75:
            cx, cy = random.choice(clusters)
            spread = random.uniform(*cfg["cluster_spread_range"])

            if use_aniso:
                a = np.random.normal(0, spread * (1 + 2 * strength))
                b = np.random.normal(0, spread * (1 - 0.6 * strength))
                pos = np.array([cx, cy]) + a * direction + b * normal
                x, y = pos
            else:
                x = np.random.normal(cx, spread)
                y = np.random.normal(cy, spread)
        else:
            x = random.uniform(-r, W + r)
            y = random.uniform(-r, W + r)

        particles.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "type": "base"
        })

    return particles


# ============================================================
# 5. Agglomerate 생성
# ============================================================

def add_agglomerates(particles, cfg):
    W = cfg["image_size"]

    if random.random() > cfg["agglomerate_prob"]:
        return particles

    n_agglomerates = random.randint(*cfg["num_agglomerates_range"])

    for _ in range(n_agglomerates):
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * W, 0.95 * W)

        n_local = random.randint(*cfg["particles_per_agglomerate_range"])
        spread = random.uniform(*cfg["agglomerate_spread_range"])

        for _ in range(n_local):
            r = random.uniform(3, 11)

            x = np.random.normal(cx, spread)
            y = np.random.normal(cy, spread)

            particles.append({
                "x": float(x),
                "y": float(y),
                "r": float(r),
                "type": "agglomerate"
            })

    return particles


# ============================================================
# 6. 좌우 관통 filler path 생성
# ============================================================

def generate_percolating_path(cfg, mode):
    """
    좌측에서 우측까지 이어지는 filler chain 생성.
    high mode에서는 거의 항상 연결된 네트워크가 생기도록 설계.
    """
    W = cfg["image_size"]
    particles = []

    if mode == "high":
        n_path = random.randint(70, 100)
        jitter_scale = random.uniform(5, 12)
        radius_boost = 1.25
        branch_prob = 0.45
    elif mode == "medium":
        n_path = random.randint(48, 75)
        jitter_scale = random.uniform(10, 22)
        radius_boost = 1.0
        branch_prob = 0.30
    else:
        n_path = random.randint(28, 48)
        jitter_scale = random.uniform(18, 35)
        radius_boost = 0.85
        branch_prob = 0.15

    x_values = np.linspace(-8, W + 8, n_path)

    y0 = random.uniform(0.2 * W, 0.8 * W)
    amp = random.uniform(15, 60)
    freq = random.uniform(1.0, 2.3)
    phase_shift = random.uniform(0, 2 * math.pi)

    for i, x in enumerate(x_values):
        t = i / max(1, n_path - 1)

        y = (
            y0
            + amp * math.sin(2 * math.pi * freq * t + phase_shift)
            + np.random.normal(0, jitter_scale)
        )

        y = clamp(y, -20, W + 20)

        r = random.uniform(*cfg["percolation_path_radius_range"]) * radius_boost

        particles.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "type": f"percolation_{mode}"
        })

        # branch 생성
        if random.random() < branch_prob:
            branch_len = random.randint(3, 10)
            branch_angle = random.choice([
                random.uniform(math.pi / 4, 3 * math.pi / 4),
                random.uniform(-3 * math.pi / 4, -math.pi / 4)
            ])

            bx, by = x, y

            for _ in range(branch_len):
                step = random.uniform(8, 18)
                bx += step * math.cos(branch_angle)
                by += step * math.sin(branch_angle)

                br = random.uniform(3, 9)

                particles.append({
                    "x": float(bx),
                    "y": float(by),
                    "r": float(br),
                    "type": f"branch_{mode}"
                })

    return particles


# ============================================================
# 7. Filler network 생성
# low / medium / high 골고루 생성
# ============================================================

def generate_network_particles(cfg, mode):
    W = cfg["image_size"]
    particles = []

    if mode == "high":
        force_prob = cfg["force_percolation_prob_high"]
    elif mode == "medium":
        force_prob = cfg["force_percolation_prob_medium"]
    else:
        force_prob = cfg["force_percolation_prob_low"]

    # 강제 좌우 관통 path
    if random.random() < force_prob:
        particles += generate_percolating_path(cfg, mode)

    # 기존 random walk network도 추가
    if random.random() < cfg["network_prob"]:
        n_networks = random.randint(*cfg["num_networks_range"])

        for _ in range(n_networks):
            x = random.uniform(0, W)
            y = random.uniform(0, W)
            angle = random.uniform(0, 2 * math.pi)

            if mode == "high":
                n_steps = random.randint(55, 90)
                step_range = (6, 16)
            elif mode == "medium":
                n_steps = random.randint(35, 65)
                step_range = (8, 20)
            else:
                n_steps = random.randint(15, 40)
                step_range = (12, 28)

            for _ in range(n_steps):
                r = random.uniform(*cfg["network_radius_range"])

                particles.append({
                    "x": float(x),
                    "y": float(y),
                    "r": float(r),
                    "type": f"network_{mode}"
                })

                angle += np.random.normal(0, 0.45)
                step = random.uniform(*step_range)

                x += step * math.cos(angle)
                y += step * math.sin(angle)

                if x < -40 or x > W + 40 or y < -40 or y > W + 40:
                    break

    return particles


# ============================================================
# 8. Void 생성
# ============================================================

def generate_voids(cfg, mode):
    W = cfg["image_size"]
    voids = []

    if random.random() > cfg["void_prob"]:
        return voids

    # high percolation에서는 void가 너무 많이 path를 끊지 않게 살짝 줄임
    if mode == "high":
        n_voids_range = (2, 8)
    elif mode == "medium":
        n_voids_range = cfg["num_voids_range"]
    else:
        n_voids_range = (5, 15)

    n_voids = random.randint(*n_voids_range)

    for _ in range(n_voids):
        r = random.randint(*cfg["void_radius_range"])

        x = random.uniform(r, W - r)
        y = random.uniform(r, W - r)

        if random.random() < 0.7:
            aspect = random.uniform(0.8, 1.5)
        else:
            aspect = random.uniform(1.5, 3.2)

        angle = random.uniform(0, math.pi)

        voids.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "aspect": float(aspect),
            "angle": float(angle),
            "type": "irregular_void"
        })

    return voids


# ============================================================
# 9. Particle 렌더링
# ============================================================

def draw_particle(draw, p):
    x, y, r = p["x"], p["y"], p["r"]

    base = random.randint(165, 230)
    edge = random.randint(70, 130)

    # network/percolation 입자는 살짝 더 밝게
    if "percolation" in p["type"] or "network" in p["type"] or "branch" in p["type"]:
        base = random.randint(190, 245)
        edge = random.randint(90, 145)

    # 약간 불규칙한 입자
    if random.random() < 0.35:
        n_lobes = random.randint(2, 5)

        for _ in range(n_lobes):
            rr = random.uniform(0.45 * r, 0.95 * r)
            dx = random.uniform(-0.35 * r, 0.35 * r)
            dy = random.uniform(-0.35 * r, 0.35 * r)
            gray = int(clamp(base + random.randint(-25, 25), 80, 255))

            draw.ellipse(
                [x + dx - rr, y + dy - rr, x + dx + rr, y + dy + rr],
                fill=gray,
                outline=edge
            )
    else:
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=base,
            outline=edge
        )

    # 내부 texture
    for _ in range(random.randint(4, 12)):
        rr = random.uniform(0.08 * r, 0.28 * r)
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, 0.7 * r)

        tx = x + dist * math.cos(angle)
        ty = y + dist * math.sin(angle)

        gray = int(clamp(base + random.randint(-40, 35), 70, 255))

        draw.ellipse(
            [tx - rr, ty - rr, tx + rr, ty + rr],
            fill=gray
        )

    if random.random() < 0.75:
        hx = x - 0.35 * r
        hy = y - 0.35 * r
        hr = 0.18 * r

        draw.ellipse(
            [hx - hr, hy - hr, hx + hr, hy + hr],
            fill=random.randint(225, 255)
        )


# ============================================================
# 10. Irregular void 렌더링
# ============================================================

def draw_irregular_void(img, v):
    W, H = img.size

    x, y = v["x"], v["y"]
    r = v["r"]
    aspect = v["aspect"]
    angle = v["angle"]

    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)

    n_blobs = random.randint(3, 8)

    for _ in range(n_blobs):
        rr_x = random.uniform(0.35 * r, 0.95 * r) * aspect
        rr_y = random.uniform(0.35 * r, 0.95 * r)

        dx = random.uniform(-0.7 * r, 0.7 * r)
        dy = random.uniform(-0.7 * r, 0.7 * r)

        rx = dx * math.cos(angle) - dy * math.sin(angle)
        ry = dx * math.sin(angle) + dy * math.cos(angle)

        cx = x + rx
        cy = y + ry

        d.ellipse(
            [cx - rr_x, cy - rr_y, cx + rr_x, cy + rr_y],
            fill=255
        )

    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 5.5)))

    dark_value = random.randint(0, 12)
    void_img = Image.new("L", (W, H), dark_value)

    img.paste(void_img, (0, 0), mask)


# ============================================================
# 11. SEM-like noise / texture
# ============================================================

def add_noise(img, cfg):
    arr = np.array(img).astype(np.float32)
    H, W = arr.shape

    arr += np.random.normal(0, cfg["gaussian_noise_std"], arr.shape)

    texture = np.random.normal(0, 1, (H, W))
    texture_img = Image.fromarray(
        ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.uint8)
    )

    texture_img = texture_img.filter(ImageFilter.GaussianBlur(radius=20))
    texture = np.array(texture_img).astype(np.float32)
    texture = texture - texture.mean()

    arr += texture * (cfg["texture_strength"] / 255)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=cfg["blur_radius"]))

    return img


# ============================================================
# 12. Metadata feature 계산
# ============================================================

def compute_metadata_features(particles, voids, cfg, mode):
    W = cfg["image_size"]
    area_total = W * W

    filler_area = sum(math.pi * (p["r"] ** 2) for p in particles)
    void_area = sum(math.pi * (v["r"] ** 2) * v["aspect"] for v in voids)

    filler_area_fraction = filler_area / area_total
    void_area_fraction = void_area / area_total

    centers = np.array([[p["x"], p["y"]] for p in particles], dtype=np.float32)
    radii = np.array([p["r"] for p in particles], dtype=np.float32)

    connected_pairs = 0
    if len(particles) > 1:
        sample_n = min(len(particles), 450)
        idx = np.random.choice(len(particles), sample_n, replace=False)
        c = centers[idx]
        r = radii[idx]

        for i in range(sample_n):
            d = np.sqrt(((c[i + 1:] - c[i]) ** 2).sum(axis=1))
            threshold = r[i] + r[i + 1:] + 4
            connected_pairs += int(np.sum(d < threshold))

    connectivity_proxy = connected_pairs / max(1, len(particles))

    percolation_particle_count = sum(
        1 for p in particles if "percolation" in p["type"] or "branch" in p["type"]
    )

    network_particle_count = sum(
        1 for p in particles if "network" in p["type"]
    )

    k_matrix = 0.2
    k_filler = 20.0

    k_proxy = k_matrix * (1 - filler_area_fraction) + k_filler * filler_area_fraction
    k_proxy *= (1 + 0.08 * connectivity_proxy)
    k_proxy *= math.exp(-3.0 * void_area_fraction)

    return {
        "percolation_mode": mode,
        "num_particles": int(len(particles)),
        "num_voids": int(len(voids)),
        "num_percolation_particles": int(percolation_particle_count),
        "num_network_particles": int(network_particle_count),
        "filler_area_fraction": float(filler_area_fraction),
        "void_area_fraction": float(void_area_fraction),
        "connectivity_proxy": float(connectivity_proxy),
        "thermal_conductivity_proxy": float(k_proxy),
    }


# ============================================================
# 13. 이미지 1장 생성
# ============================================================

def generate_image(cfg):
    W = cfg["image_size"]

    mode = choose_percolation_mode(cfg)

    img = Image.new("L", (W, W), random.randint(55, 85))
    draw = ImageDraw.Draw(img)
    
    particles = []
    voids = []

    if cfg.get("use_filler", True):
        particles = generate_base_particles(cfg)
        particles = add_agglomerates(particles, cfg)
        particles += generate_network_particles(cfg, mode)

    if cfg.get("use_void", True):
        voids = generate_voids(cfg, mode)
        
        random.shuffle(particles)

    for p in particles:
        draw_particle(draw, p)

    for v in voids:
        draw_irregular_void(img, v)

    img = add_noise(img, cfg)

    features = compute_metadata_features(particles, voids, cfg, mode)

    metadata = {
        "percolation_mode": mode,
        "particles": particles,
        "voids": voids,
        "features": features
    }

    return img, metadata


# ============================================================
# 14. ZIP 생성
# ============================================================

def zip_dataset(output_dir, zip_name):
    zip_path = os.path.join(output_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                path = os.path.join(root, file)

                if path == zip_path:
                    continue

                arcname = os.path.relpath(path, output_dir)
                zipf.write(path, arcname)

    print("ZIP saved:", zip_path)


# ============================================================
# 15. 전체 dataset 생성
# ============================================================

def generate_dataset(cfg):
    out = cfg["output_dir"]
    img_dir = os.path.join(out, "images")
    meta_dir = os.path.join(out, "metadata")

    ensure_dir(img_dir)
    ensure_dir(meta_dir)

    summary = []

    mode_counter = {"low": 0, "medium": 0, "high": 0}

    for i in tqdm(range(cfg["num_images"])):
        img, metadata = generate_image(cfg)

        name = f"img_{i:05d}"

        img_path = os.path.join(img_dir, name + ".png")
        json_path = os.path.join(meta_dir, name + ".json")

        img.save(img_path)

        metadata["image_path"] = img_path

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mode = metadata["percolation_mode"]
        mode_counter[mode] += 1

        summary.append({
            "image_path": img_path,
            "metadata_path": json_path,
            **metadata["features"]
        })

    with open(os.path.join(out, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out, "mode_summary.json"), "w") as f:
        json.dump(mode_counter, f, indent=2)

    print("Dataset generated:", out)
    print("Percolation mode counts:", mode_counter)

    if cfg["make_zip"]:
        zip_dataset(out, cfg["zip_name"])


def run_generation(num_images, use_filler=True, use_void=True):
    cfg = CONFIG.copy()

    cfg["num_images"] = int(num_images)
    cfg["use_filler"] = use_filler
    cfg["use_void"] = use_void

    output_dir = f"synthetic_sem_dataset_{num_images}"
    zip_name = f"synthetic_sem_dataset_{num_images}.zip"

    cfg["output_dir"] = output_dir
    cfg["zip_name"] = zip_name
    cfg["make_zip"] = True

    generate_dataset(cfg)

    zip_path = os.path.join(output_dir, zip_name)
    return zip_path, output_dir
