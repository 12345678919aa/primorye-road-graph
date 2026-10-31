"""
Разбиение дорожного графа Приморского края на части и визуализация.

Вход:  data/primorye.graphml  (OSMnx, автомобильная сеть края)
Выход: out/  — карты PNG, GraphML с атрибутами part и level, CSV со статистикой.

Этапы:
  1. Иерархия сети: первичная (trunk/primary), вторичная (secondary/tertiary),
     местная (unclassified/residential/living_street).
  2. Разбиение графа на части алгоритмом Лувена (максимизация модулярности).
  3. Визуализация слоёв сети и частей разбиения.
"""
import csv
import math
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SRC = "data/primorye.graphml"
OUT = "out"
RESOLUTION = 0.005   # параметр разрешения Лувена: подобран так, чтобы получить ~11 частей
SEED = 42
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ---------------------------------------------------------------- 1. загрузка
G = nx.read_graphml(SRC)
for _, d in G.nodes(data=True):
    d["x"], d["y"] = float(d["x"]), float(d["y"])
for _, _, d in G.edges(data=True):
    d["length"] = float(d.get("length", 0))
print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")

# ---------------------------------------------------- 2. уровни дорожной сети
RANK = {"trunk": 0, "trunk_link": 0, "primary": 0, "primary_link": 0,
        "secondary": 1, "secondary_link": 1, "tertiary": 1, "tertiary_link": 1,
        "unclassified": 2, "residential": 2, "living_street": 2}
LEVELS = ["Первичная сеть", "Вторичная сеть", "Местная сеть"]
LEVEL_HINT = ["федеральные и региональные трассы (trunk, primary)",
              "межмуниципальные дороги (secondary, tertiary)",
              "улицы и местные проезды (unclassified, residential, living_street)"]


def level_of(hw):
    """Уровень ребра; для списка классов ('[\'tertiary\', \'residential\']') берётся старший."""
    classes = re.findall(r"[a-z_]+", hw)
    return min(RANK.get(c, 2) for c in classes) if classes else 2


for _, _, d in G.edges(data=True):
    d["level"] = level_of(d.get("highway", ""))


def edge_coords(u, v, d):
    """Полилиния ребра: из WKT-геометрии, если есть, иначе отрезок между узлами."""
    geom = d.get("geometry")
    if geom:
        pts = re.findall(r"(-?\d+\.?\d*) (-?\d+\.?\d*)", geom)
        return [(float(x), float(y)) for x, y in pts]
    return [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


# В ориентированном мультиграфе двусторонняя дорога — два ребра; для рисования и подсчёта км
# оставляем одно ребро на каждую пару узлов (+ключ).
seen = set()
segments = []          # (u, v, coords, level, length)
for u, v, k, d in G.edges(keys=True, data=True):
    key = (min(u, v), max(u, v), k)
    if key in seen:
        continue
    seen.add(key)
    segments.append((u, v, edge_coords(u, v, d), d["level"], d["length"]))

# --------------------------------------------------- 3. разбиение на части
U = nx.Graph()
U.add_nodes_from(G.nodes)
U.add_edges_from((u, v) for u, v, *_ in segments if u != v)

communities = nx.community.louvain_communities(U, resolution=RESOLUTION, seed=SEED)
communities = sorted(communities, key=len, reverse=True)

# Мелкие сообщества (изолированные фрагменты, < 1 % узлов) присоединяем к ближайшей
# по расстоянию крупной части, чтобы не плодить «частей» из десятка узлов.
min_size = 0.01 * U.number_of_nodes()
big = [set(c) for c in communities if len(c) >= min_size]
small = [c for c in communities if len(c) < min_size]
coords = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
for c in small:
    cx, cy = np.mean([coords[n] for n in c], axis=0)
    best = min(range(len(big)), key=lambda i: min(
        (coords[n][0] - cx) ** 2 + (coords[n][1] - cy) ** 2 for n in big[i]))
    big[best] |= c
    print(f"Фрагмент из {len(c)} узлов присоединён к части {best + 1}")

part = {n: i for i, c in enumerate(big) for n in c}
K = len(big)
Q = nx.community.modularity(U, big)
print(f"Частей: {K}, модулярность Q = {Q:.3f}")

# Название части — по самому крупному городу, попавшему в неё (координаты центров).
CITIES = [  # примерно по убыванию численности населения
    ("Владивосток", 43.115, 131.885), ("Уссурийск", 43.800, 131.950),
    ("Находка", 42.820, 132.890), ("Артём", 43.360, 132.190),
    ("Арсеньев", 44.160, 133.270), ("Спасск-Дальний", 44.600, 132.820),
    ("Большой Камень", 43.110, 132.350), ("Партизанск", 43.130, 133.120),
    ("Лесозаводск", 45.480, 133.420), ("Дальнегорск", 44.550, 135.570),
    ("Дальнереченск", 45.930, 133.730), ("Лучегорск", 46.480, 134.250),
    ("Кавалерово", 44.270, 135.050), ("Михайловка", 43.930, 132.000),
    ("Пограничный", 44.400, 131.380), ("Славянка", 42.860, 131.380),
    ("Хороль", 44.430, 132.080), ("Чугуевка", 44.170, 133.860),
    ("Кировский", 45.100, 133.510), ("Покровка", 43.940, 131.620),
    ("Камень-Рыболов", 44.760, 132.040), ("Яковлевка", 44.420, 133.470),
    ("Анучино", 43.970, 133.060), ("Ольга", 43.750, 135.290),
    ("Пластун", 44.730, 136.300), ("Терней", 45.050, 136.610),
    ("Новопокровка", 45.860, 134.540), ("Рощино", 45.890, 134.120),
]
node_ids = list(G.nodes)
node_xy = np.array([coords[n] for n in node_ids])
names = [None] * K
for city, lat, lon in CITIES:
    nearest = node_ids[int(np.argmin(((node_xy - (lon, lat)) ** 2).sum(axis=1)))]
    p = part[nearest]
    if names[p] is None:
        names[p] = city
names = [nm or f"Часть {i + 1}" for i, nm in enumerate(names)]

# ------------------------------------------------------- 4. статистика частей
stats = []
cut_edges = 0
km = defaultdict(lambda: [0.0, 0.0, 0.0])
for u, v, _, lvl, length in segments:
    if part[u] == part[v]:
        km[part[u]][lvl] += length / 1000
    else:
        cut_edges += 1
for i, c in enumerate(big):
    stats.append({
        "part": i + 1, "name": names[i], "nodes": len(c),
        "km_primary": round(km[i][0]), "km_secondary": round(km[i][1]),
        "km_local": round(km[i][2]), "km_total": round(sum(km[i])),
        "inner_edges": U.subgraph(c).number_of_edges(),
    })
print(f"Разрезанных рёбер: {cut_edges} из {len(segments)} ({100 * cut_edges / len(segments):.2f} %)")

with open(f"{OUT}/parts_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0]), delimiter=";")
    w.writeheader()
    w.writerows(stats)

# GraphML с результатом — открывается в Gephi/QGIS/NetworkX
for n in G.nodes:
    G.nodes[n]["part"] = part[n] + 1
    G.nodes[n]["part_name"] = names[part[n]]
for u, v, d in G.edges(data=True):
    d["level_name"] = LEVELS[d["level"]]
    d["is_cut"] = str(part[u] != part[v])
nx.write_graphml(G, f"{OUT}/primorye_partitioned.graphml")

# ------------------------------------------------------------ 5. визуализация
ASPECT = 1 / math.cos(math.radians(44.5))   # равные масштабы по широте и долготе
BG, INK, MUTED = "#FBFAF7", "#1F2328", "#6B7079"
LEVEL_COLORS = ["#C8102E", "#E08A00", "#7A8594"]
LEVEL_WIDTH = [1.1, 0.6, 0.18]
PALETTE = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF",
           "#8C564B", "#E377C2", "#BCBD22", "#393B79", "#637939", "#AD494A"]

xs, ys = node_xy[:, 0], node_xy[:, 1]
pad = 0.15
XLIM = (xs.min() - pad, xs.max() + pad)
YLIM = (ys.min() - pad, ys.max() + pad)


def setup(ax, title=None):
    ax.set_facecolor(BG)
    ax.set_aspect(ASPECT)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#D8D6D0")
    ax.set_xlabel("долгота, °E", color=MUTED, fontsize=8)
    ax.set_ylabel("широта, °N", color=MUTED, fontsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=8, weight="bold")


def draw_levels(ax, levels, colors=LEVEL_COLORS, widths=LEVEL_WIDTH, alpha=1.0):
    for lvl in sorted(levels, reverse=True):          # местные снизу, трассы сверху
        segs = [c for _, _, c, l, _ in segments if l == lvl]
        ax.add_collection(LineCollection(segs, colors=colors[lvl], linewidths=widths[lvl],
                                         alpha=alpha, capstyle="round"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=220, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  →", f"{OUT}/{name}.png")


level_km = [sum(s[f"km_{k}"] for s in stats) for k in ("primary", "secondary", "local")]

# 5.1 Иерархия сети: три слоя по отдельности
fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), facecolor=BG)
for lvl, ax in enumerate(axes):
    setup(ax, f"{LEVELS[lvl]}")
    draw_levels(ax, [l for l in range(3) if l != lvl],
                colors=["#E4E2DC"] * 3, widths=[0.3, 0.25, 0.12])
    draw_levels(ax, [lvl], widths=[1.2, 0.8, 0.45])
    km_txt = f"{level_km[lvl]:,}".replace(",", " ")
    ax.text(0.02, 0.98, f"{LEVEL_HINT[lvl]}\n≈ {km_txt} км",
            transform=ax.transAxes, va="top", fontsize=9, color=MUTED,
            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=3))
fig.suptitle("Приморский край: иерархия автомобильной сети", x=0.01, ha="left",
             fontsize=16, weight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "1_network_levels")

# 5.2 Все уровни вместе
fig, ax = plt.subplots(figsize=(10, 12), facecolor=BG)
setup(ax, "Приморский край: первичная, вторичная и местная сети")
draw_levels(ax, [0, 1, 2])
ax.legend(handles=[Line2D([], [], color=LEVEL_COLORS[i], lw=[3, 2, 1.2][i],
                          label=f"{LEVELS[i]} — {level_km[i]:,} км".replace(",", " "))
                   for i in range(3)],
          loc="lower right", frameon=True, facecolor=BG, edgecolor="#D8D6D0", fontsize=9)
save(fig, "2_network_overlay")

# 5.3 Разбиение графа на части
fig, ax = plt.subplots(figsize=(11, 13), facecolor=BG)
setup(ax, f"Разбиение дорожного графа Приморского края на {K} частей")
for lvl in (2, 1, 0):
    segs, cols = [], []
    for u, v, c, l, _ in segments:
        if l == lvl and part[u] == part[v]:
            segs.append(c)
            cols.append(PALETTE[part[u] % len(PALETTE)])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=[1.3, 0.7, 0.25][lvl],
                                     alpha=[1, 0.95, 0.75][lvl]))
cut_segs = [c for u, v, c, *_ in segments if part[u] != part[v]]
ax.add_collection(LineCollection(cut_segs, colors=INK, linewidths=2.2))
cut_nodes = {n for u, v, *_ in segments if part[u] != part[v] for n in (u, v)}
ax.scatter([coords[n][0] for n in cut_nodes], [coords[n][1] for n in cut_nodes],
           s=10, color=INK, zorder=5, linewidths=0)

CITY_XY = {nm: (lon, lat) for nm, lat, lon in CITIES}
# смещения подписей (в пунктах), чтобы в плотном юге они не налезали друг на друга
OFFSETS = {"Владивосток": (-40, -70), "Славянка": (-95, 25), "Большой Камень": (30, -55),
           "Находка": (45, -40), "Уссурийск": (-110, 10), "Арсеньев": (60, 15),
           "Спасск-Дальний": (-120, 20), "Кавалерово": (40, 30), "Дальнегорск": (40, -30)}
for i, c in enumerate(big):
    if names[i] in CITY_XY:
        cx, cy = CITY_XY[names[i]]
    else:
        cx, cy = np.median([coords[n] for n in c], axis=0)
    color = PALETTE[i % len(PALETTE)]
    ax.scatter([cx], [cy], s=45, color=color, edgecolor="white", linewidth=1.5, zorder=7)
    ax.annotate(f"{i + 1}. {names[i]}", (cx, cy), xytext=OFFSETS.get(names[i], (30, 20)),
                textcoords="offset points", ha="center", va="center", fontsize=9,
                weight="bold", color="white", zorder=8,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="white", lw=1.2))

handles = [Line2D([], [], color=PALETTE[i % len(PALETTE)], lw=4,
                  label=f"{i + 1}. {names[i]} — {s['nodes']:,} узл., {s['km_total']:,} км"
                  .replace(",", " "))
           for i, s in enumerate(stats)]
handles.append(Line2D([], [], color=INK, lw=2.2, marker="o", ms=4,
                      label=f"межчастные связи (разрез) — {cut_edges} рёбер"))
ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True,
          facecolor=BG, edgecolor="#D8D6D0", title="Части графа", title_fontsize=9.5)
ax.text(0.02, 0.985,
        f"Алгоритм Лувена, γ = {RESOLUTION}\nМодулярность Q = {Q:.3f}\n"
        f"Узлов: {U.number_of_nodes():,}   рёбер: {U.number_of_edges():,}".replace(",", " "),
        transform=ax.transAxes, va="top", fontsize=9, color=MUTED,
        bbox=dict(facecolor=BG, edgecolor="#D8D6D0", pad=5))
save(fig, "3_partition_map")

# 5.4 Отдельная панель для каждой части + её вторичные сети
cols = 3
rows = math.ceil(K / cols)
fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.6 * rows), facecolor=BG)
for i, ax in enumerate(axes.flat):
    if i >= K:
        ax.axis("off")
        continue
    nodes = big[i]
    pxy = np.array([coords[n] for n in nodes])
    for lvl in (2, 1, 0):
        ax.add_collection(LineCollection(
            [c for u, v, c, l, _ in segments if l == lvl and part[u] == i and part[v] == i],
            colors=LEVEL_COLORS[lvl], linewidths=[1.3, 0.7, 0.2][lvl]))
    m = 0.05
    ax.set_xlim(pxy[:, 0].min() - m, pxy[:, 0].max() + m)
    ax.set_ylim(pxy[:, 1].min() - m, pxy[:, 1].max() + m)
    ax.set_aspect(ASPECT)
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PALETTE[i % len(PALETTE)]); s.set_linewidth(3)
    s = stats[i]
    ax.set_title(f"{i + 1}. {names[i]}", loc="left", fontsize=11, weight="bold", color=INK)
    ax.text(0.02, 0.02, f"{s['nodes']} узлов · {s['km_total']} км\n"
                        f"перв. {s['km_primary']} · втор. {s['km_secondary']} · мест. {s['km_local']} км",
            transform=ax.transAxes, fontsize=7.5, color=MUTED,
            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=2))
fig.legend(handles=[Line2D([], [], color=LEVEL_COLORS[i], lw=[3, 2, 1][i], label=LEVELS[i])
                    for i in range(3)], loc="upper right", ncol=3, frameon=False, fontsize=10)
fig.suptitle("Части графа и их дорожные сети по уровням", x=0.01, ha="left",
             fontsize=16, weight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "4_parts_detail")

# 5.5 Диаграмма: протяжённость дорог по частям и уровням
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
ax.set_facecolor(BG)
order = sorted(range(K), key=lambda i: stats[i]["km_total"])
labels = [f"{i + 1}. {names[i]}" for i in order]
left = np.zeros(K)
for lvl, key in enumerate(("km_primary", "km_secondary", "km_local")):
    vals = np.array([stats[i][key] for i in order])
    ax.barh(labels, vals, left=left, color=LEVEL_COLORS[lvl], label=LEVELS[lvl], height=0.65)
    left += vals
for y, total in enumerate(left):
    ax.text(total + 30, y, f"{int(total):,} км".replace(",", " "), va="center",
            fontsize=8.5, color=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#D8D6D0"); ax.spines["bottom"].set_color("#D8D6D0")
ax.tick_params(colors=INK, labelsize=9)
ax.set_xlabel("протяжённость дорог, км", color=MUTED)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Протяжённость дорог в частях графа по уровням сети", loc="left",
             fontsize=13, weight="bold", color=INK)
save(fig, "5_parts_km_chart")

print("\nСтатистика частей:")
for s in stats:
    print(f"  {s['part']:2d}. {s['name']:16s} узлов {s['nodes']:5d}  км {s['km_total']:5d} "
          f"(перв {s['km_primary']}, втор {s['km_secondary']}, мест {s['km_local']})")
