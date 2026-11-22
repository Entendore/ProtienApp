#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProteinForge v7 — de-novo protein sequence design with 3D cellular automata.

New in v7
---------
• Wave Function Collapse mode: 9-state protein tileset (void, hydrophobic
  core, helix H/P faces, sheet H/P faces, polar surface, +/− surface) with
  directional adjacency rules, weighted radial priors, minimum-entropy
  observation, constraint propagation, and soft contradiction repair.
  Collapses incrementally so you watch the fold crystallize from a
  superposition cloud. Field selector chooses topology flavor:
  Core=Mixed helix/sheet · Band=Sheet-rich · Invert=Helix-rich.
• Structure-true SS in WFC mode (roles are ground truth, feed the ribbon,
  SS track, HELIX records and the Structure score part).
• Per-role amino-acid grammar (core→LIVAFM, helix-H→LAVIM, helix-P→EKQS,
  sheet→VIFTY/NQTS, surface→STNQ, +→KR, −→DE).
• Explode view slider — radial blow-apart of voxels + backbone.
• Depth-fog rendering toggle.
• WFC presets: Protein crystal, Helix bundle, Beta-slab cage (C4).
• Run auto-stops when the wave completes; status shows collapse % + repairs.

Controls: drag rotate · wheel zoom · space run/pause · S step · E extract ·
P pin · R reset view · click residue + 1–4 mutate · ✨ Polish to anneal.

Dependencies: PySide6, numpy
"""

import json
import math
import random
import re
import sys
import time
from collections import Counter, deque

import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QThread, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QKeySequence, QPainter,
                           QPainterPath, QPen, QPolygonF, QRadialGradient,
                           QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QFileDialog, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter,
                               QTabWidget, QTableWidget, QTableWidgetItem,
                               QToolTip, QVBoxLayout, QWidget)

# ----------------------------------------------------------------------------
# Amino-acid data
# ----------------------------------------------------------------------------
CLASS_AAS   = {1: "AVLMIWFCP", 2: "STNQYG", 3: "KRH", 4: "DE"}
CLASS_NAMES = {1: "Hydrophobic", 2: "Polar / Gly", 3: "Positive (+)", 4: "Negative (\u2212)"}
CLASS_HEX   = {1: "#ee9930", 2: "#4dc9a3", 3: "#5c94f2", 4: "#ee5c77"}
CLASS_RGB   = {1: (0.93, 0.60, 0.19), 2: (0.30, 0.79, 0.64),
               3: (0.36, 0.58, 0.95), 4: (0.93, 0.36, 0.47)}
CLASS_RGB_ARR = np.array([[0, 0, 0]] + [CLASS_RGB[i] for i in (1, 2, 3, 4)], np.float32)

AA_CLASS = {a: c for c, s in CLASS_AAS.items() for a in s}
AA_COLOR = {
    "A": "#e8a33d", "V": "#e08f2f", "L": "#d9841f", "I": "#d17b13", "M": "#e8b34a",
    "F": "#c96a10", "W": "#a85408", "C": "#f2d24b", "P": "#e07a9b",
    "S": "#4ecfa8", "T": "#3dbd98", "N": "#7fd6c2", "Q": "#5cc0a5", "G": "#9aa7b5",
    "K": "#5b8ff2", "R": "#3f74e0", "H": "#7aa5f7",
    "D": "#f2617a", "E": "#e04a66",
}
AA3 = dict(zip("ARNDCEQGHILKMFPSTWYV",
               ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS",
                "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
                "TYR", "VAL"]))

AA_MASS = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
           "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
           "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
           "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326}

AA_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
         "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
         "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

PKA_N, PKA_C = 9.69, 2.34
PKA_POS = {"K": 10.53, "R": 12.48, "H": 6.00}
PKA_NEG = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07}

AA_CHARGE7 = {}
for _a, _p in PKA_POS.items():
    AA_CHARGE7[_a] = 1.0 / (1.0 + 10 ** (7.0 - _p))
AA_CHARGE7["H"] = 1.0 / (1.0 + 10 ** (7.0 - PKA_POS["H"]))
for _a, _p in PKA_NEG.items():
    AA_CHARGE7[_a] = -1.0 / (1.0 + 10 ** (_p - 7.0))

OFFSETS = [(dx, dy, dz)
           for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
           if (dx, dy, dz) != (0, 0, 0)]

MODES = [("potts",  "Hydrophobic Collapse (Potts\u2013Metropolis)"),
         ("wfc",    "Wave Function Collapse (protein tiles)"),
         ("excite", "Excitable Medium (Greenberg\u2013Hastings)"),
         ("turing", "Turing Patterns (Gray\u2013Scott RD)"),
         ("lenia",  "Lenia Organism (continuous CA)"),
         ("cyclic", "Cyclic Spirals (4-phase waves)"),
         ("block",  "Block Copolymer (Schelling coarsening)"),
         ("grow",   "Diffusion-Limited Growth (DLA-style)")]
MODE_SHORT = {"potts": "Potts", "wfc": "WFC", "excite": "Excite", "turing": "Turing",
              "lenia": "Lenia", "cyclic": "Cyclic", "block": "Block", "grow": "DLA"}
MODE_COLOR = {"potts": "#ee9930", "wfc": "#5ee0c8", "excite": "#c58af9",
              "turing": "#6cc3f5", "lenia": "#7be3ae", "cyclic": "#ff9aa8",
              "block": "#4dc9a3", "grow": "#ffd166"}

PARAM_LABELS = {
    "potts":  {"temp": "Thermal noise",   "thresh": None,                   "branch": None},
    "wfc":    {"temp": None,              "thresh": None,                   "branch": None},
    "excite": {"temp": None,              "thresh": "Excitation threshold", "branch": None},
    "turing": {"temp": "Feed rate (F)",   "thresh": None,                   "branch": "Kill rate (k)"},
    "lenia":  {"temp": "dt \u00d7100",       "thresh": None,                   "branch": "Target \u03bc \u00d7100"},
    "cyclic": {"temp": None,              "thresh": "Spiral threshold",     "branch": None},
    "block":  {"temp": "Migration noise", "thresh": None,                   "branch": None},
    "grow":   {"temp": "Field noise",     "thresh": None,                   "branch": "Branching rate"},
}

FIELDS = [("core", "Core shell"), ("invert", "Inverted"), ("band", "Membrane band")]

SS_COLOR = {"h": "#ff6f91", "e": "#ffd166", "c": "#8b9aad"}

PART_MAX = {"Core burial": 30, "Compactness": 20, "Charge balance": 15,
            "Diversity": 10, "Structure": 10, "Amphipathicity": 15}
PART_COL = {"Core burial": "#ee9930", "Compactness": "#6cc3f5",
            "Charge balance": "#7be3ae", "Diversity": "#c58af9",
            "Structure": "#ff6f91", "Amphipathicity": "#ffd166"}

GOALS = {
    "Balanced": None,
    "Max hydrophobic core": {"Core burial": 2.0, "Compactness": 1.5,
                             "Charge balance": 0.4, "Diversity": 0.5,
                             "Structure": 0.5, "Amphipathicity": 0.5},
    "Amphipathic helix":    {"Core burial": 0.6, "Compactness": 0.6,
                             "Charge balance": 0.6, "Diversity": 0.8,
                             "Structure": 1.2, "Amphipathicity": 2.5},
    "Compact miniprotein":  {"Core burial": 1.2, "Compactness": 2.2,
                             "Charge balance": 0.5, "Diversity": 0.5,
                             "Structure": 0.8, "Amphipathicity": 0.5},
    "Catalytic pocket":     {"Core burial": 0.8, "Compactness": 1.2,
                             "Charge balance": 0.6, "Diversity": 0.8,
                             "Structure": 0.8, "Amphipathicity": 0.4},
}

MOTIFS = [
    ("N-glyco (N-X-S/T)", r"N[^P][ST]"),
    ("Hexa-His tag", r"H{6}"),
    ("RGD integrin loop", r"RGD"),
    ("CXXC redox pair", r"C[^C]{2}C"),
    ("HExxH protease", r"HE[^H]{2}H"),
    ("GxxxG dimerization", r"G[^G]{3}G"),
    ("P-loop (GxxxxGKS/T)", r"G[^G]{4}GK[ST]"),
    ("Poly-basic \u22654 (NLS-like)", r"[KR]{4,}"),
]

# ----------------------------------------------------------------------------
# Wave Function Collapse tileset
# ----------------------------------------------------------------------------
(WFC_VOID, WFC_CORE, WFC_HELH, WFC_HELP, WFC_SHEH, WFC_SHEP,
 WFC_SP, WFC_SPOS, WFC_SNEG) = range(9)
WFC_K = 9
WFC_STATE_NAME = ["void", "core-H", "helix-H", "helix-P", "sheet-H", "sheet-P",
                  "surface-P", "surface+", "surface\u2212"]
WFC_CLASS = np.array([0, 1, 1, 2, 1, 2, 2, 3, 4], np.int8)
WFC_ROLE = {WFC_HELH: "h", WFC_HELP: "h", WFC_SHEH: "e", WFC_SHEP: "e"}
WFC_BRIGHT = np.array([0.0, 0.80, 1.00, 0.95, 0.95, 0.90, 0.90, 1.00, 1.00], np.float32)
WFC_AA = {
    WFC_CORE: {"L": 3, "V": 3, "I": 3, "A": 2, "M": 1, "F": 1, "C": 0.5},
    WFC_HELH: {"L": 2, "A": 2, "V": 2, "I": 1, "M": 1, "F": 1},
    WFC_HELP: {"E": 2, "K": 1.5, "Q": 2, "S": 1.5, "A": 1},
    WFC_SHEH: {"V": 3, "I": 3, "F": 2, "Y": 1, "T": 1, "L": 1},
    WFC_SHEP: {"T": 2, "N": 2, "Q": 2, "S": 2, "Y": 1, "G": 1},
    WFC_SP:   {"S": 2, "T": 2, "N": 2, "Q": 2, "G": 1, "Y": 0.5},
    WFC_SPOS: {"K": 2.5, "R": 2, "H": 0.5},
    WFC_SNEG: {"D": 2.5, "E": 2},
}


def _build_wfc_pairs():
    """Binary adjacency per axis (+x, +y, +z). P[ax][s, t]: state t may sit
    next to state s in that direction. Helix axis = Z; sheet plane = XZ."""
    V, C, HH, HP, EH, EP, SP, PP, NN = range(9)
    P = [np.zeros((WFC_K, WFC_K), bool) for _ in range(3)]

    def allow(ax, a, b, both=True):
        P[ax][a, b] = True
        if both:
            P[ax][b, a] = True

    # void touches only the solvent-facing surface layer
    for s in (SP, PP, NN):
        for ax in range(3):
            allow(ax, V, s)
    for ax in range(3):
        allow(ax, V, V)
    # hydrophobic core packs against anything buried
    for s in (C, HH, HP, EH, EP, SP):
        for ax in range(3):
            allow(ax, C, s)
    # helix rods along Z; H/P interface in cross-section
    allow(2, HH, HH)
    allow(2, HH, HP)
    allow(2, HP, HP)
    for ax in (0, 1):
        allow(ax, HH, HP)
        allow(ax, HP, HP)
        allow(ax, HP, SP)
        allow(ax, HH, SP)          # exposed hydrophobic face (amphipathic)
    # beta slabs in the XZ plane, H/P faces alternate along Y
    for ax in (0, 2):
        allow(ax, EH, EH)
        allow(ax, EH, EP)
        allow(ax, EP, EP)
        allow(ax, EP, SP)
    allow(1, EH, EP)
    allow(1, EP, EP)
    allow(1, EP, SP)
    # polar surface + salt bridges
    for ax in range(3):
        allow(ax, SP, SP)
        allow(ax, SP, PP)
        allow(ax, SP, NN)
        allow(ax, PP, NN)          # salt bridge!
        allow(ax, PP, PP)
        allow(ax, NN, NN)
        allow(ax, PP, HP)
        allow(ax, NN, HP)
    return P


def _build_wfc_strength():
    """Soft weights for the observation sampling (coherence bias)."""
    V, C, HH, HP, EH, EP, SP, PP, NN = range(9)
    S = [P.astype(np.float32) for P in WFC_PAIR]
    for ax in range(3):
        S[ax][C, C] = 2.6
        S[ax][C, HH] = 2.0
        S[ax][C, EH] = 2.0
        S[ax][SP, SP] = 1.8
        S[ax][PP, NN] = 3.2
    S[2][HH, HH] = 3.2
    S[2][HP, HP] = 2.6
    S[2][HH, HP] = 2.0
    for ax in (0, 2):
        S[ax][EH, EH] = 3.0
        S[ax][EP, EP] = 2.4
    S[1][EH, EP] = 2.8
    return S


WFC_PAIR = _build_wfc_pairs()
WFC_STR = _build_wfc_strength()

# ----------------------------------------------------------------------------
# Example presets
# ----------------------------------------------------------------------------
PRESETS = [
    dict(name="Globular fold (baseline)", mode="potts", n=14, sym=0, occ=60,
         temp=25, thresh=6, branch=55, steps=2, target=120, exotic=25,
         exclude="", cut=False, cutz=7, field="core",
         take="Reference soluble-globule scenario",
         desc="4-state Potts\u2013Metropolis collapse with a core-shell radial "
              "field. Hydrophobes condense inward, polar/charged residues coat "
              "the surface \u2014 then press \u2728 Polish to affinity-mature "
              "the sequence."),
    dict(name="WFC protein crystal", mode="wfc", n=14, sym=0, occ=62,
         temp=25, thresh=6, branch=55, steps=2, target=130, exotic=25,
         exclude="", cut=False, cutz=7, field="core",
         take="Watch a fold crystallize from superposition",
         desc="Wave Function Collapse over a 9-state protein tileset: each "
              "lattice cell starts as a superposition of core / helix / sheet / "
              "surface states, then collapses cell-by-cell (minimum entropy "
              "first) with constraint propagation sweeping the rules through "
              "the lattice. The pale cloud is unobserved; colour crystallizes "
              "behind the wavefront. Helices become rods, sheets become slabs, "
              "the surface gains K/R\u2013D/E salt bridges \u2014 and secondary "
              "structure is ground truth, not guessed. Field = Core selects "
              "the mixed helix/sheet flavor. Press E any time to extract; the "
              "run auto-stops at 100%."),
    dict(name="WFC helix bundle", mode="wfc", n=16, sym=0, occ=66,
         temp=25, thresh=6, branch=55, steps=2, target=150, exotic=25,
         exclude="", cut=False, cutz=8, field="invert",
         take="Helix-rich topology flavor",
         desc="WFC with the helix-rich weight flavor: rods of helix-H with "
              "helix-P half-pipes pack against the core while polar surfaces "
              "close the shell. Compare the helical wheel afterwards \u2014 "
              "H/P striping along the rods reads as amphipathic helices."),
    dict(name="WFC beta-slab cage (C4)", mode="wfc", n=16, sym=4, occ=60,
         temp=25, thresh=6, branch=55, steps=2, target=150, exotic=25,
         exclude="", cut=True, cutz=8, field="band",
         take="Sheet-rich slabs under 4-fold symmetry",
         desc="WFC with the sheet-rich flavor and exact C4 symmetry: "
              "hydrophobic/polar slab pairs tile the shell like a \u03b2-barrel "
              "annulus. Cut it open to watch the slab sandwich; the SS track "
              "will be dense with 'e'."),
    dict(name="Cold molten globule", mode="potts", n=12, sym=0, occ=45,
         temp=8, thresh=6, branch=55, steps=2, target=100, exotic=30,
         exclude="", cut=False, cutz=6, field="core",
         take="Low noise, small crystalline core",
         desc="Low thermal noise and sparse occupancy give a small, well-"
              "annealed core. Cut it open: the core is lattice-crystalline at "
              "low temperature, like a molten-globule intermediate."),
    dict(name="C4 symmetric cage", mode="potts", n=16, sym=4, occ=72,
         temp=18, thresh=6, branch=55, steps=2, target=160, exotic=25,
         exclude="", cut=True, cutz=8, field="core",
         take="90\u00b0-symmetric shell, ferritin-like",
         desc="Exact C4 rotational symmetry constrains every update, producing "
              "a shell architecture reminiscent of ferritin or viral cages. "
              "Switch on the oligomer view to see the assembly implied by the "
              "symmetry."),
    dict(name="Four-helix bundle", mode="potts", n=16, sym=0, occ=68,
         temp=14, thresh=6, branch=55, steps=2, target=150, exotic=25,
         exclude="", cut=False, cutz=8, field="core",
         take="Compact up-down bundle archetype",
         desc="Moderate noise on a dense sphere anneals blocky hydrophobic "
              "stripes \u2014 the traced chain reads as alternating amphipathic "
              "segments. Check the helical wheel: the hydrophobic face should "
              "line up."),
    dict(name="\u03b2-barrel shell (C4)", mode="potts", n=18, sym=4, occ=55,
         temp=20, thresh=6, branch=55, steps=2, target=160, exotic=25,
         exclude="", cut=True, cutz=9, field="band",
         take="Hydrophobic ring wall, barrel-like",
         desc="The membrane-band field confines hydrophobes to a cylindrical "
              "wall at mid-radius \u2014 an annular \u03b2-barrel-like shell. "
              "The Radial tab shows the orange band peaking at r\u22480.6."),
    dict(name="Membrane hairpin", mode="potts", n=14, sym=0, occ=75,
         temp=10, thresh=6, branch=55, steps=2, target=120, exotic=25,
         exclude="", cut=False, cutz=7, field="band",
         take="Bilayer-spanning hydrophobic band",
         desc="A dense lattice with the membrane-band field: hydrophobic "
              "residues occupy the bilayer-equivalent shell while loops stay "
              "polar \u2014 a model of a two-pass membrane hairpin."),
    dict(name="Coiled-coil dimer (C2)", mode="potts", n=14, sym=2, occ=58,
         temp=16, thresh=6, branch=55, steps=2, target=110, exotic=25,
         exclude="", cut=False, cutz=7, field="core",
         take="Two-fold symmetric rope-like fold",
         desc="C2 symmetry with a tight core field gives elongated paired "
              "hydrophobic seams \u2014 a coiled-coil-like dimer archetype."),
    dict(name="Miniprotein (Trp-cage-like)", mode="potts", n=10, sym=0, occ=70,
         temp=8, thresh=6, branch=55, steps=2, target=38, exotic=70,
         exclude="", cut=False, cutz=5, field="core",
         take="Tiny, Trp-rich, compact",
         desc="A 38-residue target in a tiny lattice with the exotic-residue "
              "bias cranked up \u2014 W/F/Y-rich like real miniprotein folds. "
              "Polish it with the 'Compact miniprotein' mindset."),
    dict(name="Catalytic pocket globule", mode="potts", n=16, sym=0, occ=62,
         temp=15, thresh=6, branch=55, steps=2, target=130, exotic=25,
         exclude="", cut=False, cutz=8, field="core", pocket=2,
         take="Central ligand void + polar shell",
         desc="A void is carved at the centre and the field enriches "
              "polar/charged residues around it \u2014 a stylized active site. "
              "The ligand glows magenta; Properties lists the pocket residues "
              "and net pocket charge."),
    dict(name="Amyloid surface (risk demo)", mode="potts", n=12, sym=0, occ=55,
         temp=4, thresh=6, branch=55, steps=2, target=120, exotic=25,
         exclude="", cut=False, cutz=6, field="invert",
         take="Inverted field \u2192 sticky hydrophobic surface",
         desc="The field is flipped: hydrophobes are expelled to the surface. "
              "The quality flags go red on purpose \u2014 a teaching example of "
              "what NOT to design."),
    dict(name="Charged nanocage (C4)", mode="potts", n=16, sym=4, occ=70,
         temp=45, thresh=6, branch=55, steps=2, target=160, exotic=25,
         exclude="", cut=False, cutz=8, field="core",
         take="Hot, charge-decorated symmetric cage",
         desc="High thermal noise + C4 symmetry leaves a fuzzy, highly charged "
              "surface \u2014 like designed ferritins with tuned solubility."),
    dict(name="Turing spots (nano-dots)", mode="turing", n=18, sym=0, occ=60,
         temp=31, thresh=6, branch=57, steps=1, target=140, exotic=25,
         exclude="", cut=False, cutz=9, field="core",
         take="Gray\u2013Scott spots \u2192 discrete domains",
         desc="Reaction\u2013diffusion in 3D with spot-forming feed/kill. "
              "High-V domains map to residue classes, so the sequence is read "
              "straight off the Turing pattern."),
    dict(name="Catalytic spot map", mode="turing", n=16, sym=0, occ=60,
         temp=18, thresh=6, branch=40, steps=1, target=140, exotic=25,
         exclude="", cut=False, cutz=8, field="core",
         take="Sparse charged islands \u2014 'active site' pattern",
         desc="Lower kill rate keeps few, sharp Turing spots. The sparse "
              "high-V islands read out as charged residues scattered on a "
              "polar background \u2014 a stylized active-site map."),
    dict(name="Turing labyrinth", mode="turing", n=18, sym=0, occ=60,
         temp=66, thresh=6, branch=86, steps=1, target=150, exotic=25,
         exclude="", cut=True, cutz=9, field="core",
         take="Maze tunnels \u2192 alternating bands",
         desc="Higher feed/kill moves Gray\u2013Scott into the labyrinth "
              "regime: worm-like tunnels wind through the sphere and the "
              "extracted chain alternates hydrophobic/polar \u2014 amphipathic "
              "by construction."),
    dict(name="Lenia organism", mode="lenia", n=18, sym=0, occ=60,
         temp=30, thresh=6, branch=55, steps=1, target=130, exotic=25,
         exclude="", cut=False, cutz=9, field="core",
         take="Continuous CA \u2014 self-organizing gliders",
         desc="A Lenia-style continuous cellular automaton: smooth shell-kernel "
              "convolution plus a growth mapping create self-organizing blobs "
              "that glide, merge and dissolve. The field u maps to residue "
              "classes like the Turing mode."),
    dict(name="Lenia C4 colony", mode="lenia", n=18, sym=4, occ=60,
         temp=30, thresh=6, branch=55, steps=1, target=140, exotic=25,
         exclude="", cut=True, cutz=9, field="core",
         take="Lenia under exact 4-fold symmetry",
         desc="The same continuous CA with exact C4 symmetry enforced every "
              "step \u2014 four identical organisms dancing in lockstep."),
    dict(name="Cyclic spiral galaxy", mode="cyclic", n=18, sym=0, occ=75,
         temp=25, thresh=13, branch=55, steps=1, target=150, exotic=25,
         exclude="", cut=False, cutz=9, field="core",
         take="Rotating 4-phase spiral arms",
         desc="A cyclic cellular automaton: each cell advances 1\u21922\u21923"
              "\u21924\u21921 when enough neighbours hold the next state. In 3D "
              "this grows rotating spiral arms \u2014 state = phase, so the "
              "traced sequence is a time-record of the wave."),
    dict(name="Block copolymer melt", mode="block", n=16, sym=0, occ=70,
         temp=10, thresh=6, branch=55, steps=2, target=140, exotic=25,
         exclude="", cut=False, cutz=8, field="core",
         take="Phase-separating domains (Schelling-style)",
         desc="Schelling-style coarsening: cells join the local majority, "
              "carving ever-larger same-class domains like a block copolymer "
              "melt. Migration noise keeps boundaries alive."),
    dict(name="Mitosis waves", mode="excite", n=16, sym=0, occ=55,
         temp=25, thresh=5, branch=55, steps=2, target=130, exotic=25,
         exclude="", cut=False, cutz=8, field="core",
         take="Greenberg\u2013Hastings scroll waves",
         desc="An excitable medium supporting 3D scroll waves. Wave phase maps "
              "to residue class, so passing waves write pulsing charge "
              "patterns into the sequence."),
    dict(name="Dendrite crystal", mode="grow", n=18, sym=0, occ=35,
         temp=30, thresh=6, branch=18, steps=2, target=150, exotic=25,
         exclude="", cut=False, cutz=9, field="core",
         take="Diffusion-limited branching growth",
         desc="DLA-style growth with a low branching rate. Classes propagate "
              "from the invading neighbourhood \u2014 a fossil record of "
              "accretion events."),
    dict(name="Knotty dendrite", mode="grow", n=18, sym=0, occ=30,
         temp=35, thresh=6, branch=75, steps=2, target=150, exotic=25,
         exclude="", cut=False, cutz=9, field="core",
         take="Aggressive branching \u2192 tangled topologies",
         desc="High branching rate grows a dense thicket; the traced chain "
              "takes sharp turns and the \u0394E scan lights up at the "
              "junctions where classes clash."),
    dict(name="High-temp spin glass", mode="potts", n=12, sym=0, occ=65,
         temp=100, thresh=6, branch=55, steps=2, target=110, exotic=25,
         exclude="", cut=False, cutz=6, field="core",
         take="Near-random \u2014 the denatured limit",
         desc="Thermal noise cranked to maximum: states flip essentially "
              "randomly and no core forms. A calibration baseline for "
              "'denatured'."),
]


def net_charge(seq, ph):
    pos = 1 / (1 + 10 ** (ph - PKA_N)) + \
          sum(seq.count(a) / (1 + 10 ** (ph - p)) for a, p in PKA_POS.items())
    neg = 1 / (1 + 10 ** (PKA_C - ph)) + \
          sum(seq.count(a) / (1 + 10 ** (p - ph)) for a, p in PKA_NEG.items())
    return pos - neg


def net_charge_from_counts(cnt, ph):
    pos = 1 / (1 + 10 ** (ph - PKA_N)) + \
          sum(cnt.get(a, 0) / (1 + 10 ** (ph - p)) for a, p in PKA_POS.items())
    neg = 1 / (1 + 10 ** (PKA_C - ph)) + \
          sum(cnt.get(a, 0) / (1 + 10 ** (p - ph)) for a, p in PKA_NEG.items())
    return pos - neg


def estimate_pi(seq):
    if not seq:
        return 7.0
    lo, hi = 0.0, 14.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def radius_gyration(path):
    if len(path) < 2:
        return 0.0
    pts = np.array(path, np.float64)
    c = pts.mean(axis=0)
    return float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean()))


def shannon_entropy(seq):
    if not seq:
        return 0.0
    n = len(seq)
    cnt = Counter(seq)
    return -sum((v / n) * math.log(v / n) for v in cnt.values())


def best_amphi_window(seq, w=10, angle_deg=100.0):
    if len(seq) < w:
        return None
    th = math.radians(angle_deg)
    best = (0, 0.0)
    for i in range(len(seq) - w + 1):
        x = y = 0.0
        for k, a in enumerate(seq[i:i + w]):
            h = AA_KD[a]
            x += h * math.cos(k * th)
            y += h * math.sin(k * th)
        mu = math.hypot(x, y) / w
        if mu > best[1]:
            best = (i, mu)
    return best


def hydrophobic_moment(seq, window=10, angle_deg=100.0):
    r = best_amphi_window(seq, window, angle_deg)
    return r[1] if r else 0.0


def glyco_sites(seq):
    return len(re.findall(r"N[^P][ST]", seq))


def sap_proxy(seq, path):
    n = len(seq)
    if n < 5:
        return 0.0
    P = np.asarray(path, np.float64)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    hv = np.array([AA_KD[a] for a in seq])
    cls = np.array([AA_CLASS[a] for a in seq])
    hi = np.where(cls == 1)[0]
    if not len(hi):
        return 0.0
    vals = []
    for i in hi:
        m = D[i] <= 1.6
        loc = float(hv[m].mean())
        exposure = 1.0 - min(1.0, float((cls[m] == 1).sum()) / 8.0)
        vals.append(max(0.0, loc) * exposure)
    return float(np.mean(vals))


def motif_scan(seq):
    out = []
    for name, pat in MOTIFS:
        ms = [m.start() for m in re.finditer(pat, seq)]
        if ms:
            out.append((name, [p + 1 for p in ms[:8]]))
    return out


def salt_bridges(seq, contact):
    if not contact.size:
        return []
    out = []
    for i, a in enumerate(seq):
        ci = AA_CLASS[a]
        if ci not in (3, 4):
            continue
        for j in np.where(contact[i])[0]:
            j = int(j)
            if j <= i:
                continue
            cj = AA_CLASS[seq[j]]
            if (ci, cj) in ((3, 4), (4, 3)):
                out.append((i, j))
    return out


def residue_potential(seq, path):
    n = len(seq)
    if n < 2:
        return []
    P = np.asarray(path, np.float64)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    q = np.array([1.0 if a in "KR" else -1.0 if a in "DE"
                  else 0.1 if a == "H" else 0.0 for a in seq])
    np.fill_diagonal(D, 1e9)
    return list((q[None, :] / (D + 0.8)).sum(axis=1))


def pocket_residues(path, classes, pocket_r, n):
    c = (n - 1) / 2.0
    out = []
    for k, (x, y, z) in enumerate(path):
        d = math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
        if d <= pocket_r + 1.7:
            out.append((k, int(classes[k])))
    return out


def pick_aa(cls, x, y, z, salt, exotic=2.0, exclude=""):
    aas = CLASS_AAS[cls]
    h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^ salt
    h &= 0x7FFFFFFF
    u = (h >> 7) / float(1 << 24)
    weights = []
    for a in aas:
        w = exotic if a in "WCFMY" else 1.0
        if a in exclude:
            w = 0.0
        weights.append(w)
    tot = sum(weights)
    if tot <= 0:
        return aas[h % len(aas)]
    acc = 0.0
    for a, w in zip(aas, weights):
        acc += w / tot
        if u <= acc:
            return a
    return aas[-1]


def pick_aa_role(cls, state, x, y, z, salt, exotic=2.0, exclude=""):
    """Role-aware deterministic residue choice (WFC)."""
    aas = CLASS_AAS[cls]
    base = WFC_AA.get(state, {})
    h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^ (salt ^ 0x5bf03635)
    h &= 0x7FFFFFFF
    u = (h >> 7) / float(1 << 24)
    weights = []
    for a in aas:
        w = base.get(a, 0.35)
        if a in "WCFMY":
            w *= exotic * 0.5
        if a in exclude:
            w = 0.0
        weights.append(w)
    tot = sum(weights)
    if tot <= 0:
        return pick_aa(cls, x, y, z, salt, exotic, exclude)
    acc = 0.0
    for a, w in zip(aas, weights):
        acc += w / tot
        if u <= acc:
            return a
    return aas[-1]


# ----------------------------------------------------------------------------
# Pairwise contact energy + structure analysis
# ----------------------------------------------------------------------------
def _build_J():
    J = np.zeros((5, 5), np.float32)
    def s(a, b, v):
        J[a, b] = v; J[b, a] = v
    s(1, 1, -1.05); s(2, 2, -0.50); s(3, 4, -1.30)
    s(3, 3, 0.45);  s(4, 4, 0.45)
    s(1, 2, 0.25);  s(1, 3, 0.30);  s(1, 4, 0.30)
    s(2, 3, -0.15); s(2, 4, -0.15)
    return J


ENERGY_J = _build_J()


def assign_ss(path):
    n = len(path)
    ss = ["c"] * n
    if n < 5:
        return ss
    P = np.asarray(path, np.float64)
    D = np.diff(P, axis=0)
    L = np.linalg.norm(D, axis=1)
    L[L == 0] = 1.0
    U = D / L[:, None]
    m = len(U)
    for i in range(m - 3):
        dots = [float(np.dot(U[i + k], U[i + k + 1])) for k in range(3)]
        if all(abs(d) < 0.5 for d in dots):
            t = float(np.dot(np.cross(U[i], U[i + 1]), U[i + 2]))
            if abs(t) > 0.4:
                for k in range(i, min(i + 5, n)):
                    ss[k] = "h"
    for i in range(m - 2):
        if float(np.dot(U[i], U[i + 2])) > 0.92 and \
                ss[i] == "c" and ss[i + 1] == "c" and ss[i + 2] == "c":
            ss[i] = ss[i + 1] = ss[i + 2] = "e"
    i = 0
    while i < n:
        j = i
        while j < n and ss[j] == ss[i]:
            j += 1
        if ss[i] != "c" and (j - i) < 3:
            for k in range(i, j):
                ss[k] = "c"
        i = j
    return ss


def residue_contacts(path, sep_min=3, cutoff=1.9):
    n = len(path)
    if n < sep_min + 1:
        return np.zeros((n, n), bool)
    P = np.asarray(path, np.float64)
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return (D2 <= cutoff ** 2) & (sep >= sep_min)


def long_range_contacts(contact):
    n = contact.shape[0]
    if n <= 10:
        return 0
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    return int((contact & (sep >= 10)).sum() // 2)


def contact_arcs(contact, max_arcs=90, min_sep=12):
    if not contact.size:
        return []
    n = contact.shape[0]
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    ii, jj = np.where(contact & (sep >= min_sep))
    if not len(ii):
        return []
    order = np.argsort(-(jj - ii))[:max_arcs]
    return [(int(ii[k]), int(jj[k]), int(jj[k] - ii[k])) for k in order]


def design_score(seq, path, classes, contact, ss, weights=None):
    parts = {k: 0.0 for k in PART_MAX}
    n = len(seq)
    if n < 10:
        return 0.0, parts
    cls = np.asarray(classes)
    hh = contact & (cls[:, None] == 1) & (cls[None, :] == 1)
    hi = np.where(cls == 1)[0]
    f_burial = 0.0
    if len(hi):
        f_burial = float(np.clip(hh[hi].sum(axis=1) / 4.0, 0, 1).mean())
    rg = radius_gyration(path)
    f_comp = 0.0
    if rg > 1e-6:
        rg_sph = math.sqrt(0.6) * (3 * n / (4 * math.pi)) ** (1 / 3)
        f_comp = float(np.clip(1.6 * rg_sph / rg, 0, 1))
    q = net_charge(seq, 7.0)
    f_charge = float(np.clip(1 - abs(q) / (0.22 * n), 0, 1))
    f_div = float(shannon_entropy(seq) / math.log(20))
    f_struct = float(np.clip(sum(1 for s in ss if s in "he") / n / 0.45, 0, 1))
    f_amphi = float(np.clip(hydrophobic_moment(seq) / 0.6, 0, 1))
    fracs = {"Core burial": f_burial, "Compactness": f_comp,
             "Charge balance": f_charge, "Diversity": f_div,
             "Structure": f_struct, "Amphipathicity": f_amphi}
    for k, f in fracs.items():
        parts[k] = f * PART_MAX[k]
    if weights:
        num = sum(parts[k] * weights.get(k, 1.0) for k in parts)
        den = sum(PART_MAX[k] * weights.get(k, 1.0) for k in parts)
        return 100.0 * num / max(den, 1e-9), parts
    return sum(parts.values()), parts


def mutation_scan(classes, contact):
    n = len(classes)
    if n < 2 or not contact.size:
        return np.zeros(0, np.float32)
    C = contact.astype(np.float32)
    cls = np.asarray(classes)
    onehot = np.stack([(cls == k).astype(np.float32) for k in range(5)], 1)
    cnt = C @ onehot
    E = cnt @ ENERGY_J.T
    idx = np.arange(n)
    Ecur = E[idx, cls]
    dE = E - Ecur[:, None]
    dE[idx, 0] = -1e9
    dE[idx, cls] = -1e9
    return np.maximum(dE.max(axis=1), 0.0)


def radial_profile(path, classes, bins=18):
    if len(path) < 4:
        return None
    P = np.asarray(path, np.float64)
    c = P.mean(axis=0)
    d = np.linalg.norm(P - c, axis=1)
    dmx = max(float(d.max()), 1e-6)
    idx = np.clip((d / dmx * bins).astype(int), 0, bins - 1)
    cls = np.asarray(classes)
    cnt = np.bincount(idx, minlength=bins).astype(np.float32)
    out = np.zeros((bins, 4), np.float32)
    for k in range(4):
        out[:, k] = np.bincount(idx[cls == k + 1], minlength=bins) / np.maximum(cnt, 1)
    return out


def quality_flags(seq, classes, lc, sb=0, pocket_n=-1, sap=None):
    flags = []
    if not seq:
        return flags
    n = len(seq)
    best = cur = 0
    for c in classes:
        cur = cur + 1 if c == 1 else 0
        best = max(best, cur)
    if best >= 9:
        flags.append(("bad", f"Hydrophobic run of {best} residues \u2014 aggregation-prone"))
    elif best >= 6:
        flags.append(("warn", f"Hydrophobic run of {best} residues \u2014 check solubility"))
    else:
        flags.append(("ok", f"No long hydrophobic runs (max {best})"))
    q = net_charge(seq, 7.0)
    if abs(q) > 0.28 * n:
        flags.append(("bad", f"Extreme net charge {q:+.0f} \u2014 unlikely to fold alone"))
    elif abs(q) > 0.15 * n:
        flags.append(("warn", f"High net charge {q:+.0f} at pH 7"))
    else:
        flags.append(("ok", f"Charge balanced ({q:+.1f} at pH 7)"))
    if lc == 0:
        flags.append(("warn", "No long-range contacts (sep \u2265 10) \u2014 extended fold"))
    else:
        flags.append(("ok", f"{lc} long-range contacts \u2014 compact topology"))
    if sb >= 1:
        flags.append(("ok", f"{sb} salt bridge(s) (K/R\u2013D/E contacts)"))
    else:
        flags.append(("warn", "No salt bridges \u2014 surface may lack charge pairs"))
    if sap is not None:
        if sap > 1.4:
            flags.append(("bad", f"High surface hydrophobicity (SAP {sap:.1f})"))
        elif sap > 0.9:
            flags.append(("warn", f"Elevated surface hydrophobicity (SAP {sap:.1f})"))
        else:
            flags.append(("ok", f"Surface hydrophobicity ok (SAP {sap:.1f})"))
    cys = seq.count("C")
    if cys == 1:
        flags.append(("warn", "Single Cys \u2014 unpaired thiol"))
    elif cys >= 2:
        flags.append(("ok", f"{cys} Cys \u2014 possible disulfides"))
    g = glyco_sites(seq)
    if g:
        flags.append(("warn", f"{g} N-glycosylation motif(s) N-X-S/T"))
    if sum(seq.count(a) for a in "FWY") == 0:
        flags.append(("warn", "No aromatic residues (no UV/fluorescence handle)"))
    if pocket_n >= 0:
        if pocket_n >= 4:
            flags.append(("ok", f"Pocket lined by {pocket_n} residues"))
        else:
            flags.append(("warn", f"Pocket too shallow ({pocket_n} lining residues)"))
    return flags


def suggest_disulfide(seq, contact):
    best = None
    n = len(seq)
    if not contact.size:
        return None
    for i in range(n):
        for j in np.where(contact[i])[0]:
            j = int(j)
            if j >= i + 10 and seq[i] != "C" and seq[j] != "C":
                if best is None or j - i > best[0]:
                    best = (j - i, i, j)
    return best


def build_pdb(seq, path, classes, ss, meta, extra_chains=None, ligand=None):
    rule, seed, step, score = meta
    out = [f"REMARK   1 ProteinForge de-novo design rule={rule} seed={seed} step={step}",
           f"REMARK   2 score={score:.1f} length={len(seq)} CA-trace, 1 voxel = 3.8 A",
           "REMARK   3 B-factor = residue class (10=hydrophobic 20=polar 30=+ 40=-)"]
    if extra_chains:
        out.append(f"REMARK   4 stylized {1 + len(extra_chains)}-chain assembly from "
                   f"lattice symmetry")
    k, i = 0, 0
    n = len(ss)
    while i < n:
        if ss[i] == "h":
            j = i
            while j < n and ss[j] == "h":
                j += 1
            if j - i >= 4:
                k += 1
                out.append(f"HELIX  {k:3d} {k:3d} {AA3[seq[i]]:3s} A {i + 1:4d}  "
                           f"{AA3[seq[j - 1]]:3s} A {j:4d}  1")
            i = j
        else:
            i += 1
    chains = [([(p[0], p[1], p[2]) for p in path], "A")]
    if extra_chains:
        for kk, ch in enumerate(extra_chains):
            chains.append(([(p[0], p[1], p[2]) for p in ch], "ABCD"[(kk + 1) % 26]))
    atom = 0
    for pts, cid in chains:
        for idx, (p, a) in enumerate(zip(pts, seq)):
            atom += 1
            x, y, z = (c * 3.8 for c in p)
            out.append(f"ATOM  {atom:5d}  CA  {AA3[a]:3s} {cid}{idx + 1:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{classes[idx] * 10:6.2f}           C")
    if ligand is not None:
        lx, ly, lz, lr = ligand
        atom += 1
        out.append(f"HETATM{atom:5d}  LIG LIG A 999    "
                   f"{lx * 3.8:8.3f}{ly * 3.8:8.3f}{lz * 3.8:8.3f}  1.00{lr * 3.8:6.2f}           L")
    out.append("TER")
    out.append("END")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# CA engine
# ----------------------------------------------------------------------------
class CAEngine:
    J = ENERGY_J
    TURING_SUBSTEPS = 6
    WFC_CELLS_PER_STEP = 45

    def __init__(self, n=14, mode="potts", seed=424242, density=0.6, symmetry=0,
                 profile="core", pocket_r=0.0):
        self.n = n
        self.mode = mode
        self.pocket_r = float(pocket_r or 0.0)
        ii = np.arange(n)
        self.ix, self.iy, self.iz = np.meshgrid(ii, ii, ii, indexing="ij")
        c = (n - 1) / 2.0
        self.dist = np.sqrt((self.ix - c) ** 2 + (self.iy - c) ** 2 +
                            (self.iz - c) ** 2).astype(np.float32)
        self.parity = ((self.ix + self.iy + self.iz) % 2).astype(np.int8)
        self.profile = profile
        self.last_roles = None
        self.set_symmetry(symmetry)
        self.reseed(seed, density)

    # -- symmetry -------------------------------------------------------------
    def _sym_maps(self, order):
        if order not in (2, 4):
            return None
        n = self.n
        n1 = n - 1
        x2d, y2d = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        if order == 2:
            xs = np.stack([x2d, n1 - x2d])
            ys = np.stack([y2d, n1 - y2d])
        else:
            xs = np.stack([x2d, y2d, n1 - x2d, n1 - y2d])
            ys = np.stack([y2d, n1 - x2d, n1 - y2d, x2d])
        k = (xs * n + ys).argmin(axis=0)
        cx = np.take_along_axis(xs, k[None], 0)[0]
        cy = np.take_along_axis(ys, k[None], 0)[0]
        return cx, cy

    def set_symmetry(self, order):
        self.symmetry = order
        self._sym_xy = self._sym_maps(order)
        if getattr(self, "grid", None) is not None or \
                getattr(self, "u", None) is not None or \
                getattr(self, "allowed", None) is not None:
            self.enforce_symmetry()

    def enforce_symmetry(self):
        if self._sym_xy is None:
            return
        cx, cy = self._sym_xy
        if self.mode == "wfc":
            self.allowed = self.allowed[:, cx, cy]
            self.collapsed = self.collapsed[cx, cy]
            self._wfc_shimmer = self._wfc_shimmer[cx, cy]
            self._wfc_cnt = self.allowed.sum(axis=0).astype(np.int8)
        elif self.mode in ("turing", "lenia"):
            self.u = self.u[cx, cy]
            if self.mode == "turing":
                self.v = self.v[cx, cy]
        else:
            self.grid = self.grid[cx, cy]

    # -- field / pocket -------------------------------------------------------
    def _make_bias(self):
        n = self.n
        r = self._r
        bias = np.zeros((5, n, n, n), np.float32)
        s = np.clip((self.dist - 0.5 * r) / max(0.5 * r, 1e-6), 0.0, 1.0)
        u = s * 2 - 1
        if self.profile == "band":
            band = np.exp(-((self.dist - 0.62 * r) / (0.18 * r + 1e-6)) ** 2)
            f = band * 2 - 1
        elif self.profile == "invert":
            f = -u
        else:
            f = u
        bias[1] = 2.6 * f
        bias[2] = bias[3] = bias[4] = -2.6 * f
        pr = self.pocket_r
        if pr > 0:
            shell = np.clip((pr + 2.2 - self.dist) / 2.2, 0.0, 1.0) * \
                (self.dist >= pr - 0.5)
            bias[1] += 1.8 * shell
            bias[2] -= 1.8 * shell
            bias[3] -= 1.8 * shell
            bias[4] -= 1.8 * shell
        self.bias = bias

    def set_profile(self, profile):
        self.profile = profile
        if self.mode == "wfc":
            if getattr(self, "mask", None) is not None:
                self._wfc_setup()
        elif getattr(self, "_r", None) is not None:
            self._make_bias()

    def set_pocket(self, r):
        self.pocket_r = float(r or 0.0)
        self._cg_cache_step = -1
        if self.mode == "wfc":
            if getattr(self, "mask", None) is not None:
                self._wfc_setup()
            return
        if getattr(self, "mask", None) is None:
            return
        pr = self.pocket_r
        self.mask = self._base_mask & (self.dist > pr) if pr > 0 else self._base_mask
        self._make_bias()
        if self.mode == "turing":
            self.v[~self.mask] = 0.0
        elif self.mode == "lenia":
            self.u[~self.mask] = 0.0
        elif getattr(self, "grid", None) is not None:
            self.grid[~self.mask] = 0

    # -- setup ----------------------------------------------------------------
    def reseed(self, seed, density):
        self.seed = seed
        self.density = float(np.clip(density, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.aa_salt = (seed * 2654435761) % (2 ** 31)
        self.step_count = 0
        self._cg_cache_step = -1
        self.last_roles = None
        n = self.n
        self.rmax = n / 2 - 0.6
        r = max(1.5, (0.55 + 0.45 * self.density) * self.rmax)
        self._r = r
        base = self.dist <= r
        self._base_mask = base
        pr = self.pocket_r
        self.mask = base & (self.dist > pr) if pr > 0 else base
        self._make_bias()

        if self.mode == "wfc":
            self._wfc_setup()
            return

        if self.mode == "turing":
            self.u = np.ones((n, n, n), np.float32)
            self.v = np.zeros((n, n, n), np.float32)
            self.v[self.dist <= 1.3 + 1.2 * self.density] = 0.30
            self.enforce_symmetry()
            return

        if self.mode == "lenia":
            self._init_kernel()
            self._make_lenia_seeds()
            self.enforce_symmetry()
            return

        g = np.zeros((n, n, n), np.int8)
        if self.mode in ("potts", "cyclic", "block"):
            g[self.mask] = self.rng.integers(1, 5, int(self.mask.sum()))
        elif self.mode == "excite":
            g[self.mask & (self.rng.random((n, n, n)) < 0.02)] = 1
        else:
            core = self.dist <= (1.2 + 1.8 * self.density)
            g[core] = self.rng.integers(1, 5, int(core.sum()))
        self.grid = g
        self.enforce_symmetry()

    # -- WFC ------------------------------------------------------------------
    def _wfc_setup(self):
        n, K = self.n, WFC_K
        self._wfc_repairs = 0
        r = max(self._r, 1e-6)
        f = np.clip(self.dist / r, 0.0, 1.2)
        flavor = {"core": "mixed", "band": "sheet",
                  "invert": "helix"}.get(self.profile, "mixed")
        if flavor == "helix":
            bumps = {WFC_CORE: (0.30, 0.28, 1.6), WFC_HELH: (0.55, 0.30, 2.2),
                     WFC_HELP: (0.55, 0.30, 1.9), WFC_SHEH: (0.50, 0.30, 0.5),
                     WFC_SHEP: (0.50, 0.30, 0.5), WFC_SP: (0.85, 0.25, 1.4),
                     WFC_SPOS: (0.88, 0.22, 0.8), WFC_SNEG: (0.88, 0.22, 0.8)}
        elif flavor == "sheet":
            bumps = {WFC_CORE: (0.30, 0.26, 1.2), WFC_HELH: (0.55, 0.30, 0.6),
                     WFC_HELP: (0.55, 0.30, 0.6), WFC_SHEH: (0.52, 0.30, 2.4),
                     WFC_SHEP: (0.52, 0.30, 2.1), WFC_SP: (0.85, 0.25, 1.4),
                     WFC_SPOS: (0.88, 0.22, 0.8), WFC_SNEG: (0.88, 0.22, 0.8)}
        else:
            bumps = {WFC_CORE: (0.30, 0.27, 1.7), WFC_HELH: (0.55, 0.30, 1.5),
                     WFC_HELP: (0.55, 0.30, 1.4), WFC_SHEH: (0.52, 0.30, 1.1),
                     WFC_SHEP: (0.52, 0.30, 1.0), WFC_SP: (0.85, 0.25, 1.5),
                     WFC_SPOS: (0.88, 0.22, 0.9), WFC_SNEG: (0.88, 0.22, 0.9)}
        w = np.zeros((K, n, n, n), np.float32)
        for s, (mu, wd, amp) in bumps.items():
            w[s] = amp * np.exp(-((f - mu) / wd) ** 2).astype(np.float32)
        self._wfc_w = w

        allowed = np.zeros((K, n, n, n), bool)
        inside = self.mask
        for s in (WFC_CORE, WFC_HELH, WFC_HELP, WFC_SHEH, WFC_SHEP,
                  WFC_SP, WFC_SPOS, WFC_SNEG):
            allowed[s] = inside
        allowed[[WFC_SP, WFC_SPOS, WFC_SNEG]] &= (f > 0.28)[None]
        allowed[WFC_CORE] &= (f < 0.85)[None]
        allowed[WFC_VOID] = ~inside
        self.allowed = allowed
        self.collapsed = (~inside).copy()

        self._wfc_shimmer = self.rng.random((n, n, n)).astype(np.float32)
        self._wfc_cnt = allowed.sum(axis=0).astype(np.int8)
        self._wfc_base_ok = (f > 0.28)
        self._wfc_base_coreok = (f < 0.85)
        self.enforce_symmetry()
        self._wfc_autocollapse()

    def _wfc_repair(self, x, y, z):
        self._wfc_repairs += 1
        A = self.allowed
        A[:, x, y, z] = False
        if not self.mask[x, y, z]:
            A[WFC_VOID, x, y, z] = True
            self.collapsed[x, y, z] = True
            self._wfc_cnt[x, y, z] = 1
            return
        for s in (WFC_CORE, WFC_HELH, WFC_HELP, WFC_SHEH, WFC_SHEP,
                  WFC_SP, WFC_SPOS, WFC_SNEG):
            A[s, x, y, z] = True
        f = float(self.dist[x, y, z]) / max(self._r, 1e-6)
        if f <= 0.28:
            A[WFC_SP, x, y, z] = A[WFC_SPOS, x, y, z] = A[WFC_SNEG, x, y, z] = False
        if f > 0.85:
            A[WFC_CORE, x, y, z] = False
        self.collapsed[x, y, z] = False
        self._wfc_cnt[x, y, z] = int(A[:, x, y, z].sum())

    def _wfc_propagate(self, x, y, z):
        A = self.allowed
        n = self.n
        Pf = [P.astype(np.float32) for P in WFC_PAIR]
        PTf = [P.T.astype(np.float32) for P in WFC_PAIR]
        q = deque([(x, y, z)])
        while q:
            cx, cy, cz = q.popleft()
            src = A[:, cx, cy, cz].astype(np.float32)
            if not src.any():
                self._wfc_repair(cx, cy, cz)
                src = A[:, cx, cy, cz].astype(np.float32)
            for (dxx, dyy, dzz) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                    (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = cx + dxx, cy + dyy, cz + dzz
                if not (0 <= nx < n and 0 <= ny < n and 0 <= nz < n):
                    continue
                if dxx:
                    pm = Pf[0] if dxx > 0 else PTf[0]
                elif dyy:
                    pm = Pf[1] if dyy > 0 else PTf[1]
                else:
                    pm = Pf[2] if dzz > 0 else PTf[2]
                sup = (src @ pm) > 0.5
                tgt = A[:, nx, ny, nz]
                new = tgt & sup
                if not new.any():
                    self._wfc_repair(nx, ny, nz)
                    q.append((nx, ny, nz))
                elif not np.array_equal(new, tgt):
                    A[:, nx, ny, nz] = new
                    self._wfc_cnt[nx, ny, nz] = int(new.sum())
                    q.append((nx, ny, nz))

    def _wfc_autocollapse(self, max_pass=6):
        for _ in range(max_pass):
            unit = self.mask & (~self.collapsed) & (self._wfc_cnt == 1)
            if not unit.any():
                break
            xs, ys, zs = np.where(unit)
            for x, y, z in zip(xs, ys, zs):
                self.collapsed[x, y, z] = True
                self._wfc_propagate(int(x), int(y), int(z))

    def _wfc_collapse_one(self):
        cand = self.mask & (~self.collapsed)
        if not cand.any():
            return
        ent = self._wfc_cnt.astype(np.float32) + self._wfc_shimmer * 0.9
        ent[~cand] = 1e9
        x, y, z = np.unravel_index(int(np.argmin(ent)), ent.shape)
        if self._wfc_cnt[x, y, z] == 0:
            self._wfc_repair(x, y, z)
            self._wfc_propagate(x, y, z)
            return
        A = self.allowed
        dom = A[:, x, y, z].astype(np.float32)
        coh = np.zeros(WFC_K, np.float32)
        n = self.n
        for ax, (dx, dy, dz) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            for sgn in (1, -1):
                xx, yy, zz = x + sgn * dx, y + sgn * dy, z + sgn * dz
                if 0 <= xx < n and 0 <= yy < n and 0 <= zz < n and \
                        self.collapsed[xx, yy, zz]:
                    sn = A[:, xx, yy, zz]
                    if sn.sum() == 1:
                        s = int(np.argmax(sn))
                        coh += WFC_STR[ax][s] if sgn > 0 else WFC_STR[ax][:, s]
        score = self._wfc_w[:, x, y, z] * dom * (1.0 + 0.7 * coh)
        tot = float(score.sum())
        if tot <= 1e-9:
            p = dom / max(1.0, float(dom.sum()))
        else:
            p = score / tot
        s = int(self.rng.choice(WFC_K, p=p))
        A[:, x, y, z] = False
        A[s, x, y, z] = True
        self._wfc_cnt[x, y, z] = 1
        self.collapsed[x, y, z] = True
        self._wfc_propagate(x, y, z)

    def _step_wfc(self):
        cells = CAEngine.WFC_CELLS_PER_STEP
        for _ in range(cells):
            self._wfc_collapse_one()
        self._wfc_autocollapse()

    def wfc_progress(self):
        tot = int(self.mask.sum())
        if tot == 0:
            return 1.0, 0
        done = int((self.collapsed & self.mask).sum())
        return done / tot, self._wfc_repairs

    def wfc_complete(self):
        return not bool(self.mask & (~self.collapsed)).any() if \
            hasattr(self, "mask") else False

    def _wfc_state_grid(self):
        return np.argmax(self.allowed, axis=0).astype(np.int8)

    def _init_kernel(self):
        r0, sk = 2.05, 0.85
        self._kernel = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d == 0 or d > 3.05:
                        continue
                    w = math.exp(-((d - r0) / sk) ** 2)
                    if w > 0.02:
                        self._kernel.append((dx, dy, dz, w))
        self._ksum = sum(w for _, _, _, w in self._kernel)

    def _make_lenia_seeds(self):
        n = self.n
        u = np.zeros((n, n, n), np.float32)
        c = (n - 1) / 2.0
        blobs = [(c, c, c)]
        for _ in range(2 + int(3 * self.density)):
            ang = self.rng.random() * 2 * math.pi
            rad = 1.5 + self.rng.random() * max(1.0, self._r * 0.5)
            blobs.append((c + rad * math.cos(ang), c + rad * math.sin(ang),
                          c + (self.rng.random() - 0.5) * 2.0))
        for (bx, by, bz) in blobs:
            d = np.sqrt((self.ix - bx) ** 2 + (self.iy - by) ** 2 +
                        (self.iz - bz) ** 2)
            u = np.maximum(u, np.exp(-(d / 1.1) ** 2).astype(np.float32))
        if self._sym_xy is not None:
            cx, cy = self._sym_xy
            u = u[cx, cy]
        u[~self.mask] = 0.0
        self.u = u

    # -- helpers ----------------------------------------------------------------
    def _onehot(self, g):
        return (g[None, ...] == np.arange(5, dtype=np.int8)[:, None, None, None]) \
            .astype(np.float32)

    def _shift_sum(self, arr):
        out = np.zeros_like(arr)
        for dx, dy, dz in OFFSETS:
            s = arr
            if dx: s = np.roll(s, dx, axis=-3)
            if dy: s = np.roll(s, dy, axis=-2)
            if dz: s = np.roll(s, dz, axis=-1)
            out += s
        return out

    def _lap(self, a):
        s = np.zeros_like(a)
        for axis in (0, 1, 2):
            s += np.roll(a, 1, axis)
            s += np.roll(a, -1, axis)
        return s - 6 * a

    def step(self, p):
        if self.mode == "potts":
            self._step_potts(p["temp"])
        elif self.mode == "wfc":
            self._step_wfc()
        elif self.mode == "excite":
            self._step_excite(p["thresh"])
        elif self.mode == "turing":
            self._step_turing(p["feed"], p["kill"])
        elif self.mode == "lenia":
            self._step_lenia(p["dt"], p["mu"])
        elif self.mode == "cyclic":
            self._step_cyclic(p["thresh"])
        elif self.mode == "block":
            self._step_block(p["noise01"])
        else:
            self._step_grow(p["branch"], p["flip_noise"])
        self.enforce_symmetry()
        self.step_count += 1

    def _step_potts(self, temp):
        T = max(0.03, temp)
        g = self.grid
        cnt = self._shift_sum(self._onehot(g))
        Eall = np.tensordot(CAEngine.J, cnt, axes=([1], [0])) + self.bias
        Ecur = np.take_along_axis(Eall, g[None].astype(np.intp), 0)[0]
        prop = self.rng.integers(1, 5, size=g.shape)
        Eprop = np.take_along_axis(Eall, prop[None].astype(np.intp), 0)[0]
        dE = Eprop - Ecur
        acc = (dE <= 0) | (self.rng.random(g.shape) <
                           np.exp(-np.clip(dE, 0.0, 40.0) / T))
        self.grid = np.where(self.mask & acc, prop, g).astype(np.int8)

    def _step_excite(self, thresh):
        g = self.grid
        en = self._shift_sum((g == 1).astype(np.int8))
        new = np.zeros_like(g)
        new[g == 1] = 2
        born = (g == 0) & (en >= thresh)
        spark = (g == 0) & (self.rng.random(g.shape) < 0.0004)
        new[born | spark] = 1
        self.grid = new

    def _step_turing(self, feed, kill):
        u, v = self.u, self.v
        if self._sym_xy is not None:
            cx, cy = self._sym_xy
            nz = self.rng.random(v.shape).astype(np.float32)
            nz = nz[cx, cy]
        else:
            nz = self.rng.random(v.shape).astype(np.float32)
        v = v + (nz - 0.5) * 0.006
        for _ in range(CAEngine.TURING_SUBSTEPS):
            lu = self._lap(u)
            lv = self._lap(v)
            uvv = u * v * v
            u = u + (0.15 * lu - uvv + feed * (1.0 - u))
            v = v + (0.075 * lv + uvv - (feed + kill) * v)
            np.clip(u, 0.0, 1.2, out=u)
            np.clip(v, 0.0, 1.2, out=v)
            u[~self.mask] = 1.0
            v[~self.mask] = 0.0
        self.u, self.v = u, v

    def _step_lenia(self, dt, mu):
        u = self.u
        p = np.zeros_like(u)
        for dx, dy, dz, w in self._kernel:
            s = u
            if dx: s = np.roll(s, dx, axis=0)
            if dy: s = np.roll(s, dy, axis=1)
            if dz: s = np.roll(s, dz, axis=2)
            p += w * s
        p /= self._ksum
        g = 2.0 * np.exp(-((p - mu) / 0.045) ** 2) - 1.0
        u = u + dt * g
        if self.rng.random() < 0.5:
            nz = (self.rng.random(u.shape) < 0.0005).astype(np.float32) * 0.8
            if self._sym_xy is not None:
                cx, cy = self._sym_xy
                nz = nz[cx, cy]
            u = np.maximum(u, nz)
        self.u = np.clip(u, 0.0, 1.0)
        self.u[~self.mask] = 0.0

    def _step_cyclic(self, thresh):
        g = self.grid
        cnt = self._shift_sum(self._onehot(g))
        nxt = (g % 4) + 1
        cnt_next = np.take_along_axis(cnt, nxt[None].astype(np.intp), 0)[0]
        born = self.mask & (g > 0) & (cnt_next >= thresh)
        new = g.copy()
        new[born] = nxt[born]
        spark = self.mask & (g == 0) & (self.rng.random(g.shape) < 0.0006)
        new[spark] = 1
        self.grid = new

    def _step_block(self, noise01):
        g = self.grid
        cnt = self._shift_sum(self._onehot(g))
        same = np.take_along_axis(cnt, g[None].astype(np.intp), 0)[0]
        low = self.mask & (g > 0) & (same < 13.0)
        maj = cnt.argmax(axis=0).astype(np.int8)
        maj[maj == 0] = 2
        new = g.copy()
        new[low] = maj[low]
        flip = self.mask & (self.rng.random(g.shape) < noise01 * 0.12)
        if flip.any():
            new[flip] = self.rng.integers(1, 5, int(flip.sum()))
        fav = self.bias.argmax(axis=0).astype(np.int8)
        pull = self.mask & (self.rng.random(g.shape) < 0.05) & (new != fav)
        new[pull] = fav[pull]
        self.grid = new

    def _step_grow(self, p_grow, noise):
        g = self.grid
        occ = g > 0
        en = self._shift_sum(occ.astype(np.int8))
        th = 3 if occ.mean() < 0.42 else 6
        grow = (~occ) & (en >= th) & (self.rng.random(g.shape) < p_grow)
        flip = occ & (self.rng.random(g.shape) < 0.10)
        if grow.any() or flip.any():
            cnt = self._shift_sum(self._onehot(g))
            cnt[0] = 0
            cnt += self.rng.random(cnt.shape).astype(np.float32) * noise
            newcls = np.argmax(cnt, axis=0).astype(np.int8)
            g[grow] = newcls[grow]
            g[flip] = newcls[flip]
        die = occ & (en <= 1) & (self.rng.random(g.shape) < 0.06)
        g[die] = 0
        self.grid = g

    def _quantile_classes(self, field, thresh=0.12):
        act = self.mask & (field > thresh)
        cg = np.zeros(field.shape, np.int8)
        if act.any():
            q1, q2, q3 = np.quantile(field[act], [0.55, 0.80, 0.94])
            cg[act] = (4 - np.digitize(field[act], [q1, q2, q3])).astype(np.int8)
        return cg

    def class_grid(self):
        if self.mode == "wfc":
            if self._cg_cache_step != self.step_count:
                st = self._wfc_state_grid()
                ok = self.collapsed & (st != WFC_VOID)
                cg = np.zeros(st.shape, np.int8)
                cg[ok] = WFC_CLASS[st[ok]]
                self._cg_cache = cg
                self._cg_cache_step = self.step_count
            return self._cg_cache
        if self.mode in ("turing", "lenia"):
            if self._cg_cache_step != self.step_count:
                self._cg_cache = self._quantile_classes(
                    self.v if self.mode == "turing" else self.u)
                self._cg_cache_step = self.step_count
            return self._cg_cache
        if self.mode == "excite":
            g = self.grid
            cls = np.full(g.shape, 2, np.int8)
            cls[g == 1] = 1
            ref = g == 2
            cls[ref] = np.where(self.parity[ref] == 0, 3, 4)
            return cls
        return self.grid

    def render_arrays(self):
        if self.mode == "wfc":
            st = self._wfc_state_grid()
            n = self.n
            cloud = self.mask & (~self.collapsed)
            matter = self.collapsed & (st != WFC_VOID)
            occ = cloud | matter
            cols = np.zeros((n, n, n, 3), np.float32)
            if cloud.any():
                g = 0.30 + 0.30 * self._wfc_shimmer[cloud]
                cols[cloud] = np.stack([g * 0.72, g * 0.84, g * 1.05],
                                       axis=-1)
            if matter.any():
                br = WFC_BRIGHT[st[matter]]
                cols[matter] = CLASS_RGB_ARR[st[matter]] * br[:, None]
            return occ, np.clip(cols, 0.0, 1.0)
        g = self.grid
        if self.mode in ("turing", "lenia"):
            cg = self.class_grid()
            if self.mode == "turing":
                occ = self.v > 0.10
                bright = 0.35 + 1.5 * np.minimum(self.v, 0.8)
            else:
                occ = self.u > 0.12
                bright = 0.35 + 1.4 * np.minimum(self.u, 0.85)
            cols = CLASS_RGB_ARR[cg] * bright[..., None]
        elif self.mode == "excite":
            cg = self.class_grid()
            bright = np.full(g.shape, 0.30, np.float32)
            bright[g == 2] = 0.62
            bright[g == 1] = 1.0
            cols = CLASS_RGB_ARR[cg] * bright[..., None]
            occ = self.mask
        else:
            occ = g > 0
            cols = CLASS_RGB_ARR[g].astype(np.float32)
        return occ, np.clip(cols, 0.0, 1.0)

    def metrics(self):
        cg = self.class_grid()
        occ = cg > 0
        if not occ.any():
            return 0.0, 0.0, 0.0
        cnt = self._shift_sum(self._onehot(cg))
        E = np.tensordot(CAEngine.J, cnt, axes=([1], [0]))
        e_cell = np.take_along_axis(E, cg[None].astype(np.intp), 0)[0]
        if self.mode == "potts":
            e_cell = e_cell + np.take_along_axis(
                self.bias, cg[None].astype(np.intp), 0)[0]
        energy = float(e_cell[occ].mean())
        hc = self._shift_sum((cg == 1).astype(np.float32))
        hmask = cg == 1
        burial = float(((hc >= 5) & hmask).sum() / max(1, int(hmask.sum())))
        return energy, burial, float(occ.mean())

    def extract(self, target_len, exotic=2.0, exclude=""):
        if self.mode == "wfc":
            st = self._wfc_state_grid()
            occ = self.collapsed & (st != WFC_VOID)
            cg = np.zeros(st.shape, np.int8)
            cg[occ] = WFC_CLASS[st[occ]]
        else:
            cg = self.class_grid()
            occ = cg > 0
            st = None
        if not occ.any():
            self.last_roles = None
            return [], "", []
        n = self.n
        deg = np.where(occ, self._shift_sum(occ.astype(np.int8)), 127).astype(np.int32)
        c0 = (n - 1) / 2.0
        restarts = 3 if target_len > 160 else 4
        best_path = []
        for _ in range(restarts):
            starts = np.argwhere(deg == deg.min())
            s0 = starts[self.rng.integers(len(starts))]
            cur = (int(s0[0]), int(s0[1]), int(s0[2]))
            visited = {cur}
            path = [cur]
            while len(path) < target_len:
                cx_, cy_, cz_ = cur
                best, bestsc = None, 1e18
                for dx, dy, dz in OFFSETS:
                    nx, ny, nz = cx_ + dx, cy_ + dy, cz_ + dz
                    if 0 <= nx < n and 0 <= ny < n and 0 <= nz < n \
                            and occ[nx, ny, nz] and (nx, ny, nz) not in visited:
                        sc = (float(deg[nx, ny, nz])
                              + 0.55 * self.rng.random()
                              + 0.12 * math.sqrt((nx - c0) ** 2 + (ny - c0) ** 2
                                                 + (nz - c0) ** 2))
                        if sc < bestsc:
                            bestsc, best = sc, (nx, ny, nz)
                if best is None:
                    break
                path.append(best)
                visited.add(best)
                cur = best
            if len(path) > len(best_path):
                best_path = path
            if len(best_path) >= target_len:
                break
        seq, classes, roles = [], [], []
        for (x, y, z) in best_path:
            c = int(cg[x, y, z])
            classes.append(c)
            if st is not None:
                s = int(st[x, y, z])
                roles.append(WFC_ROLE.get(s, "c"))
                seq.append(pick_aa_role(c, s, x, y, z, self.aa_salt,
                                        exotic, exclude))
            else:
                roles.append("c")
                seq.append(pick_aa(c, x, y, z, self.aa_salt, exotic, exclude))
        self.last_roles = roles if st is not None else None
        return best_path, "".join(seq), classes

    def get_roles(self):
        return self.last_roles


# ----------------------------------------------------------------------------
# Breeding Lab worker
# ----------------------------------------------------------------------------
def seq_identity(a, b):
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    return sum(1 for x, y in zip(a[:m], b[:m]) if x == y) / m


class BatchThread(QThread):
    progress = Signal(int, int, float)
    done = Signal(list)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cancelled = False

    def run(self):
        res = []
        best = -1.0
        try:
            p = self.cfg["params"]
            w = self.cfg.get("weights")
            for k in range(self.cfg["trials"]):
                if self.cancelled:
                    break
                seed = random.randint(1, 9999999)
                eng = CAEngine(self.cfg["n"], self.cfg["mode"], seed,
                               self.cfg["occ"], self.cfg["sym"],
                               self.cfg.get("field", "core"),
                               self.cfg.get("pocket", 0.0))
                for _ in range(self.cfg["steps"]):
                    eng.step(p)
                path, seq, classes = eng.extract(
                    self.cfg["target"], self.cfg["exotic"], self.cfg["exclude"])
                ss = assign_ss(path)
                contact = residue_contacts(path)
                score, parts = design_score(seq, path, classes, contact, ss,
                                            weights=w)
                entry = {
                    "seq": seq, "classes": [int(c) for c in classes],
                    "path": [[int(v) for v in q] for q in path], "ss": ss,
                    "score": score, "parts": parts, "mode": self.cfg["mode"],
                    "seed": seed, "step": self.cfg["steps"], "len": len(seq),
                    "gravy": sum(AA_KD[a] for a in seq) / max(1, len(seq)),
                    "pi": estimate_pi(seq),
                    "lc": long_range_contacts(contact),
                    "goal": self.cfg.get("goal", "Balanced")}
                res.append(entry)
                best = max(best, score)
                self.progress.emit(k + 1, self.cfg["trials"], best)
        except Exception:
            pass
        finally:
            res.sort(key=lambda e: -e["score"])
            kept = []
            for e in res:
                if all(seq_identity(e["seq"], k["seq"]) < 0.9 for k in kept):
                    kept.append(e)
                if len(kept) >= self.cfg["keep"]:
                    break
            self.done.emit(kept)


class BreedingDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Breeding Lab \u2014 batch design search")
        self.setMinimumWidth(420)
        self.cfg_getter = parent._batch_config
        self.on_done = parent._merge_batch
        self.thread = None

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Run many independent CA runs in a background thread and keep the "
            "best-scoring designs (near-duplicates filtered). Uses the current "
            "rule, lattice, symmetry, field, pocket and extraction settings."))
        grid = QVBoxLayout()
        self.goal = self._combo(grid, "Design goal", list(GOALS.keys()))
        self.trials = self._row(grid, "Trials", 1, 500, 48)
        self.steps = self._row(grid, "Steps per run", 10, 3000, 250)
        self.keep = self._row(grid, "Keep top", 1, 50, 12)
        v.addLayout(grid)
        self.prog = QProgressBar()
        self.info = QLabel("idle")
        v.addWidget(self.prog)
        v.addWidget(self.info)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        v.addLayout(row)

        self.start_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        close_btn.clicked.connect(self.close)

    def _row(self, lay, text, lo, hi, val):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        row.addStretch(1)
        row.addWidget(sp)
        lay.addLayout(row)
        return sp

    def _combo(self, lay, text, items):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        cb = QComboBox()
        cb.addItems(items)
        row.addStretch(1)
        row.addWidget(cb)
        lay.addLayout(row)
        return cb

    def start(self):
        if self.thread is not None:
            return
        cfg = self.cfg_getter()
        cfg["trials"] = self.trials.value()
        cfg["steps"] = self.steps.value()
        cfg["keep"] = self.keep.value()
        cfg["goal"] = self.goal.currentText()
        cfg["weights"] = GOALS[cfg["goal"]]
        self.thread = BatchThread(cfg)
        self.thread.progress.connect(self.on_progress)
        self.thread.done.connect(self.on_done)
        self.thread.done.connect(self._finished)
        self.thread.start()
        self.prog.setRange(0, cfg["trials"])
        self.prog.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.info.setText("running\u2026")

    def on_progress(self, k, total, best):
        self.prog.setValue(k)
        self.info.setText(f"trial {k}/{total} \u00b7 best score so far {best:.1f}")

    def _finished(self):
        self.thread = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def cancel(self):
        if self.thread is not None:
            self.thread.cancelled = True

    def closeEvent(self, e):
        self.cancel()
        if self.thread is not None:
            self.thread.wait(3000)
        super().closeEvent(e)


# ----------------------------------------------------------------------------
# 3D viewport
# ----------------------------------------------------------------------------
CORNERS = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                    for z in (-0.5, 0.5)], np.float32)
FACE_IDX = [(4, 5, 7, 6), (0, 1, 3, 2), (2, 3, 7, 6),
            (0, 1, 5, 4), (1, 5, 7, 3), (0, 4, 6, 2)]
FACE_NRM = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


class Viewport3D(QWidget):
    residuePicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 360)
        self.setCursor(Qt.CrossCursor)
        self.yaw = math.radians(35)
        self.pitch = math.radians(22)
        self.zoom = 1.0
        self.occ = None
        self.cols = None
        self.backbone = []
        self.seq_str = ""
        self.color_mode = "class"
        self.ss_list = []
        self.ribbon = False
        self.extra_chains = None
        self.salt_pairs = []
        self.arcs = []
        self.phi = []
        self.ligand = None
        self.show_stage = True
        self.explode = 0.0
        self.fog = False
        self.selected = None
        self.show_volume = True
        self.show_backbone = True
        self.slice_axis = "z"
        self.slice_z = None
        self.meta = ("", 0, 0)
        self._cache = {}
        self._press = None
        self._prev = None
        self._dragged = False
        self._proj = []
        self._proj_map = {}
        n = math.sqrt(0.42 ** 2 + 0.52 ** 2 + 0.74 ** 2)
        self.L = (-0.42 / n, 0.52 / n, 0.74 / n)

    def set_scene(self, occ, cols, backbone, show_volume, show_backbone,
                  slice_axis, slice_z, meta, seq_str,
                  color_mode="class", ss_list=None, ribbon=False,
                  extra_chains=None, explode=0.0, fog=False):
        self.occ, self.cols, self.backbone = occ, cols, backbone
        self.show_volume, self.show_backbone = show_volume, show_backbone
        self.slice_axis, self.slice_z = slice_axis, slice_z
        self.meta, self.seq_str = meta, seq_str
        self.color_mode = color_mode
        self.ss_list = ss_list or []
        self.ribbon = ribbon
        self.extra_chains = extra_chains
        self.explode, self.fog = explode, fog
        self.update()

    def set_overlay(self, salt_pairs, phi, ligand, arcs=None):
        self.salt_pairs = salt_pairs or []
        self.phi = list(phi) if phi is not None else []
        self.ligand = ligand
        self.arcs = arcs or []
        self.update()

    def set_selected(self, i):
        self.selected = i
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position()
            self._prev = e.position()
            self._dragged = False

    def mouseMoveEvent(self, e):
        if self._press is not None:
            d = e.position() - self._prev
            if abs(d.x()) + abs(d.y()) > 1.5:
                self._dragged = True
            self.yaw += d.x() * 0.008
            self.pitch = max(-1.5, min(1.5, self.pitch + d.y() * 0.008))
            self._prev = e.position()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._dragged and \
                e.button() == Qt.LeftButton:
            pos, best, bd = e.position(), -1, 16 * 16
            for x, y, i in self._proj:
                d2 = (x - pos.x()) ** 2 + (y - pos.y()) ** 2
                if d2 < bd:
                    bd, best = d2, i
            self.residuePicked.emit(best)
        self._press = None
        self._dragged = False

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.35, min(3.5, self.zoom * f))
        self.update()

    def _chain_color(self, i, cls):
        if self.color_mode == "rainbow":
            n = max(1, len(self.backbone) - 1)
            return QColor.fromHsvF(0.66 * (1.0 - i / n), 0.82, 0.96)
        if self.color_mode == "ss":
            s = self.ss_list[i] if i < len(self.ss_list) else "c"
            return QColor(SS_COLOR[s])
        if self.color_mode == "potential":
            if i < len(self.phi):
                v = self.phi[i]
                m = min(1.0, abs(v) / 1.2)
                if v >= 0:
                    return QColor.fromRgbF(0.32 + 0.28 * (1 - m),
                                           0.48 + 0.24 * (1 - m), 1.0)
                return QColor.fromRgbF(1.0, 0.36 + 0.26 * (1 - m),
                                       0.46 + 0.24 * (1 - m))
            return QColor("#aab4c2")
        return QColor.fromRgbF(*CLASS_RGB[cls])

    def _project_chain(self, chain, center, ca, sa, cb, sb, cx, cy, scale,
                       exf=1.0):
        if not chain:
            return None, None, None
        pts = (np.array([(b[0], b[1], b[2]) for b in chain], np.float32) -
               center) * exf
        x1 = pts[:, 0] * ca - pts[:, 1] * sa
        y1 = pts[:, 0] * sa + pts[:, 1] * ca
        y2 = y1 * cb + pts[:, 2] * sb
        z2 = -y1 * sb + pts[:, 2] * cb
        return cx + x1 * scale, cy - y2 * scale, z2

    def _draw_residue(self, qp, chain, SX, SY, i, rb, dim=1.0):
        nb = len(chain)
        _, _, _, cls = chain[i]
        c = self._chain_color(i, cls)
        if dim < 1.0:
            c = QColor(c.red(), c.green(), c.blue(), int(255 * dim))
        thin = self.color_mode in ("rainbow", "ss")
        if self.ribbon:
            s = self.ss_list[i] if i < len(self.ss_list) else "c"
            lw = max(2.0, rb * (1.9 if s == "h" else 1.3 if s == "e" else 0.8))
            path = QPainterPath()
            if nb == 2 or i == 0:
                path.moveTo(float(SX[i]), float(SY[i]))
                if nb >= 2:
                    path.lineTo(float((SX[i] + SX[i + 1]) / 2),
                                float((SY[i] + SY[i + 1]) / 2))
            elif i == nb - 1:
                path.moveTo(float((SX[i - 1] + SX[i]) / 2),
                            float((SY[i - 1] + SY[i]) / 2))
                path.lineTo(float(SX[i]), float(SY[i]))
            else:
                path.moveTo(float((SX[i - 1] + SX[i]) / 2),
                            float((SY[i - 1] + SY[i]) / 2))
                path.quadTo(float(SX[i]), float(SY[i]),
                            float((SX[i] + SX[i + 1]) / 2),
                            float((SY[i] + SY[i + 1]) / 2))
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(c, lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            qp.drawPath(path)
        else:
            pt = QPointF(float(SX[i]), float(SY[i]))
            if i > 0:
                qp.setBrush(Qt.NoBrush)
                if thin:
                    qp.setPen(QPen(c, max(2.0, rb * 0.8), Qt.SolidLine,
                                   Qt.RoundCap))
                else:
                    qp.setPen(QPen(QColor(c.red(), c.green(), c.blue(),
                                          int(130 * dim)),
                                   max(1.5, rb * 0.45), Qt.SolidLine,
                                   Qt.RoundCap))
                qp.drawLine(QPointF(float(SX[i - 1]), float(SY[i - 1])), pt)
            brr = rb * (0.55 if thin else 1.0)
            rg = QRadialGradient(pt + QPointF(-brr * 0.32, -brr * 0.36), brr * 1.6)
            rg.setColorAt(0.0, c.lighter(170))
            rg.setColorAt(0.55, c)
            rg.setColorAt(1.0, c.darker(200))
            qp.setBrush(QBrush(rg))
            qp.setPen(QPen(c.darker(230), 1))
            qp.drawEllipse(pt, brr, brr)

    def paintEvent(self, _ev):
        qp = QPainter(self)
        w, h = self.width(), self.height()
        bg = QRadialGradient(w * 0.5, h * 0.42, max(w, h) * 0.85)
        bg.setColorAt(0.0, QColor("#1a2330"))
        bg.setColorAt(1.0, QColor("#0a0d12"))
        qp.fillRect(0, 0, w, h, QBrush(bg))
        qp.setRenderHint(QPainter.Antialiasing, True)

        mode_name, step, seed = self.meta
        qp.setPen(QColor(140, 160, 185))
        qp.drawText(14, 24, f"{mode_name}    \u00b7    step {step}    \u00b7    seed {seed}")
        qp.setPen(QColor(82, 96, 114))
        qp.drawText(14, h - 12, "drag rotate \u00b7 wheel zoom \u00b7 space run \u00b7 "
                                "S step \u00b7 E extract \u00b7 R reset \u00b7 "
                                "click residue + 1-4 mutate")

        self._proj = []
        self._proj_map = {}
        if self.occ is None:
            return
        N = self.occ.shape[0]
        scale = min(w, h) / (N * 1.5) * self.zoom
        cx, cy = w * 0.5, h * 0.52
        ca, sa = math.cos(self.yaw), math.sin(self.yaw)
        cb, sb = math.cos(self.pitch), math.sin(self.pitch)
        center = (N - 1) / 2.0
        LX, LY, LZ = self.L
        exf = 1.0 + 0.55 * self.explode

        if self.show_stage:
            px, py, pz = center, (N - 1 + 0.9) - center, center
            x1 = px * ca - py * sa
            y1 = px * sa + py * ca
            y2 = y1 * cb + pz * sb
            sx = cx + x1 * scale
            sy = cy - y2 * scale
            R = N * scale * 0.8 * exf
            sh = QRadialGradient(QPointF(sx, sy), R)
            sh.setColorAt(0.0, QColor(0, 0, 0, 150))
            sh.setColorAt(0.7, QColor(0, 0, 0, 60))
            sh.setColorAt(1.0, QColor(0, 0, 0, 0))
            qp.setBrush(QBrush(sh))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(QPointF(sx, sy), R, R * 0.5)
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor(40, 52, 66, 120), 1))
            qp.drawEllipse(QPointF(sx, sy), R * 1.08, R * 0.56)

        ents, cube_data = [], None

        if self.show_volume:
            occd = self.occ
            if self.slice_z is not None:
                occd = self.occ.copy()
                sz = self.slice_z + 1
                if self.slice_axis == "x":
                    occd[:, :, sz:] = False
                elif self.slice_axis == "y":
                    occd[:, sz:, :] = False
                else:
                    occd[sz:, :, :] = False
            idxs = np.argwhere(occd)
            if len(idxs):
                base = (idxs.astype(np.float32) - center) * exf
                corners = base[:, None, :] + CORNERS * 0.85
                x1 = corners[..., 0] * ca - corners[..., 1] * sa
                y1 = corners[..., 0] * sa + corners[..., 1] * ca
                y2 = y1 * cb + corners[..., 2] * sb
                SX, SY = cx + x1 * scale, cy - y2 * scale
                bx1 = base[:, 0] * ca - base[:, 1] * sa
                by1 = base[:, 0] * sa + base[:, 1] * ca
                depth = -by1 * sb + base[:, 2] * cb
                ccols = self.cols[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
                fvis, fbr = [], []
                for nx, ny, nz in FACE_NRM:
                    fx1 = nx * ca - ny * sa
                    fy1 = nx * sa + ny * ca
                    fnz = fy1 * sb + nz * cb
                    fny = fy1 * cb - nz * sb
                    fvis.append(fnz > 0.02)
                    fbr.append(0.44 + 0.56 * max(0.0, fx1 * LX + fny * LY + fnz * LZ))
                if self.fog and len(depth) > 1:
                    dmin, dmax = float(depth.min()), float(depth.max())
                    span = max(dmax - dmin, 1e-6)
                    fogf = 0.35 + 0.65 * (1.0 - (depth - dmin) / span)
                    fogf = np.round(fogf * 10.0) / 10.0
                else:
                    fogf = np.ones(len(depth), np.float32)
                cube_data = (SX, SY, ccols, fvis, fbr, fogf)
                for i in np.argsort(depth):
                    ents.append((float(depth[i]), 0, int(i)))

        bSX = bSY = None
        if self.show_backbone and self.backbone:
            bSX, bSY, z2 = self._project_chain(self.backbone, center, ca, sa,
                                               cb, sb, cx, cy, scale, exf)
            for i in range(len(self.backbone)):
                self._proj.append((float(bSX[i]), float(bSY[i]), i))
                self._proj_map[i] = (float(bSX[i]), float(bSY[i]))
                ents.append((float(z2[i]), 1, i))

        if self.ligand is not None:
            lx, ly, lz, lr = self.ligand
            px, py, pz = lx - center, ly - center, lz - center
            x1 = px * ca - py * sa
            y1 = px * sa + py * ca
            y2 = y1 * cb + pz * sb
            z2 = -y1 * sb + pz * cb
            ents.append((float(z2), 2, (cx + x1 * scale, cy - y2 * scale, lr)))

        ents.sort(key=lambda e: e[0])
        rb = max(2.5, 0.44 * scale)

        if cube_data is not None:
            SX, SY, ccols, fvis, fbr, fogf = cube_data
        for d, kind, i in ents:
            if kind == 0:
                col = ccols[i]
                fq = float(fogf[i])
                ri = min(255, int(col[0] * 255 * fq))
                gi = min(255, int(col[1] * 255 * fq))
                bi = min(255, int(col[2] * 255 * fq))
                if ri + gi + bi == 0:
                    continue
                for f in range(6):
                    if not fvis[f]:
                        continue
                    key = (ri, gi, bi, f)
                    pair = self._cache.get(key)
                    if pair is None:
                        br = fbr[f]
                        r = min(255, int(ri * br))
                        g = min(255, int(gi * br))
                        b = min(255, int(bi * br))
                        pair = (QColor(r, g, b),
                                QColor(int(r * 0.72), int(g * 0.72), int(b * 0.72)))
                        if len(self._cache) > 8192:
                            self._cache.clear()
                        self._cache[key] = pair
                    qp.setBrush(pair[0])
                    qp.setPen(QPen(pair[1], 1))
                    qp.drawPolygon(QPolygonF([QPointF(SX[i, k], SY[i, k])
                                              for k in FACE_IDX[f]]))
            elif kind == 1:
                self._draw_residue(qp, self.backbone, bSX, bSY, i, rb, 1.0)
            else:
                sx, sy, lr = i
                R = max(3.0, lr * scale * 1.35)
                pt = QPointF(sx, sy)
                rg = QRadialGradient(pt + QPointF(-R * 0.3, -R * 0.35), R * 1.7)
                rg.setColorAt(0.0, QColor(255, 232, 250))
                rg.setColorAt(0.4, QColor(255, 110, 225, 220))
                rg.setColorAt(1.0, QColor(255, 60, 200, 30))
                qp.setBrush(QBrush(rg))
                qp.setPen(QPen(QColor(255, 140, 230, 140), 1))
                qp.drawEllipse(pt, R, R)
                qp.setPen(QColor(255, 175, 235))
                qp.drawText(QPointF(sx + R * 0.8, sy - R * 0.8), "LIG")

        if self.show_backbone and self.extra_chains:
            for ch in self.extra_chains:
                SXc, SYc, _ = self._project_chain(ch, center, ca, sa, cb, sb,
                                                  cx, cy, scale, exf)
                if SXc is None:
                    continue
                for i in range(len(ch)):
                    self._draw_residue(qp, ch, SXc, SYc, i, rb, 0.45)

        if self.show_backbone and self.arcs and self._proj_map:
            qp.setBrush(Qt.NoBrush)
            for (i, j, sep) in self.arcs:
                if i in self._proj_map and j in self._proj_map:
                    x1, y1 = self._proj_map[i]
                    x2, y2 = self._proj_map[j]
                    L = math.hypot(x2 - x1, y2 - y1)
                    if L < 4:
                        continue
                    col = QColor("#e8b34a" if sep < 16 else "#4dc9a3"
                                 if sep < 30 else "#5c94f2")
                    col.setAlpha(70)
                    path = QPainterPath(QPointF(x1, y1))
                    path.quadTo(QPointF((x1 + x2) / 2,
                                        (y1 + y2) / 2 - 0.22 * L),
                                QPointF(x2, y2))
                    qp.setPen(QPen(col, 1))
                    qp.drawPath(path)

        if self.salt_pairs and self.show_backbone and self._proj_map:
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor(255, 122, 224, 170), 1.5, Qt.DashLine))
            for (i, j) in self.salt_pairs:
                if i in self._proj_map and j in self._proj_map:
                    x1, y1 = self._proj_map[i]
                    x2, y2 = self._proj_map[j]
                    qp.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        if self.show_backbone and self.ribbon and bSX is not None and \
                len(self.backbone) >= 3:
            i = len(self.backbone) - 1
            cls = self.backbone[i][3]
            c = self._chain_color(i, cls)
            dx = float(bSX[i] - bSX[i - 2])
            dy = float(bSY[i] - bSY[i - 2])
            dl = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dl, dy / dl
            px, py = -uy, ux
            tipx, tipy = bSX[i] + ux * rb * 2.2, bSY[i] + uy * rb * 2.2
            b1x = bSX[i] - ux * rb * 0.4 + px * rb * 1.1
            b1y = bSY[i] - uy * rb * 0.4 + py * rb * 1.1
            b2x = bSX[i] - ux * rb * 0.4 - px * rb * 1.1
            b2y = bSY[i] - uy * rb * 0.4 - py * rb * 1.1
            qp.setBrush(QBrush(c))
            qp.setPen(QPen(c.darker(150), 1))
            qp.drawPolygon(QPolygonF([QPointF(tipx, tipy), QPointF(b1x, b1y),
                                      QPointF(b2x, b2y)]))

        if self.show_backbone and self.backbone and bSX is not None:
            qp.setPen(QColor(234, 242, 251))
            f = qp.font()
            f.setBold(True)
            qp.setFont(f)
            x0, y0 = self._proj_map.get(0, (0, 0))
            xe, ye = self._proj_map.get(len(self.backbone) - 1, (0, 0))
            qp.drawText(QPointF(x0 - 4, y0 - rb - 6), "N")
            qp.drawText(QPointF(xe - 2, ye + rb + 16), "C")

        if self.selected is not None and self.selected in self._proj_map:
            x, y = self._proj_map[self.selected]
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(QColor(240, 246, 252), 2))
            qp.drawEllipse(QPointF(x, y), rb * 1.9, rb * 1.9)
            if self.selected < len(self.seq_str):
                txt = f"{self.seq_str[self.selected]}{self.selected + 1}"
                cls = self.backbone[self.selected][3]
                acc = QColor(CLASS_HEX[cls])
                fm = qp.fontMetrics()
                tw = fm.horizontalAdvance(txt) + 10
                rx = min(w - tw - 4, max(4, x + rb * 1.6))
                ry = max(14, y - rb * 1.6 - 18)
                qp.setBrush(QColor(16, 21, 28, 225))
                qp.setPen(QPen(acc, 1))
                qp.drawRoundedRect(QRectF(rx, ry, tw, 17), 4, 4)
                qp.setPen(QColor(234, 242, 251))
                qp.drawText(QPointF(rx + 5, ry + 13), txt)


# ----------------------------------------------------------------------------
# Analysis widgets
# ----------------------------------------------------------------------------
def _panel(qp, w, h):
    qp.setBrush(QColor("#10151c"))
    qp.setPen(QPen(QColor("#1d2632"), 1))
    qp.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 8, 8)


class MetricsChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.series = []

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Live CA metrics (auto-normalised)")
        l, r, t, b = 10, 10, 22, 16
        pw, ph = w - l - r, h - t - b
        if not self.series or all(len(s[2]) < 2 for s in self.series):
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter,
                        "run the automaton to collect metrics")
            return
        qp.setPen(QPen(QColor("#1a2230"), 1))
        for f in (0.25, 0.5, 0.75):
            y = t + ph * f
            qp.drawLine(QPointF(l, y), QPointF(l + pw, y))
        for name, col, data, fmt in self.series:
            vals = list(data)
            if len(vals) < 2:
                continue
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-9:
                vmin, vmax = vmin - 1, vmax + 1
            poly = QPolygonF()
            for i, v in enumerate(vals):
                x = l + i / (len(vals) - 1) * pw
                y = t + (1 - (v - vmin) / (vmax - vmin)) * ph
                poly.append(QPointF(x, y))
            qp.setPen(QPen(QColor(col), 1.6))
            qp.drawPolyline(poly)
        qp.setFont(QFont(".", 8))
        x = w - 10
        for name, col, data, fmt in reversed(self.series):
            if not len(data):
                continue
            txt = f"{name} {format(data[-1], fmt)}"
            tw = qp.fontMetrics().horizontalAdvance(txt)
            x -= tw + 14
            qp.setPen(QColor("#8b9aad"))
            qp.drawText(QPointF(x, 14), txt)
            qp.setPen(QPen(QColor(col), 3))
            qp.drawLine(QPointF(x - 8, 10), QPointF(x - 3, 10))


class ContactMap(QWidget):
    residuePicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.seq = ""
        self.classes = []
        self.contact = np.zeros((0, 0), bool)
        self.sel = None
        self._geom = (0, 0, 0, 0)

    def set_data(self, seq, classes, contact):
        self.seq, self.classes, self.contact = seq, classes, contact
        self.update()

    def set_selected(self, i):
        self.sel = i
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        n = len(self.seq)
        if n < 2:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "no sequence yet")
            return
        l, t = 30, 18
        side = min(w - l - 8, h - t - 8)
        cell = side / n
        ox, oy = l, t
        self._geom = (ox, oy, cell, n)
        qp.setPen(QColor("#46586c"))
        for k in range(0, n, 10):
            qp.drawText(QPointF(ox + (k + 0.5) * cell, t - 5), str(k + 1))
            qp.drawText(QPointF(4, oy + (k + 0.5) * cell + 3), str(k + 1))
        for i, a in enumerate(self.seq):
            qp.fillRect(QRectF(ox + i * cell, oy + i * cell, cell, cell),
                        QColor(CLASS_HEX[self.classes[i]]))
        if self.contact.size:
            ii, jj = np.where(np.triu(self.contact, 3))
            for i, j in zip(ii, jj):
                sep = j - i
                col = ("#e8b34a" if sep < 8 else
                       "#4dc9a3" if sep < 16 else "#5c94f2")
                c = QColor(col)
                c.setAlpha(200)
                qp.fillRect(QRectF(ox + j * cell, oy + i * cell, cell, cell), c)
        if self.sel is not None and 0 <= self.sel < n:
            qp.setPen(QPen(QColor(234, 242, 251, 110), 1))
            qp.drawLine(QPointF(ox, oy + (self.sel + 0.5) * cell),
                        QPointF(ox + n * cell, oy + (self.sel + 0.5) * cell))
            qp.drawLine(QPointF(ox + (self.sel + 0.5) * cell, oy),
                        QPointF(ox + (self.sel + 0.5) * cell, oy + n * cell))
            qp.setPen(QPen(QColor("#eaf2fb"), 2))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(QRectF(ox + self.sel * cell - 1, oy + self.sel * cell - 1,
                               cell + 2, cell + 2))

    def _cell_at(self, pos):
        ox, oy, cell, n = self._geom
        i = int((pos.y() - oy) / cell)
        j = int((pos.x() - ox) / cell)
        if 0 <= i < n and 0 <= j < n:
            return i, j
        return None

    def mouseMoveEvent(self, e):
        ij = self._cell_at(e.position())
        if ij is None:
            return
        i, j = ij
        a = self.seq[i]
        if i == j:
            nc = int(self.contact[i].sum()) if self.contact.size else 0
            QToolTip.showText(e.globalPosition().toPoint(),
                              f"{i + 1} {a} \u00b7 {CLASS_NAMES[self.classes[i]]}"
                              f" \u00b7 {nc} contacts", self)
        elif self.contact.size and self.contact[i, j]:
            QToolTip.showText(e.globalPosition().toPoint(),
                              f"{i + 1}{a} \u2013 {j + 1}{self.seq[j]}"
                              f" \u00b7 sep {j - i}", self)

    def mousePressEvent(self, e):
        ij = self._cell_at(e.position())
        if ij:
            self.residuePicked.emit(ij[0])


class DotPlot(QWidget):
    residuePicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.seq = ""
        self.hits = None
        self.nw = 0
        self.sel = None
        self._geom = (0, 0, 0, 0)

    def set_data(self, seq):
        self.seq = seq
        self.sel = None
        self.hits = None
        n = len(seq)
        if n >= 10:
            w = 5
            nw = n - w + 1
            code = {a: k for k, a in enumerate("ARNDCEQGHILKMFPSTWYV")}
            S = np.zeros((n, 20), np.float32)
            for i, a in enumerate(seq):
                S[i, code[a]] = 1.0
            T = np.zeros((nw, nw), np.float32)
            for k in range(w):
                A = S[k:k + nw]
                T += A @ A.T
            T /= w
            iu = np.triu_indices(nw, 2)
            fr = T[iu]
            m = fr >= 0.8
            self.hits = (iu[0][m], iu[1][m], fr[m])
            self.nw = nw
        self.update()

    def set_selected(self, i):
        self.sel = i
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Self-similarity dot plot (window 5)")
        if self.hits is None:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "no sequence yet")
            return
        l, t = 30, 18
        side = min(w - l - 8, h - t - 8)
        cell = side / self.nw
        ox, oy = l, t
        self._geom = (ox, oy, cell, self.nw)
        qp.setPen(QColor("#46586c"))
        for k in range(0, self.nw, 20):
            qp.drawText(QPointF(ox + (k + 0.5) * cell, t - 5), str(k + 1))
            qp.drawText(QPointF(4, oy + (k + 0.5) * cell + 3), str(k + 1))
        qp.setPen(QPen(QColor("#243040"), 1))
        qp.drawLine(QPointF(ox, oy), QPointF(ox + self.nw * cell,
                                             oy + self.nw * cell))
        ii, jj, fr = self.hits
        for i, j, f in zip(ii, jj, fr):
            sep = j - i
            c = QColor("#e8b34a" if sep < 8 else "#4dc9a3" if sep < 16
                       else "#5c94f2")
            c.setAlpha(int(90 + 140 * (f - 0.8) / 0.2))
            qp.fillRect(QRectF(ox + j * cell, oy + i * cell, cell + 0.5,
                               cell + 0.5), c)
        if self.sel is not None and 0 <= self.sel < len(self.seq):
            k = self.sel
            qp.setPen(QPen(QColor(234, 242, 251, 120), 1))
            qp.drawLine(QPointF(ox, oy + (k + 0.5) * cell),
                        QPointF(ox + self.nw * cell, oy + (k + 0.5) * cell))
            qp.drawLine(QPointF(ox + (k + 0.5) * cell, oy),
                        QPointF(ox + (k + 0.5) * cell, oy + self.nw * cell))

    def mouseMoveEvent(self, e):
        if self.hits is None:
            return
        ox, oy, cell, nw = self._geom
        i = int((e.position().y() - oy) / cell)
        j = int((e.position().x() - ox) / cell)
        if 0 <= i < nw and 0 <= j < nw:
            QToolTip.showText(e.globalPosition().toPoint(),
                              f"{i + 1} \u2013 {j + 1} \u00b7 offset {j - i}", self)

    def mousePressEvent(self, e):
        if self.hits is None:
            return
        ox, oy, cell, nw = self._geom
        i = int((e.position().y() - oy) / cell)
        j = int((e.position().x() - ox) / cell)
        if 0 <= i < nw and 0 <= j < nw:
            self.residuePicked.emit(max(i, j))


def _slide(vals, w):
    out, h = [], w // 2
    for i in range(len(vals)):
        a, b = max(0, i - h), min(len(vals), i + h + 1)
        out.append(sum(vals[a:b]) / (b - a))
    return out


class ProfileChart(QWidget):
    residuePicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.seq, self.classes, self.hyd, self.chg = "", [], [], []
        self.dE = []
        self.sel = None
        self._geom = (0, 0, 0, 0)

    def set_data(self, seq, classes, dE=None):
        self.seq, self.classes = seq, classes or []
        self.dE = list(dE) if dE is not None else []
        n = len(seq)
        if n:
            hv = [AA_KD[a] for a in seq]
            cv = [1.0 if a in "KR" else -1.0 if a in "DE"
                  else 0.1 if a == "H" else 0.0 for a in seq]
            self.hyd, self.chg = _slide(hv, 9), _slide(cv, 9)
        else:
            self.hyd, self.chg = [], []
        self.update()

    def set_selected(self, i):
        self.sel = i
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        n = len(self.seq)
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "GRAVY \u00b7 charge \u00b7 \u0394E mutation scan")
        if n < 2:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "no sequence yet")
            return
        l, r, t = 30, 10, 22
        strip = 8
        b = 24 + strip
        pw, ph = w - l - r, h - t - b
        self._geom = (l, t, pw, ph)

        def ymap(v, lo, hi):
            return t + (1 - (v - lo) / (hi - lo)) * ph

        qp.setPen(QPen(QColor("#1a2230"), 1))
        qp.drawLine(QPointF(l, t), QPointF(l, t + ph))
        qp.setPen(QPen(QColor("#2a3644"), 1, Qt.DashLine))
        qp.drawLine(QPointF(l, ymap(0, -4.5, 4.5)), QPointF(l + pw, ymap(0, -4.5, 4.5)))
        qp.setPen(QColor("#46586c"))
        for v in (-4, -2, 0, 2, 4):
            qp.drawText(QPointF(4, ymap(v, -4.5, 4.5) + 3), str(v))
        for k in range(0, n, 20):
            x = l + k / (n - 1) * pw
            qp.setPen(QColor("#46586c"))
            qp.drawText(QPointF(x, h - strip - 10), str(k + 1))
        if len(self.dE) == n and self.dE:
            vmax = max(max(self.dE), 1e-6)
            for i, dv in enumerate(self.dE):
                f = min(1.0, dv / vmax)
                if f <= 0.02:
                    continue
                c = QColor(int(40 + 200 * f), int(70 - 40 * f), int(90 - 60 * f))
                bw = max(1.0, pw / n)
                qp.fillRect(QRectF(l + i / n * pw, t, bw, 3 + 9 * f), c)
        for data, col, lo, hi in ((self.hyd, "#ee9930", -4.5, 4.5),
                                  (self.chg, "#5c94f2", -1.0, 1.0)):
            poly = QPolygonF()
            for i, v in enumerate(data):
                poly.append(QPointF(l + i / (n - 1) * pw, ymap(v, lo, hi)))
            qp.setPen(QPen(QColor(col), 1.6))
            qp.drawPolyline(poly)
        for i, c in enumerate(self.classes):
            qp.fillRect(QRectF(l + i / n * pw, h - strip - 4, pw / n, strip),
                        QColor(CLASS_HEX[c]))
        if self.sel is not None and 0 <= self.sel < n:
            x = l + self.sel / (n - 1) * pw
            qp.setPen(QPen(QColor(234, 242, 251, 150), 1))
            qp.drawLine(QPointF(x, t), QPointF(x, t + ph))

    def _res_at(self, x):
        l, t, pw, ph = self._geom
        n = len(self.seq)
        if n and pw > 0:
            return max(0, min(n - 1, int((x - l) / pw * n)))
        return None

    def mousePressEvent(self, e):
        i = self._res_at(e.position().x())
        if i is not None:
            self.residuePicked.emit(i)


class RadialProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(190, 200)
        self.frac = None

    def set_data(self, frac):
        self.frac = frac
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Radial class profile (core \u2192 surface)")
        l, r, t, b = 30, 10, 22, 18
        pw, ph = w - l - r, h - t - b
        if self.frac is None or not len(self.frac):
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "no sequence yet")
            return
        bins = len(self.frac)
        qp.setPen(QPen(QColor("#1a2230"), 1))
        for f in (0.25, 0.5, 0.75):
            y = t + ph * f
            qp.drawLine(QPointF(l, y), QPointF(l + pw, y))
        qp.setPen(QPen(QColor("#2a3644"), 1))
        qp.drawLine(QPointF(l, t + ph / 2), QPointF(l + pw, t + ph / 2))
        qp.setPen(QColor("#46586c"))
        qp.drawText(QPointF(4, t + 9), "100%")
        qp.drawText(QPointF(8, t + ph + 4), "0")
        qp.drawText(QPointF(l + pw / 2 - 4, h - 4), "0.5r")
        qp.drawText(QPointF(l + pw - 10, h - 4), "r")
        for k in range(4):
            poly = QPolygonF()
            for i in range(bins):
                x = l + i / max(1, bins - 1) * pw
                y = t + (1 - float(self.frac[i, k])) * ph
                poly.append(QPointF(x, y))
            qp.setPen(QPen(QColor(CLASS_HEX[k + 1]), 1.8))
            qp.drawPolyline(poly)
        x = w - 10
        for k in (3, 2, 1, 0):
            x -= 12
            qp.setPen(QColor(CLASS_HEX[k + 1]))
            qp.setBrush(QColor(CLASS_HEX[k + 1]))
            qp.drawRect(QRectF(x, 6, 7, 7))


class HelicalWheel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(180, 200)
        self.data = None

    def set_data(self, seq, classes):
        r = best_amphi_window(seq, 10)
        if r is None:
            self.data = None
        else:
            s, mu = r
            self.data = (seq[s:s + 10], classes[s:s + 10], s + 1, mu)
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Helical wheel (best \u03bcH window)")
        if self.data is None:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "sequence < 10 aa")
            return
        seq, classes, start, mu = self.data
        cx, cy = w / 2, (h + 14) / 2
        R = min(w, h) / 2 - 34
        hx = hy = 0.0
        nh = 0
        for k, c in enumerate(classes):
            if c == 1:
                a = math.radians(-90 + k * 100)
                hx += math.cos(a)
                hy += math.sin(a)
                nh += 1
        if nh:
            fa = math.degrees(math.atan2(hy, hx))
            pts = [QPointF(cx, cy)]
            a0 = fa - 65
            for step in range(0, 131, 5):
                ar = math.radians(a0 + step)
                pts.append(QPointF(cx + R * math.cos(ar), cy + R * math.sin(ar)))
            wedge = QColor(CLASS_HEX[1])
            wedge.setAlpha(36)
            qp.setBrush(wedge)
            qp.setPen(Qt.NoPen)
            qp.drawPolygon(QPolygonF(pts))
        qp.setBrush(Qt.NoBrush)
        qp.setPen(QPen(QColor("#2c3a4b"), 1.4))
        qp.drawEllipse(QPointF(cx, cy), R, R)
        qp.setPen(QPen(QColor("#1c2530"), 1))
        for k in range(10):
            a = math.radians(-90 + k * 100)
            qp.drawLine(QPointF(cx, cy),
                        QPointF(cx + R * math.cos(a), cy + R * math.sin(a)))
        f = qp.font()
        f.setBold(True)
        f.setPointSize(9)
        qp.setFont(f)
        rr = R * 0.72
        for k, (a, c) in enumerate(zip(seq, classes)):
            ang = math.radians(-90 + k * 100)
            x = cx + rr * math.cos(ang)
            y = cy + rr * math.sin(ang)
            col = QColor(CLASS_HEX[classes[k]])
            qp.setBrush(col)
            qp.setPen(QPen(col.darker(160), 1))
            qp.drawEllipse(QPointF(x, y), R * 0.19, R * 0.19)
            qp.setPen(QColor("#0d1117"))
            qp.drawText(QRectF(x - 10, y - 9, 20, 18), Qt.AlignCenter, a)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#8b9aad"))
        qp.drawText(QPointF(10, h - 8),
                    f"{start}\u2013{start + 9} \u00b7 \u03bcH {mu:.2f}"
                    f" \u00b7 face {'yes' if nh >= 4 else 'weak'}")


class SlicesPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 190)
        self.cg = None
        self.occ = None
        self.pocket = 0.0
        self.mid = 0
        self.n = 0

    def set_grid(self, cg, occ, pocket):
        self.cg, self.occ, self.pocket = cg, occ, float(pocket or 0.0)
        self.n = cg.shape[0] if cg is not None else 0
        self.mid = self.n // 2
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Orthogonal slices (centre) \u00b7 magenta = pocket")
        if self.cg is None:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter, "no lattice")
            return
        n = self.n
        mid = self.mid
        panels = [("Z", self.cg[mid], self.occ[mid]),
                  ("Y", self.cg[:, mid, :], self.occ[:, mid, :]),
                  ("X", self.cg[:, :, mid], self.occ[:, :, mid])]
        pw = (w - 40) / 3
        ph = h - 44
        s = max(20.0, min(pw, ph))
        cell = s / n
        ox0 = (w - 3 * s - 20) / 2
        for k, (nm, cg2, oc2) in enumerate(panels):
            ox = ox0 + k * (s + 10)
            oy = 26
            qp.setPen(QColor("#46586c"))
            qp.drawText(QPointF(ox, oy - 5), f"{nm} = {mid}")
            qp.setPen(QPen(QColor("#1d2632"), 1))
            qp.setBrush(QColor("#0b0f14"))
            qp.drawRect(QRectF(ox, oy, s, s))
            for i in range(n):
                for j in range(n):
                    if not oc2[i, j]:
                        continue
                    cls = int(cg2[i, j])
                    if cls == 0:
                        continue
                    qp.fillRect(QRectF(ox + j * cell, oy + i * cell,
                                       cell + 0.5, cell + 0.5),
                                QColor(CLASS_HEX[cls]))
            if self.pocket > 0:
                ccx = ox + (n / 2.0) * cell
                ccy = oy + (n / 2.0) * cell
                qp.setBrush(Qt.NoBrush)
                qp.setPen(QPen(QColor(255, 90, 220, 200), 1.4))
                qp.drawEllipse(QPointF(ccx, ccy), self.pocket * cell,
                               self.pocket * cell)


class DesignSpace(QWidget):
    loadRequested = Signal(int)
    XR = (-2.5, 2.5)
    YR = (3.0, 11.0)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 200)
        self.setMouseTracking(True)
        self.points = []
        self.current = None

    def set_data(self, entries, current_id=None):
        self.current = current_id
        pts = []
        for e in entries:
            try:
                gx = float(e["gravy"])
                py = float(e["pi"])
            except (KeyError, TypeError, ValueError):
                continue
            x = max(self.XR[0], min(self.XR[1], gx))
            y = max(self.YR[0], min(self.YR[1], py))
            sc = float(e.get("score", 0.0))
            pts.append((x, y, 2.5 + 4.0 * sc / 100.0,
                        QColor(MODE_COLOR.get(e.get("mode", "potts"), "#8b9aad")),
                        e.get("_id"), sc, e.get("len", 0), e.get("seed", 0)))
        self.points = pts
        self.update()

    def _to_px(self, x, y, l, t, pw, ph):
        fx = (x - self.XR[0]) / (self.XR[1] - self.XR[0])
        fy = (y - self.YR[0]) / (self.YR[1] - self.YR[0])
        return l + fx * pw, t + (1 - fy) * ph

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        _panel(qp, w, h)
        qp.setFont(QFont(".", 8))
        qp.setPen(QColor("#6fb3e0"))
        qp.drawText(10, 14, "Design space \u2014 GRAVY \u00d7 pI (size = score)")
        l, r, t, b = 34, 12, 22, 26
        pw, ph = w - l - r, h - t - b
        if not self.points:
            qp.setPen(QColor("#5a6674"))
            qp.drawText(self.rect(), Qt.AlignCenter,
                        "pin designs or run the Breeding Lab")
            return
        qp.setPen(QPen(QColor("#1a2230"), 1))
        for f in (0.25, 0.5, 0.75):
            qp.drawLine(QPointF(l, t + ph * f), QPointF(l + pw, t + ph * f))
            qp.drawLine(QPointF(l + pw * f, t), QPointF(l + pw * f, t + ph))
        qp.setPen(QColor("#46586c"))
        for v in (-2, -1, 0, 1, 2):
            x, _ = self._to_px(v, self.YR[0], l, t, pw, ph)
            qp.drawText(QPointF(x - 6, h - 14), str(v))
        for v in (4, 6, 8, 10):
            _, y = self._to_px(self.XR[0], v, l, t, pw, ph)
            qp.drawText(QPointF(6, y + 3), str(v))
        qp.drawText(QPointF(l + pw / 2 - 20, h - 3), "GRAVY")
        qp.save()
        qp.translate(10, t + ph / 2 + 18)
        qp.rotate(-90)
        qp.drawText(QPointF(0, 0), "pI")
        qp.restore()
        for x, y, rr, col, lid, sc, ln, sd in self.points:
            px, py = self._to_px(x, y, l, t, pw, ph)
            c = QColor(col)
            c.setAlpha(170)
            qp.setBrush(c)
            qp.setPen(QPen(col.darker(140), 1))
            qp.drawEllipse(QPointF(px, py), rr, rr)
            if lid is not None and lid == self.current:
                qp.setBrush(Qt.NoBrush)
                qp.setPen(QPen(QColor(240, 246, 252), 2))
                qp.drawEllipse(QPointF(px, py), rr + 3, rr + 3)
        seen = set()
        x = w - 12
        qp.setBrush(Qt.NoBrush)
        for px_, py_, rr_, col_, lid_, sc_, ln_, sd_ in self.points:
            key = col_.name()
            if key in seen:
                continue
            seen.add(key)
            x -= 14
            qp.setPen(Qt.NoPen)
            qp.setBrush(col_)
            qp.drawEllipse(QPointF(x + 4, 10), 4, 4)

    def mouseMoveEvent(self, e):
        l, r, t, b = 34, 12, 22, 26
        pw = self.width() - l - r
        ph = self.height() - t - b
        best, bd = None, 12 * 12
        for x, y, rr, col, lid, sc, ln, sd in self.points:
            px, py = self._to_px(x, y, l, t, pw, ph)
            d2 = (px - e.position().x()) ** 2 + (py - e.position().y()) ** 2
            if d2 < bd:
                bd, best = d2, (lid, sc, ln, sd)
        if best and best[0] is not None:
            QToolTip.showText(e.globalPosition().toPoint(),
                              f"score {best[1]:.1f} \u00b7 len {best[2]} \u00b7 "
                              f"seed {best[3]} \u2014 click to load", self)

    def mousePressEvent(self, e):
        l, r, t, b = 34, 12, 22, 26
        pw = self.width() - l - r
        ph = self.height() - t - b
        best, bd = None, 11 * 11
        for x, y, rr, col, lid, sc, ln, sd in self.points:
            px, py = self._to_px(x, y, l, t, pw, ph)
            d2 = (px - e.position().x()) ** 2 + (py - e.position().y()) ** 2
            if d2 < bd:
                bd, best = d2, lid
        if best is not None:
            self.loadRequested.emit(best)


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
class ParamSlider(QWidget):
    def __init__(self, text, lo, hi, val, tip=""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.name = QLabel(text)
        self.name.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(val)
        self.val = QLabel(str(val))
        self.val.setMinimumWidth(26)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider.valueChanged.connect(self.val.setNum)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)

    def set_label(self, text):
        self.name.setText(text)


def sequence_html(seq, ss=None, sel=None):
    if not seq:
        return ("<span style='color:#5a6674;'>\u2014 no sequence extracted yet \u2014<br>"
                "run the automaton; the backbone is traced through the 3D "
                "conformation.</span>")
    lines = []
    for start in range(0, len(seq), 10):
        chunk = seq[start:start + 10]
        idx = str(start + 1).rjust(4)
        letters = ""
        for k, a in enumerate(chunk):
            i = start + k
            bg = "background-color:#2b4257;" if i == sel else ""
            letters += (f'<span style="color:{AA_COLOR[a]};font-weight:700;{bg}">'
                        f"{a}</span>")
        lines.append(f'<span style="color:#46586c;">{idx}</span>&nbsp;&nbsp;{letters}')
        if ss:
            marks = ""
            for k in range(len(chunk)):
                s = ss[start + k] if start + k < len(ss) else "c"
                marks += (f'<span style="color:{SS_COLOR[s]};font-weight:700;">'
                          f"{s}</span>")
            lines.append('<span style="color:#46586c;">&nbsp;&nbsp;&nbsp;&nbsp;'
                         f'&nbsp;&nbsp;</span>{marks}')
    return ("<div style=\"font-family:'Consolas','Menlo','DejaVu Sans Mono',monospace;"
            "font-size:14px;\">" + "<br>".join(lines) + "</div>")


def bar_html(pct, color, width=12):
    filled = max(1, int(round(pct / 100 * width))) if pct > 0 else 0
    block = "\u2588"
    return ('<span style="font-family:monospace;">'
            f'<span style="color:{color};">{block * filled}</span>'
            f'<span style="color:#242e3a;">{block * (width - filled)}</span></span>')


def _resample_snap(snap, L):
    m = len(snap)
    if m == L:
        return snap
    idx = np.linspace(0, m - 1, L)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, m - 1)
    fr = (idx - lo)[:, None]
    out = np.empty((L, 4), np.float32)
    out[:, :3] = snap[lo, :3] * (1 - fr) + snap[hi, :3] * fr
    out[:, 3] = snap[hi, 3]
    return out


LEGEND_HTML = "&nbsp;&nbsp;&nbsp;".join(
    f'<span style="color:{CLASS_HEX[c]};font-weight:700;">\u25a0</span>'
    f'<span style="color:#8b9aad;">&nbsp;{CLASS_NAMES[c]}</span>' for c in (1, 2, 3, 4))

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
STYLE = """
* { outline: none; }
QWidget { background-color:#0e1217; color:#c9d4e0; font-size:12px;
          font-family:'Segoe UI','Ubuntu','DejaVu Sans',sans-serif; }
QLabel { background:transparent; }
QGroupBox { border:1px solid #1e2733; border-radius:10px; margin-top:16px;
            padding:10px 8px 8px 8px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; top:2px; padding:0 6px;
                   color:#6fb3e0; background:#0e1217; }
QPushButton { background:#18202b; border:1px solid #263140; border-radius:7px;
              padding:6px 12px; color:#d5e2f0; }
QPushButton:hover { background:#1f2a38; border-color:#33465c; }
QPushButton:pressed { background:#141c26; }
QPushButton:disabled { color:#5a6674; }
QPushButton#runBtn { background:#123527; border-color:#1e5c41; color:#7be3ae; }
QPushButton#runBtn:checked { background:#3a1620; border-color:#6e2a38; color:#ff9aa8; }
QPushButton#polishBtn { background:#2a1f38; border-color:#5c3a8c; color:#d5b8ff; }
QPushButton#polishBtn:checked { background:#3a1620; border-color:#6e2a38; color:#ff9aa8; }
QComboBox { background:#141b24; border:1px solid #263140; border-radius:7px; padding:4px 8px; }
QComboBox QAbstractItemView { background:#141b24; border:1px solid #263140;
                              selection-background-color:#24425e; }
QSpinBox { background:#141b24; border:1px solid #263140; border-radius:6px;
           padding:3px 6px; selection-background-color:#24425e; }
QLineEdit { background:#141b24; border:1px solid #263140; border-radius:6px;
            padding:4px 6px; selection-background-color:#24425e; }
QSlider::groove:horizontal { height:5px; background:#1c2530; border-radius:2px; }
QSlider::sub-page:horizontal { background:#2e6f9e; border-radius:2px; }
QSlider::handle:horizontal { width:14px; margin:-5px 0; border-radius:7px; background:#6cc3f5; }
QSlider::handle:horizontal:hover { background:#8fd4ff; }
QCheckBox { spacing:7px; }
QCheckBox::indicator { width:14px; height:14px; border-radius:4px;
                       border:1px solid #2c3a4b; background:#141b24; }
QCheckBox::indicator:checked { background:#2e9e6f; border-color:#2e9e6f; }
QScrollArea { border:none; background:transparent; }
QTabWidget::pane { border:1px solid #1e2733; border-radius:8px; top:-1px; background:#0e1217; }
QTabBar::tab { background:#141b24; border:1px solid #263140; padding:5px 12px;
               border-top-left-radius:7px; border-top-right-radius:7px;
               margin-right:3px; color:#8b9aad; }
QTabBar::tab:selected { background:#182231; color:#6cc3f5; }
QTableWidget { background:#10151c; border:1px solid #1d2632; border-radius:8px;
               gridline-color:#1a2230; alternate-background-color:#121822; }
QTableWidget::item:selected { background:#24425e; }
QHeaderView::section { background:#141b24; border:none; border-bottom:1px solid #263140;
                       padding:4px 6px; color:#8b9aad; }
QProgressBar { background:#141b24; border:1px solid #263140; border-radius:6px;
               text-align:center; color:#c9d4e0; }
QProgressBar::chunk { background:#2e9e6f; border-radius:5px; }
QSplitter::handle { background:#0e1217; }
QToolTip { background:#141b24; color:#c9d4e0; border:1px solid #2c3a4b; padding:4px; }
#seqInner, #propsInner, #helpInner { background:#10151c; border:1px solid #1d2632;
                                     border-radius:8px; padding:8px; }
"""

# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProteinForge v7 \u2014 3D Cellular Automata Protein Designer")
        self.resize(1520, 960)
        self.setMinimumSize(1280, 820)

        self.engine = CAEngine(14, "potts", 424242, 0.6, 0, "core", 0.0)
        self.seq, self.path, self.classes, self.backbone, self.ss = "", [], [], [], []
        self.contact = np.zeros((0, 0), bool)
        self.score, self.score_parts = 0.0, {}
        self.long_contacts = 0
        self.dE = []
        self.radial = None
        self.salt_pairs = []
        self.arcs = []
        self.phi = []
        self.pocket_res = []
        self.selected = None
        self.library = []
        self._lib_seq = 0
        self._current_lib_id = None
        self._best_pinned = -1.0
        self.cys_suggestion = None
        self._demo_remaining = None
        self._steplog = deque(maxlen=96)
        self._snaps = []
        self._morph_t = None
        self._wfc_stopped = False

        self._polish_active = False
        self._p_seq = None
        self._p_cls = None
        self._p_counts = None
        self._p_q = 0.0
        self._p_E = 0.0
        self._p_E0 = 0.0
        self._p_best_E = 0.0
        self._p_best = None
        self._p_done = 0
        self._p_total = 4000
        self._p_contact = None
        self._p_voxels = None
        self._p_salt = 0
        self._p_exotic = 2.0
        self._p_excl = ""

        self.m_energy = deque(maxlen=480)
        self.m_burial = deque(maxlen=480)
        self.m_occ = deque(maxlen=480)

        self._build_ui()
        self._connect()
        self._update_param_visibility()
        self._restore_library()
        self._apply_config(PRESETS[0])
        self._sync_preset_selection(0)

        s = QSettings("ProteinForge", "ProteinForge")
        g = s.value("geometry")
        if g is not None:
            self.restoreGeometry(g)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(306)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(2, 2, 6, 2)
        lv.setSpacing(8)

        lv.addWidget(QLabel(
            '<span style="font-size:19px;font-weight:800;color:#eaf2fb;">Protein'
            '<span style="color:#6cc3f5;">Forge</span></span>'
            '<span style="color:#5a6674;font-weight:700;"> v7</span><br>'
            '<span style="font-size:11px;color:#7c8a99;">de-novo sequences from '
            '3D cellular automata</span>'))

        ca_box = QGroupBox("Cellular automaton")
        cal = QVBoxLayout(ca_box)
        cal.setSpacing(7)
        self.mode_combo = QComboBox()
        for key, name in MODES:
            self.mode_combo.addItem(name, key)
        cal.addWidget(self.mode_combo)
        row = QHBoxLayout()
        row.addWidget(QLabel("Lattice N\u00b3"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(8, 24)
        self.size_spin.setValue(14)
        row.addStretch(1)
        row.addWidget(self.size_spin)
        cal.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Symmetry"))
        self.sym_combo = QComboBox()
        for name, val in (("None", 0), ("C2 (180\u00b0 Z)", 2), ("C4 (90\u00b0 Z)", 4)):
            self.sym_combo.addItem(name, val)
        row.addStretch(1)
        row.addWidget(self.sym_combo)
        cal.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Field"))
        self.field_combo = QComboBox()
        for key, name in FIELDS:
            self.field_combo.addItem(name, key)
        self.field_combo.setToolTip(
            "core: H inside \u00b7 invert: H outside (sticky) \u00b7 "
            "band: H ring at mid-radius (bilayer).\n"
            "In WFC mode this picks the topology flavor: "
            "core=mixed \u00b7 band=sheet-rich \u00b7 invert=helix-rich.")
        row.addStretch(1)
        row.addWidget(self.field_combo)
        cal.addLayout(row)
        self.pocket_cb = QCheckBox("Ligand pocket (centre)")
        cal.addWidget(self.pocket_cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pocket radius"))
        self.pocket_spin = QSpinBox()
        self.pocket_spin.setRange(1, 6)
        self.pocket_spin.setValue(2)
        row.addStretch(1)
        row.addWidget(self.pocket_spin)
        cal.addLayout(row)
        self.density_ps = ParamSlider("Occupancy", 0, 100, 60)
        self.temp_ps = ParamSlider("Thermal noise", 0, 100, 25)
        self.thresh_ps = ParamSlider("Excitation threshold", 2, 13, 6)
        self.branch_ps = ParamSlider("Branching rate", 5, 100, 55)
        for ps in (self.density_ps, self.temp_ps, self.thresh_ps, self.branch_ps):
            cal.addWidget(ps)
        row = QHBoxLayout()
        row.addWidget(QLabel("Steps / frame"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 8)
        self.steps_spin.setValue(2)
        row.addStretch(1)
        row.addWidget(self.steps_spin)
        cal.addLayout(row)
        lv.addWidget(ca_box)

        evo_box = QGroupBox("Evolution")
        evl = QVBoxLayout(evo_box)
        evl.setSpacing(7)
        row = QHBoxLayout()
        row.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 9999999)
        self.seed_spin.setValue(424242)
        self.new_seed_btn = QPushButton("New seed")
        row.addWidget(self.seed_spin, 1)
        row.addWidget(self.new_seed_btn)
        evl.addLayout(row)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setCheckable(True)
        self.step_btn = QPushButton("Step")
        self.reseed_btn = QPushButton("Reseed")
        for b in (self.run_btn, self.step_btn, self.reseed_btn):
            b.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.step_btn)
        row.addWidget(self.reseed_btn)
        evl.addLayout(row)
        row = QHBoxLayout()
        self.turbo_cb = QCheckBox("Turbo (\u00d72, charts off)")
        self.breed_btn = QPushButton("Breeding Lab\u2026")
        row.addWidget(self.turbo_cb)
        row.addWidget(self.breed_btn, 1)
        evl.addLayout(row)
        lv.addWidget(evo_box)

        seq_box = QGroupBox("Sequence extraction")
        sql = QVBoxLayout(seq_box)
        sql.setSpacing(7)
        row = QHBoxLayout()
        row.addWidget(QLabel("Target length"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(20, 300)
        self.target_spin.setValue(120)
        self.target_spin.setSuffix(" aa")
        row.addStretch(1)
        row.addWidget(self.target_spin)
        sql.addLayout(row)
        self.exotic_ps = ParamSlider("Exotic residue bias", 10, 80, 25,
                                     "Up-weights W C F M Y within their class")
        sql.addWidget(self.exotic_ps)
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("Exclude residues, e.g.  W C")
        self.exclude_edit.editingFinished.connect(self.do_extract_refresh)
        sql.addWidget(self.exclude_edit)
        row = QHBoxLayout()
        self.auto_extract = QCheckBox("Auto-extract every")
        self.auto_extract.setChecked(True)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(2, 60)
        self.interval_spin.setValue(12)
        row.addWidget(self.auto_extract)
        row.addWidget(self.interval_spin)
        row.addWidget(QLabel("steps"))
        row.addStretch(1)
        sql.addLayout(row)
        self.extract_btn = QPushButton("Extract now  (E)")
        sql.addWidget(self.extract_btn)
        self.polish_btn = QPushButton("\u2728 Polish (anneal sequence)")
        self.polish_btn.setObjectName("polishBtn")
        self.polish_btn.setCheckable(True)
        self.polish_btn.setToolTip(
            "Simulated annealing on the sequence with the conformation fixed:\n"
            "minimises contact energy + charge + diversity. Watch the score climb.")
        sql.addWidget(self.polish_btn)
        lv.addWidget(seq_box)

        view_box = QGroupBox("View")
        vil = QVBoxLayout(view_box)
        vil.setSpacing(7)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Volume + backbone", "Volume only", "Backbone only"])
        vil.addWidget(self.view_combo)
        row = QHBoxLayout()
        row.addWidget(QLabel("Backbone"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Voxel beads", "Ribbon (smooth)"])
        row.addStretch(1)
        row.addWidget(self.style_combo)
        vil.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Chain color"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["By class", "Rainbow N\u2192C",
                                   "Secondary structure", "Electrostatic"])
        row.addStretch(1)
        row.addWidget(self.color_combo)
        vil.addLayout(row)
        self.assembly_cb = QCheckBox("Oligomer view (symmetry copies)")
        self.assembly_cb.setEnabled(False)
        vil.addWidget(self.assembly_cb)
        self.arcs_cb = QCheckBox("Long-range contact arcs")
        vil.addWidget(self.arcs_cb)
        self.core_cb = QCheckBox("Core only (hydrophobics)")
        vil.addWidget(self.core_cb)
        self.shell_cb = QCheckBox("Surface shell only")
        vil.addWidget(self.shell_cb)
        self.fog_cb = QCheckBox("Depth fog")
        vil.addWidget(self.fog_cb)
        self.auto_rotate = QCheckBox("Auto-rotate")
        self.auto_rotate.setChecked(True)
        vil.addWidget(self.auto_rotate)
        self.stage_cb = QCheckBox("Stage shadow")
        self.stage_cb.setChecked(True)
        vil.addWidget(self.stage_cb)
        row = QHBoxLayout()
        self.cutaway = QCheckBox("Cutaway")
        self.cutAxis_combo = QComboBox()
        self.cutAxis_combo.addItems(["Z", "Y", "X"])
        row.addWidget(self.cutaway)
        row.addWidget(self.cutAxis_combo)
        row.addStretch(1)
        vil.addLayout(row)
        self.slice_ps = ParamSlider("Slice", 0, 13, 13)
        self.slice_ps.slider.setEnabled(False)
        vil.addWidget(self.slice_ps)
        self.explode_ps = ParamSlider("Explode", 0, 60, 0,
                                      "Radially blow the structure apart")
        vil.addWidget(self.explode_ps)
        row = QHBoxLayout()
        self.replay_btn = QPushButton("\u25b6 Morph replay")
        self.png_btn = QPushButton("PNG")
        row.addWidget(self.replay_btn, 1)
        row.addWidget(self.png_btn)
        vil.addLayout(row)
        lv.addWidget(view_box)

        sess_box = QGroupBox("Session")
        sel_l = QVBoxLayout(sess_box)
        sel_l.setSpacing(7)
        row = QHBoxLayout()
        self.save_proj_btn = QPushButton("Save project")
        self.load_proj_btn = QPushButton("Load project")
        row.addWidget(self.save_proj_btn)
        row.addWidget(self.load_proj_btn, 1)
        sel_l.addLayout(row)
        self.npy_btn = QPushButton("Export lattice (.npy)")
        sel_l.addWidget(self.npy_btn)
        lv.addWidget(sess_box)

        lv.addWidget(QLabel(
            '<span style="color:#566374;font-size:11px;">Space run \u00b7 S step \u00b7 '
            'E extract \u00b7 P pin \u00b7 R reset \u00b7 click residue, then '
            '1\u20134 mutate \u00b7 \u2728 Polish to anneal</span>'))
        lv.addStretch(1)

        left_scroll.setWidget(left_w)
        root.addWidget(left_scroll)

        center_w = QWidget()
        cv = QVBoxLayout(center_w)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        self.viewport = Viewport3D()
        cv.addWidget(self.viewport, 1)
        legend = QLabel(LEGEND_HTML)
        legend.setAlignment(Qt.AlignCenter)
        cv.addWidget(legend)

        self.contact_map = ContactMap()
        self.metrics_chart = MetricsChart()
        self.metrics_chart.set_series([
            ("Pair energy", "#6cc3f5", self.m_energy, "+.2f"),
            ("H-core burial", "#ee9930", self.m_burial, ".0%"),
            ("Occupancy", "#4dc9a3", self.m_occ, ".0%")])
        self.profile_chart = ProfileChart()
        self.radial_chart = RadialProfile()
        self.wheel = HelicalWheel()
        self.slices_panel = SlicesPanel()
        self.dot_plot = DotPlot()
        self.design_space = DesignSpace()

        self.analysis_tabs = QTabWidget()
        for wdgt, title in ((self.contact_map, "Contacts"),
                            (self.dot_plot, "Dot plot"),
                            (self.profile_chart, "Profile"),
                            (self.radial_chart, "Radial"),
                            (self.wheel, "Helix wheel"),
                            (self.slices_panel, "Slices"),
                            (self.metrics_chart, "Metrics"),
                            (self.design_space, "Design map")):
            self.analysis_tabs.addTab(wdgt, title)

        split_v = QSplitter(Qt.Vertical)
        split_v.addWidget(center_w)
        split_v.addWidget(self.analysis_tabs)
        split_v.setChildrenCollapsible(False)
        split_v.setStretchFactor(0, 1)
        split_v.setSizes([560, 250])
        root.addWidget(split_v, 1)

        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        self.right_tabs = QTabWidget()

        seq_tab = QWidget()
        st = QVBoxLayout(seq_tab)
        st.setContentsMargins(6, 6, 6, 6)
        st.setSpacing(8)
        self.seq_label = QLabel()
        self.seq_label.setObjectName("seqInner")
        self.seq_label.setTextFormat(Qt.RichText)
        self.seq_label.setWordWrap(True)
        self.seq_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.seq_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        seq_scroll = QScrollArea()
        seq_scroll.setWidgetResizable(True)
        seq_scroll.setFrameShape(QFrame.NoFrame)
        seq_scroll.setWidget(self.seq_label)
        st.addWidget(seq_scroll, 1)
        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.fasta_btn = QPushButton("FASTA")
        self.pdb_btn = QPushButton("PDB")
        for b in (self.copy_btn, self.fasta_btn, self.pdb_btn):
            row.addWidget(b)
        st.addLayout(row)
        self.right_tabs.addTab(seq_tab, "Sequence")

        props_tab = QWidget()
        pt = QVBoxLayout(props_tab)
        pt.setContentsMargins(6, 6, 6, 6)
        pt.setSpacing(6)
        self.props_label = QLabel()
        self.props_label.setObjectName("propsInner")
        self.props_label.setTextFormat(Qt.RichText)
        self.props_label.setWordWrap(True)
        self.props_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        props_scroll = QScrollArea()
        props_scroll.setWidgetResizable(True)
        props_scroll.setFrameShape(QFrame.NoFrame)
        props_scroll.setWidget(self.props_label)
        pt.addWidget(props_scroll, 1)
        self.cys_label = QLabel("")
        self.cys_label.setWordWrap(True)
        self.cys_label.setStyleSheet("color:#c58af9;")
        pt.addWidget(self.cys_label)
        self.cys_btn = QPushButton("Apply Cys pair")
        self.cys_btn.setEnabled(False)
        pt.addWidget(self.cys_btn)
        self.right_tabs.addTab(props_tab, "Properties")

        lib_tab = QWidget()
        lt = QVBoxLayout(lib_tab)
        lt.setContentsMargins(6, 6, 6, 6)
        lt.setSpacing(8)
        self.lib_table = QTableWidget(0, 8)
        self.lib_table.setHorizontalHeaderLabels(
            ["Score", "Len", "GRAVY", "pI", "SS%", "Mode", "Seed", "Step"])
        self.lib_table.verticalHeader().setVisible(False)
        self.lib_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lib_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lib_table.setAlternatingRowColors(True)
        self.lib_table.setSortingEnabled(True)
        self.lib_table.horizontalHeader().setStretchLastSection(True)
        lt.addWidget(self.lib_table, 1)
        row = QHBoxLayout()
        self.pin_btn = QPushButton("Pin  (P)")
        self.load_btn = QPushButton("Load")
        self.del_btn = QPushButton("Delete")
        self.clear_btn = QPushButton("Clear")
        for b in (self.pin_btn, self.load_btn, self.del_btn, self.clear_btn):
            row.addWidget(b)
        lt.addLayout(row)
        row = QHBoxLayout()
        self.export_lib_btn = QPushButton("FASTA all")
        self.csv_btn = QPushButton("CSV")
        row.addWidget(self.export_lib_btn)
        row.addWidget(self.csv_btn)
        row.addStretch(1)
        lt.addLayout(row)
        row = QHBoxLayout()
        self.autopin_cb = QCheckBox("Auto-pin when score \u2265")
        self.autopin_cb.setChecked(True)
        self.autopin_spin = QSpinBox()
        self.autopin_spin.setRange(0, 100)
        self.autopin_spin.setValue(55)
        row.addWidget(self.autopin_cb)
        row.addWidget(self.autopin_spin)
        row.addStretch(1)
        lt.addLayout(row)
        self.right_tabs.addTab(lib_tab, "Library")

        self._build_examples_tab()

        rv.addWidget(self.right_tabs, 1)
        right_holder = QScrollArea()
        right_holder.setWidgetResizable(True)
        right_holder.setFixedWidth(400)
        right_holder.setFrameShape(QFrame.NoFrame)
        right_holder.setWidget(right_w)
        root.addWidget(right_holder)

        self.status = QLabel("")
        self.statusBar().addWidget(self.status, 1)

    def _build_examples_tab(self):
        ex_tab = QWidget()
        ex = QVBoxLayout(ex_tab)
        ex.setContentsMargins(6, 6, 6, 6)
        ex.setSpacing(6)
        self.preset_table = QTableWidget(0, 5)
        self.preset_table.setHorizontalHeaderLabels(
            ["Preset", "Rule", "N", "Sym", "Quick take"])
        self.preset_table.verticalHeader().setVisible(False)
        self.preset_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preset_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preset_table.horizontalHeader().setStretchLastSection(True)
        for p in PRESETS:
            r = self.preset_table.rowCount()
            self.preset_table.insertRow(r)
            vals = [p["name"], MODE_SHORT[p["mode"]], str(p["n"]),
                    {0: "\u2014", 2: "C2", 4: "C4"}[p.get("sym", 0)],
                    p["take"]]
            for c, v in enumerate(vals):
                self.preset_table.setItem(r, c, QTableWidgetItem(v))
        self.preset_table.setColumnWidth(0, 130)
        self.preset_table.setColumnWidth(1, 44)
        self.preset_table.setColumnWidth(2, 30)
        self.preset_table.setColumnWidth(3, 30)
        ex.addWidget(self.preset_table, 2)
        self.preset_desc = QLabel()
        self.preset_desc.setObjectName("helpInner")
        self.preset_desc.setTextFormat(Qt.RichText)
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        ex.addWidget(self.preset_desc, 3)
        row = QHBoxLayout()
        self.apply_preset_btn = QPushButton("Apply")
        self.demo_btn = QPushButton("\u25b6 Demo (150 steps)")
        self.surprise_btn = QPushButton("Surprise me")
        row.addWidget(self.apply_preset_btn)
        row.addWidget(self.demo_btn, 1)
        row.addWidget(self.surprise_btn)
        ex.addLayout(row)
        help_lbl = QLabel(self._help_html())
        help_lbl.setObjectName("helpInner")
        help_lbl.setTextFormat(Qt.RichText)
        help_lbl.setWordWrap(True)
        help_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        hscroll = QScrollArea()
        hscroll.setWidgetResizable(True)
        hscroll.setFrameShape(QFrame.NoFrame)
        hscroll.setWidget(help_lbl)
        ex.addWidget(hscroll, 4)
        self.right_tabs.addTab(ex_tab, "Examples")

    def _help_html(self):
        preset_list = "".join(
            f"<li><b>{p['name']}</b> \u2014 {p['take']}.</li>" for p in PRESETS)
        return (
            "<div style='font-size:12px;'>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>Workflow</h3>"
            "<ol>"
            "<li><b>Generate</b> \u2014 pick a preset or tune the CA, run, "
            "extract.</li>"
            "<li><b>Polish</b> \u2014 press <b>\u2728 Polish</b>: simulated "
            "annealing mutates the sequence (conformation fixed) to minimise "
            "contact energy, charge imbalance and low diversity.</li>"
            "<li><b>Edit</b> \u2014 click any residue, then press "
            "<b>1\u20134</b> to reclassify it (instantly rescored).</li>"
            "<li><b>Collect</b> \u2014 pin (P), auto-pin, or Breeding Lab; "
            "compare designs in the <b>Design map</b> tab and Library.</li>"
            "</ol>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>Wave Function Collapse</h3>"
            "<p>Each cell is a <b>superposition</b> of 9 protein tiles: void, "
            "hydrophobic core, helix-H / helix-P (rod tiles along Z), "
            "sheet-H / sheet-P (slab tiles in XZ), polar surface, surface+ and "
            "surface\u2212. Observation picks the <b>minimum-entropy</b> cell, "
            "samples a tile (radial prior + neighbour coherence), and "
            "<b>constraint propagation</b> prunes impossible tiles across the "
            "lattice. The pale cloud is unobserved \u2014 watch the wavefront "
            "crystallize a fold with rods, slabs, a packed core and salt-"
            "bridged surface. <b>Secondary structure is ground truth</b> in "
            "this mode (it drives the ribbon, SS track and PDB HELIX records), "
            "and residue choice follows a per-role grammar (core\u2192LIVAFM, "
            "helix\u2192LAVIM/EKQS, sheet\u2192VIFTY, surface\u2192STNQ, "
            "+\u2192KR, \u2212\u2192DE). Field flavor: core=mixed, "
            "band=sheet-rich, invert=helix-rich. Contradictions are repaired "
            "softly (counted in the status bar).</p>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>Rules (8)</h3>"
            "<ul>"
            "<li><b>Potts</b> \u2014 hydrophobic collapse</li>"
            "<li><b>WFC</b> \u2014 wave function collapse (protein tiles)</li>"
            "<li><b>Excite</b> \u2014 scroll waves \u00b7 <b>Turing</b> \u2014 "
            "Gray\u2013Scott \u00b7 <b>Lenia</b> \u2014 continuous gliders</li>"
            "<li><b>Cyclic</b> \u2014 spiral arms \u00b7 <b>Block</b> \u2014 "
            "Schelling domains \u00b7 <b>DLA</b> \u2014 dendrites</li>"
            "</ul>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>Visuals</h3>"
            "<ul>"
            "<li><b>Explode</b> slider \u2014 radially blow the structure "
            "apart (voxels + backbone) to see inside.</li>"
            "<li><b>Depth fog</b> \u2014 distant voxels fade for depth cues.</li>"
            "<li>Ribbon / beads, rainbow / SS / electrostatic coloring, "
            "contact arcs, salt bridges, morph replay, stage shadow.</li>"
            "</ul>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>All presets</h3>"
            f"<ul>{preset_list}</ul>"
            "<h3 style='color:#6fb3e0;margin:4px 0;'>Honesty box</h3>"
            "<p style='color:#8b9aad;'>Toy generative models: heuristic mapping, "
            "no experimental validation. For art, education, and as seeds for "
            "real design tools \u2014 not the bench.</p>"
            "</div>")

    # ------------------------------------------------------------ wiring ----
    def _connect(self):
        self._make_shortcuts()
        self.mode_combo.currentIndexChanged.connect(self.on_mode)
        self.size_spin.valueChanged.connect(self.on_size)
        self.sym_combo.currentIndexChanged.connect(self.on_symmetry)
        self.field_combo.currentIndexChanged.connect(self.on_field)
        self.pocket_cb.toggled.connect(lambda _: self.on_pocket())
        self.pocket_spin.valueChanged.connect(lambda _: self.on_pocket())
        self.density_ps.slider.valueChanged.connect(lambda _: self.reseed())
        self.seed_spin.valueChanged.connect(lambda _: self.reseed())
        self.new_seed_btn.clicked.connect(
            lambda: self.seed_spin.setValue(random.randint(1, 9999999)))
        self.run_btn.toggled.connect(
            lambda on: self.run_btn.setText("Pause" if on else "Run"))
        self.step_btn.clicked.connect(self.on_step)
        self.reseed_btn.clicked.connect(self.reseed)
        self.breed_btn.clicked.connect(self.on_breeding)
        self.target_spin.valueChanged.connect(self.do_extract_refresh)
        self.exotic_ps.slider.sliderReleased.connect(self.do_extract_refresh)
        self.extract_btn.clicked.connect(self.do_extract_refresh)
        self.polish_btn.toggled.connect(lambda on: self._polish_toggle(on))
        self.copy_btn.clicked.connect(self.on_copy)
        self.fasta_btn.clicked.connect(self.on_export)
        self.pdb_btn.clicked.connect(self.on_export_pdb)
        self.png_btn.clicked.connect(self.on_png)
        self.npy_btn.clicked.connect(self.on_export_grid)
        self.save_proj_btn.clicked.connect(self.on_save_project)
        self.load_proj_btn.clicked.connect(self.on_load_project)
        self.view_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self.style_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self.color_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self.assembly_cb.toggled.connect(lambda _: self.refresh())
        self.arcs_cb.toggled.connect(lambda _: self.refresh())
        self.core_cb.toggled.connect(lambda _: self.refresh())
        self.shell_cb.toggled.connect(lambda _: self.refresh())
        self.fog_cb.toggled.connect(lambda _: self.refresh())
        self.stage_cb.toggled.connect(lambda _: self.refresh())
        self.explode_ps.slider.valueChanged.connect(lambda _: self.refresh())
        self.cutaway.toggled.connect(self._slice_changed)
        self.cutAxis_combo.currentIndexChanged.connect(lambda _: self.refresh())
        self.slice_ps.slider.valueChanged.connect(lambda _: self.refresh())
        self.replay_btn.clicked.connect(self.on_replay)

        self.viewport.residuePicked.connect(self._on_pick)
        self.contact_map.residuePicked.connect(self._on_pick)
        self.profile_chart.residuePicked.connect(self._on_pick)
        self.dot_plot.residuePicked.connect(self._on_pick)
        self.design_space.loadRequested.connect(self._load_lib_by_id)

        self.lib_table.doubleClicked.connect(self._load_selected)
        self.pin_btn.clicked.connect(lambda: self.pin_current())
        self.load_btn.clicked.connect(self._load_selected)
        self.del_btn.clicked.connect(self._delete_lib)
        self.clear_btn.clicked.connect(self._clear_lib)
        self.export_lib_btn.clicked.connect(self._export_library)
        self.csv_btn.clicked.connect(self.on_export_csv)
        self.cys_btn.clicked.connect(self.on_cys_apply)

        self.preset_table.itemSelectionChanged.connect(self._on_preset_sel)
        self.preset_table.doubleClicked.connect(lambda _: self.on_apply_preset())
        self.apply_preset_btn.clicked.connect(self.on_apply_preset)
        self.demo_btn.clicked.connect(self.on_demo)
        self.surprise_btn.clicked.connect(self.on_surprise)

    def _make_shortcuts(self):
        def guard(fn):
            def run():
                w = QApplication.focusWidget()
                if isinstance(w, (QSpinBox, QComboBox, QLineEdit)):
                    return
                fn()
            return run
        QShortcut(QKeySequence(Qt.Key_Space), self,
                  activated=guard(lambda: self.run_btn.setChecked(
                      not self.run_btn.isChecked())))
        QShortcut(QKeySequence(Qt.Key_S), self, activated=guard(self.on_step))
        QShortcut(QKeySequence(Qt.Key_E), self, activated=guard(self.do_extract_refresh))
        QShortcut(QKeySequence(Qt.Key_P), self, activated=guard(lambda: self.pin_current()))
        QShortcut(QKeySequence(Qt.Key_R), self, activated=guard(self.on_reset_view))
        for k in (1, 2, 3, 4):
            QShortcut(QKeySequence(str(k)), self,
                      activated=guard(lambda c=k: self.on_mutate_selected(c)))

    # ------------------------------------------------------------- helpers --
    def _slice_changed(self, on):
        self.slice_ps.slider.setEnabled(on)
        self.refresh()

    def _update_param_visibility(self):
        labels = PARAM_LABELS[self.mode_combo.currentData()]
        self.temp_ps.setVisible(labels["temp"] is not None)
        self.thresh_ps.setVisible(labels["thresh"] is not None)
        self.branch_ps.setVisible(labels["branch"] is not None)
        if labels["temp"]:
            self.temp_ps.set_label(labels["temp"])
        if labels["thresh"]:
            self.thresh_ps.set_label(labels["thresh"])
        if labels["branch"]:
            self.branch_ps.set_label(labels["branch"])

    def _params(self):
        t = self.temp_ps.slider.value() / 100
        b = self.branch_ps.slider.value() / 100
        return dict(temp=0.05 + 2.6 * t,
                    thresh=self.thresh_ps.slider.value(),
                    branch=0.05 + 0.95 * b,
                    flip_noise=0.3 + 3.4 * t,
                    feed=0.016 + 0.032 * t,
                    kill=0.050 + 0.014 * b,
                    dt=0.02 + 0.30 * t,
                    mu=0.12 + 0.30 * b,
                    noise01=t)

    def _exclude_set(self):
        return "".join(ch for ch in self.exclude_edit.text().upper()
                       if ch in AA_CLASS)

    def _clear_metrics(self):
        self.m_energy.clear()
        self.m_burial.clear()
        self.m_occ.clear()

    def _rebuild_engine(self):
        pocket = float(self.pocket_spin.value()) if self.pocket_cb.isChecked() else 0.0
        self.engine = CAEngine(
            self.size_spin.value(), self.mode_combo.currentData(),
            self.seed_spin.value(), self.density_ps.slider.value() / 100,
            self.sym_combo.currentData(), self.field_combo.currentData(), pocket)
        self._clear_metrics()
        self._best_pinned = -1.0
        self._snaps = []
        self._morph_t = None
        self._wfc_stopped = False
        self.replay_btn.setText("\u25b6 Morph replay")
        self.slice_ps.slider.setMaximum(self.engine.n - 1)
        self.slice_ps.slider.setValue(self.engine.n - 1)
        self.pocket_spin.setRange(1, max(2, self.engine.n // 2 - 1))
        self.assembly_cb.setEnabled(self.engine.symmetry in (2, 4))

    # ------------------------------------------------- config / presets -----
    def _current_config(self):
        return dict(mode=self.mode_combo.currentData(), n=self.size_spin.value(),
                    sym=self.sym_combo.currentData(),
                    field=self.field_combo.currentData(),
                    pocket=self.pocket_spin.value() if self.pocket_cb.isChecked() else 0,
                    occ=self.density_ps.slider.value(),
                    temp=self.temp_ps.slider.value(),
                    thresh=self.thresh_ps.slider.value(),
                    branch=self.branch_ps.slider.value(),
                    steps=self.steps_spin.value(),
                    target=self.target_spin.value(),
                    exotic=self.exotic_ps.slider.value(),
                    exclude=self._exclude_set(),
                    interval=self.interval_spin.value(),
                    cut=self.cutaway.isChecked(),
                    cutz=self.slice_ps.slider.value())

    def _apply_config(self, cfg):
        widgets = (self.mode_combo, self.size_spin, self.sym_combo,
                   self.field_combo, self.density_ps.slider, self.temp_ps.slider,
                   self.thresh_ps.slider, self.branch_ps.slider,
                   self.steps_spin, self.target_spin, self.exotic_ps.slider,
                   self.pocket_cb, self.pocket_spin)
        for w in widgets:
            w.blockSignals(True)
        mi = next(i for i, (k, _) in enumerate(MODES) if k == cfg.get("mode", "potts"))
        self.mode_combo.setCurrentIndex(mi)
        self.size_spin.setValue(cfg.get("n", 14))
        si = self.sym_combo.findData(cfg.get("sym", 0))
        self.sym_combo.setCurrentIndex(max(0, si))
        fi = self.field_combo.findData(cfg.get("field", "core"))
        self.field_combo.setCurrentIndex(max(0, fi))
        pk = int(cfg.get("pocket", 0) or 0)
        self.pocket_cb.setChecked(pk > 0)
        self.pocket_spin.setValue(max(1, pk))
        self.density_ps.slider.setValue(cfg.get("occ", 60))
        self.temp_ps.slider.setValue(cfg.get("temp", 25))
        self.thresh_ps.slider.setValue(cfg.get("thresh", 6))
        self.branch_ps.slider.setValue(cfg.get("branch", 55))
        self.steps_spin.setValue(cfg.get("steps", 2))
        self.target_spin.setValue(cfg.get("target", 120))
        self.exotic_ps.slider.setValue(cfg.get("exotic", 25))
        for w in widgets:
            w.blockSignals(False)
        self.exclude_edit.setText(cfg.get("exclude", ""))
        self.interval_spin.setValue(cfg.get("interval", 12))
        self._update_param_visibility()
        self._rebuild_engine()
        self.cutaway.setChecked(bool(cfg.get("cut", False)))
        if cfg.get("cut"):
            self.slice_ps.slider.setValue(cfg.get("cutz", self.engine.n // 2))
        self.do_extract()
        self.refresh()

    def _sync_preset_selection(self, row):
        self.preset_table.blockSignals(True)
        self.preset_table.selectRow(row)
        self.preset_table.blockSignals(False)
        self._on_preset_sel()

    def _on_preset_sel(self):
        r = self.preset_table.currentRow()
        if not (0 <= r < len(PRESETS)):
            return
        p = PRESETS[r]
        symtxt = {0: "no symmetry", 2: "C2 symmetry", 4: "C4 symmetry"}[p.get("sym", 0)]
        fieldtxt = dict(FIELDS)[p.get("field", "core")]
        pk = p.get("pocket", 0)
        pockettxt = f" \u00b7 pocket r={pk}" if pk else ""
        self.preset_desc.setText(
            f"<div style='font-size:13px;font-weight:700;color:#eaf2fb;'>"
            f"{p['name']}</div>"
            f"<div style='color:#7c8a99;'>{MODE_SHORT[p['mode']]} \u00b7 "
            f"lattice {p['n']}\u00b3 \u00b7 {symtxt} \u00b7 {fieldtxt}"
            f"{pockettxt} \u00b7 target {p['target']} aa</div><div>{p['desc']}</div>")

    def on_apply_preset(self):
        r = self.preset_table.currentRow()
        if 0 <= r < len(PRESETS):
            self._apply_config(PRESETS[r])
            self.statusBar().showMessage(f"Preset applied: {PRESETS[r]['name']}", 3000)

    def on_demo(self):
        r = self.preset_table.currentRow()
        if not (0 <= r < len(PRESETS)):
            return
        self._apply_config(PRESETS[r])
        self._demo_remaining = 150
        self.run_btn.setChecked(True)
        self.statusBar().showMessage(
            f"Demo: {PRESETS[r]['name']} \u2014 running 150 steps, then pinning\u2026", 4000)

    def on_surprise(self):
        r = random.randrange(len(PRESETS))
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(random.randint(1, 9999999))
        self.seed_spin.blockSignals(False)
        self._apply_config(PRESETS[r])
        self._sync_preset_selection(r)
        self.statusBar().showMessage(f"Surprise: {PRESETS[r]['name']}", 3000)

    # ------------------------------------------------------------- actions --
    def reseed(self):
        self.engine.reseed(self.seed_spin.value(), self.density_ps.slider.value() / 100)
        self._clear_metrics()
        self._best_pinned = -1.0
        self._wfc_stopped = False
        self.do_extract()
        self.refresh()

    def on_mode(self):
        self._update_param_visibility()
        self._rebuild_engine()
        self.do_extract()
        self.refresh()

    def on_size(self):
        self._rebuild_engine()
        self.do_extract()
        self.refresh()

    def on_symmetry(self):
        self.engine.set_symmetry(self.sym_combo.currentData())
        self.assembly_cb.setEnabled(self.engine.symmetry in (2, 4))
        self.do_extract()
        self.refresh()

    def on_field(self):
        self.engine.set_profile(self.field_combo.currentData())
        self._clear_metrics()
        self._wfc_stopped = False
        self.do_extract()
        self.refresh()

    def on_pocket(self):
        r = float(self.pocket_spin.value()) if self.pocket_cb.isChecked() else 0.0
        self.engine.set_pocket(r)
        self._clear_metrics()
        self._best_pinned = -1.0
        self._wfc_stopped = False
        self.do_extract()
        self.refresh()

    def on_step(self):
        self._advance(1)
        self.do_extract()
        self.refresh()

    def _advance(self, steps):
        for _ in range(steps):
            self.engine.step(self._params())
        e, b, o = self.engine.metrics()
        self.m_energy.append(e)
        self.m_burial.append(b)
        self.m_occ.append(o)
        self._steplog.append((time.monotonic(), steps))

    def on_reset_view(self):
        self.viewport.yaw = math.radians(35)
        self.viewport.pitch = math.radians(22)
        self.viewport.zoom = 1.0
        self.viewport.update()

    def do_extract_refresh(self):
        self.do_extract()
        self.refresh()

    def do_extract(self):
        if self._polish_active:
            self._polish_finish()
        path, seq, classes = self.engine.extract(
            int(self.target_spin.value()),
            self.exotic_ps.slider.value() / 10.0,
            self._exclude_set())
        self.path, self.seq, self.classes = path, seq, classes
        self.selected = None
        self.viewport.set_selected(None)
        if self.run_btn.isChecked() and seq:
            snap = np.array([[x, y, z, c] for (x, y, z), c in zip(path, classes)],
                            np.float32)
            if len(snap):
                self._snaps.append(snap)
                if len(self._snaps) > 60:
                    self._snaps.pop(0)
        self._recompute_analysis()
        if (self.autopin_cb.isChecked() and seq
                and self.score >= self.autopin_spin.value()
                and self.score > self._best_pinned):
            self.pin_current(auto=True)

    def _recompute_analysis(self):
        self.backbone = [(int(x), int(y), int(z), int(c))
                         for (x, y, z), c in zip(self.path, self.classes)]
        seq = self.seq
        roles = self.engine.get_roles()
        if roles and len(roles) == len(self.path) and self.path:
            self.ss = list(roles)            # structure-true (WFC)
        else:
            self.ss = assign_ss(self.path)
        self.contact = residue_contacts(self.path)
        self.score, self.score_parts = design_score(
            seq, self.path, self.classes, self.contact, self.ss)
        self.long_contacts = long_range_contacts(self.contact)
        self.dE = list(mutation_scan(self.classes, self.contact))
        self.radial = radial_profile(self.path, self.classes)
        self.salt_pairs = salt_bridges(seq, self.contact)
        self.arcs = contact_arcs(self.contact) if self.arcs_cb.isChecked() else []
        self.phi = residue_potential(seq, self.path)
        if self.engine.pocket_r > 0:
            self.pocket_res = pocket_residues(self.path, self.classes,
                                              self.engine.pocket_r, self.engine.n)
        else:
            self.pocket_res = []
        self.seq_label.setText(sequence_html(seq, self.ss, self.selected))
        self.props_label.setText(self._props_html())
        self.contact_map.set_data(seq, self.classes, self.contact)
        self.dot_plot.set_data(seq)
        self.profile_chart.set_data(seq, self.classes, self.dE)
        self.radial_chart.set_data(self.radial)
        self.wheel.set_data(seq, self.classes)
        self._update_cys_ui()

    def _update_cys_ui(self):
        sug = suggest_disulfide(self.seq, self.contact)
        self.cys_suggestion = sug
        if sug:
            sep, i, j = sug
            self.cys_label.setText(
                f"Disulfide idea: mutate {i + 1}{self.seq[i]} and "
                f"{j + 1}{self.seq[j]} \u2192 Cys (contact pair, separation {sep}).")
            self.cys_btn.setEnabled(True)
        else:
            self.cys_label.setText(
                "No disulfide candidate \u2014 needs a contacting pair with "
                "separation \u2265 10.")
            self.cys_btn.setEnabled(False)

    def on_cys_apply(self):
        if not self.cys_suggestion or not self.seq:
            return
        _, i, j = self.cys_suggestion
        seq = list(self.seq)
        seq[i] = seq[j] = "C"
        self.seq = "".join(seq)
        self.classes[i] = self.classes[j] = 1
        self._recompute_analysis()
        self.refresh()
        self.statusBar().showMessage(
            f"Engineered Cys pair {i + 1}\u2013{j + 1}", 3500)

    # ----------------------------------------------------- point mutations --
    def on_mutate_selected(self, c):
        if self._polish_active or not self.seq:
            return
        i = self.selected
        if i is None or not (0 <= i < len(self.seq)):
            self.statusBar().showMessage(
                "Click a residue first, then press 1\u20134 to mutate", 2500)
            return
        x, y, z = self.path[i]
        st = None
        if self.engine.mode == "wfc":
            st = int(self.engine._wfc_state_grid()[x, y, z])
        letter = pick_aa_role(c, st, x, y, z, self.engine.aa_salt,
                              self.exotic_ps.slider.value() / 10.0,
                              self._exclude_set()) if st is not None else \
            pick_aa(c, x, y, z, self.engine.aa_salt,
                    self.exotic_ps.slider.value() / 10.0, self._exclude_set())
        seq = list(self.seq)
        seq[i] = letter
        self.seq = "".join(seq)
        self.classes[i] = c
        self._recompute_analysis()
        self.refresh()
        self.statusBar().showMessage(
            f"Mutated position {i + 1} \u2192 {letter} "
            f"({CLASS_NAMES[c]}) \u00b7 score {self.score:.1f}", 3000)

    # ------------------------------------------------------------- polish ---
    def _polish_energy(self, seq, cls):
        n = len(seq)
        clsarr = np.asarray(cls)
        pair = 0.0
        if self._p_contact.size:
            pair = float((self._p_contact *
                          ENERGY_J[np.ix_(clsarr, clsarr)]).sum()) * 0.5
        q = net_charge_from_counts(Counter(seq), 7.0)
        usage = sum(v * v for v in Counter(cls).values())
        return pair + 0.06 * q * q + 0.5 * usage / n

    def _polish_toggle(self, on):
        if on:
            self._polish_start()
        else:
            self._polish_finish()

    def _polish_start(self):
        if self._polish_active or not self.seq or len(self.seq) < 10:
            self.polish_btn.setChecked(False)
            return
        if self.run_btn.isChecked():
            self.run_btn.setChecked(False)
        if self._morph_t is not None:
            self._morph_t = None
            self.replay_btn.setText("\u25b6 Morph replay")
        self._polish_active = True
        self.polish_btn.setText("\u25a0 Stop polish")
        self._p_seq = list(self.seq)
        self._p_cls = list(self.classes)
        self._p_contact = self.contact
        self._p_voxels = [(int(x), int(y), int(z)) for x, y, z in self.path]
        self._p_salt = self.engine.aa_salt
        self._p_exotic = self.exotic_ps.slider.value() / 10.0
        self._p_excl = self._exclude_set()
        self._p_counts = [0] * 5
        for c in self._p_cls:
            self._p_counts[c] += 1
        self._p_q = net_charge("".join(self._p_seq), 7.0)
        self._p_E = self._polish_energy("".join(self._p_seq), self._p_cls)
        self._p_E0 = self._p_E
        self._p_best_E = self._p_E
        self._p_best = (list(self._p_seq), list(self._p_cls))
        self._p_done = 0
        self._p_total = 4000
        self.statusBar().showMessage("Polishing\u2026", 2000)

    def _polish_chunk(self):
        if not self._polish_active:
            return
        rng = random.Random()
        n = len(self._p_seq)
        if n == 0:
            self._polish_finish()
            return
        J = ENERGY_J
        cls = self._p_cls
        C = self._p_contact
        for _ in range(350):
            i = rng.randrange(n)
            ci = cls[i]
            cj = rng.choice([k for k in (1, 2, 3, 4) if k != ci])
            dE = 0.0
            if C.size:
                row = C[i]
                js = np.where(row)[0]
                for j in js:
                    dE += float(J[cj, cls[j]] - J[ci, cls[j]])
            cnts = self._p_counts
            du = ((cnts[cj] + 1) ** 2 - cnts[cj] ** 2) - \
                 (cnts[ci] ** 2 - (cnts[ci] - 1) ** 2)
            dE += 0.5 * du / n
            a = self._p_seq[i]
            x, y, z = self._p_voxels[i]
            b = pick_aa(cj, x, y, z, self._p_salt, self._p_exotic, self._p_excl)
            q2 = self._p_q
            if b != a:
                dq = AA_CHARGE7.get(b, 0.0) - AA_CHARGE7.get(a, 0.0)
                q2 = self._p_q + dq
                dE += 0.06 * (q2 * q2 - self._p_q * self._p_q)
            T = 0.5 * math.pow(0.04, min(1.0, self._p_done / self._p_total))
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
                self._p_seq[i] = b
                cls[i] = cj
                cnts[cj] += 1
                cnts[ci] -= 1
                self._p_q = q2
                self._p_E += dE
                if self._p_E < self._p_best_E - 1e-9:
                    self._p_best_E = self._p_E
                    self._p_best = (list(self._p_seq), list(self._p_cls))
        self._p_done += 350
        self.seq_label.setText(sequence_html("".join(self._p_seq), self.ss, None))
        self.status.setText(
            f"polish {self._p_done}/{self._p_total}   \u00b7   E "
            f"{self._p_E:.1f} (was {self._p_E0:.1f})   \u00b7   best "
            f"{self._p_best_E:.1f}")
        if self._p_done >= self._p_total:
            self._polish_finish()

    def _polish_finish(self):
        if not self._polish_active:
            self.polish_btn.setChecked(False)
            return
        self._polish_active = False
        self.polish_btn.setChecked(False)
        self.polish_btn.setText("\u2728 Polish (anneal sequence)")
        self.seq, self.classes = self._p_best
        self._recompute_analysis()
        self.refresh()
        ds = self.score
        self.statusBar().showMessage(
            f"Polish finished \u2014 energy {self._p_E0:.1f} \u2192 "
            f"{self._p_best_E:.1f} \u00b7 score now {ds:.1f}", 4500)
        if (self.autopin_cb.isChecked() and self.seq
                and ds >= self.autopin_spin.value()
                and ds > self._best_pinned):
            self.pin_current(auto=True)

    # ------------------------------------------------- scene push / replay --
    def _assembly_chains(self):
        sym = self.engine.symmetry
        if not (self.assembly_cb.isChecked() and sym in (2, 4) and self.backbone):
            return None
        n1 = self.engine.n - 1
        if sym == 2:
            rots = [lambda x, y: (n1 - x, n1 - y)]
        else:
            rots = [lambda x, y: (y, n1 - x),
                    lambda x, y: (n1 - x, n1 - y),
                    lambda x, y: (n1 - y, x)]
        out = []
        for rot in rots:
            ch = []
            for (x, y, z, c) in self.backbone:
                nx, ny = rot(int(x), int(y))
                ch.append((nx, ny, int(z), int(c)))
            out.append(ch)
        return out

    def _push_scene(self, backbone=None):
        occ, cols = self.engine.render_arrays()
        cg = self.engine.class_grid()
        if self.core_cb.isChecked():
            occ = occ & (cg == 1)
        if self.shell_cb.isChecked():
            nb = self.engine._shift_sum(occ.astype(np.int8))
            occ = occ & (nb < 26)
        else:
            nb = self.engine._shift_sum(occ.astype(np.int8))
            cols = np.clip(cols * np.where(nb >= 26, 0.62, 1.0)
                           .astype(np.float32)[..., None], 0.0, 1.0)
        vm = self.view_combo.currentIndex()
        slice_z = self.slice_ps.slider.value() if self.cutaway.isChecked() else None
        color_mode = ("class", "rainbow", "ss", "potential")[
            self.color_combo.currentIndex()]
        bb = self.backbone if backbone is None else backbone
        self.viewport.set_scene(
            occ, cols, bb, vm in (0, 1), vm in (0, 2),
            self.cutAxis_combo.currentText().lower(), slice_z,
            (self.mode_combo.currentText(), self.engine.step_count,
             self.engine.seed),
            self.seq, color_mode, self.ss,
            self.style_combo.currentIndex() == 1,
            self._assembly_chains(),
            self.explode_ps.slider.value() / 100.0,
            self.fog_cb.isChecked())
        ligand = None
        if self.engine.pocket_r > 0:
            c = (self.engine.n - 1) / 2.0
            ligand = (c, c, c, self.engine.pocket_r)
        arcs = self.arcs if self.arcs_cb.isChecked() else []
        self.viewport.set_overlay(self.salt_pairs, self.phi, ligand, arcs)
        self.viewport.show_stage = self.stage_cb.isChecked()
        return occ

    def refresh(self):
        occ = self._push_scene()
        self.slices_panel.set_grid(self.engine.class_grid(),
                                   self.engine.render_arrays()[0],
                                   self.engine.pocket_r)
        now = time.monotonic()
        while self._steplog and now - self._steplog[0][0] > 2.0:
            self._steplog.popleft()
        sps = "\u2014"
        if self._steplog:
            span = now - self._steplog[0][0]
            total = sum(s for _, s in self._steplog)
            if span > 0.2:
                sps = f"{total / span:,.0f}"
        extra = ""
        if self.engine.mode == "wfc":
            pct, rep = self.engine.wfc_progress()
            extra = f"   \u00b7   WFC {pct * 100:.0f}% (repairs {rep})"
        self.status.setText(
            f"step {self.engine.step_count}   \u00b7   occupied "
            f"{int(occ.sum())}/{self.engine.n ** 3}   \u00b7   "
            f"{len(self.seq)} residues   \u00b7   score {self.score:.1f}   \u00b7   "
            f"{sps} steps/s{extra}   \u00b7   {self.engine.n}\u00b3   \u00b7   "
            f"{dict(FIELDS)[self.field_combo.currentData()]}")

    def on_replay(self):
        if self._morph_t is not None:
            self._morph_t = None
            self.replay_btn.setText("\u25b6 Morph replay")
            self.refresh()
            return
        if self.run_btn.isChecked():
            self.run_btn.setChecked(False)
        if len(self._snaps) < 2:
            self.statusBar().showMessage(
                "Need at least 2 snapshots \u2014 run the automaton first", 3000)
            return
        self._morph_t = 0.0
        self.replay_btn.setText("\u25a0 Stop replay")

    def tick(self):
        if self._polish_active:
            self._polish_chunk()
            return
        rotating = self.auto_rotate.isChecked()
        if self._morph_t is not None:
            if rotating:
                self.viewport.yaw += 0.005
            self._morph_t += 0.06
            snaps = self._snaps
            if len(snaps) < 2 or self._morph_t >= len(snaps) - 1:
                self._morph_t = None
                self.replay_btn.setText("\u25b6 Morph replay")
                self.refresh()
            else:
                k = int(self._morph_t)
                f = float(self._morph_t - k)
                A, B = snaps[k], snaps[k + 1]
                L = max(len(A), len(B))
                A2 = _resample_snap(A, L)
                B2 = _resample_snap(B, L)
                xyz = A2[:, :3] * (1.0 - f) + B2[:, :3] * f
                cls = B2[:, 3].astype(int)
                bb = [(float(x), float(y), float(z), int(c))
                      for (x, y, z), c in zip(xyz, cls)]
                self._push_scene(backbone=bb)
            return
        if rotating:
            self.viewport.yaw += 0.005
        if self._demo_remaining is not None:
            if not self.run_btn.isChecked():
                self._demo_remaining = None
            else:
                self._demo_remaining -= 1
                if self._demo_remaining <= 0:
                    self.run_btn.setChecked(False)
                    self._demo_remaining = None
                    if self.seq:
                        self.pin_current(auto=True)
                    self.statusBar().showMessage(
                        "Demo finished \u2014 design pinned to Library", 3500)
        if self.run_btn.isChecked():
            mult = 2 if self.turbo_cb.isChecked() else 1
            self._advance(self.steps_spin.value() * mult)
            if self.engine.mode == "wfc" and not self._wfc_stopped and \
                    self.engine.wfc_complete():
                self._wfc_stopped = True
                self.run_btn.setChecked(False)
                self.statusBar().showMessage(
                    "WFC wave fully collapsed \u2014 press E to extract, "
                    "\u2728 Polish to refine", 4500)
            if self.auto_extract.isChecked() and \
                    self.engine.step_count % max(1, self.interval_spin.value()) == 0:
                self.do_extract()
            if not self.turbo_cb.isChecked():
                self.metrics_chart.update()
            self.refresh()
        elif rotating:
            self.viewport.update()

    # ------------------------------------------------------------- picking --
    def _on_pick(self, i):
        self.selected = i if 0 <= i < len(self.seq) else None
        self.viewport.set_selected(self.selected)
        self.contact_map.set_selected(self.selected)
        self.profile_chart.set_selected(self.selected)
        self.dot_plot.set_selected(self.selected)
        self.seq_label.setText(sequence_html(self.seq, self.ss, self.selected))
        if self.selected is not None:
            a = self.seq[self.selected]
            c = self.classes[self.selected]
            nc = int(self.contact[self.selected].sum()) if self.contact.size else 0
            de = self.dE[self.selected] if self.selected < len(self.dE) else 0.0
            ph = self.phi[self.selected] if self.selected < len(self.phi) else 0.0
            self.status.setText(
                f"residue {self.selected + 1}{a} \u00b7 {CLASS_NAMES[c]} \u00b7 "
                f"{nc} contacts \u00b7 SSA '{self.ss[self.selected]}' \u00b7 "
                f"\u0394E {de:+.2f} \u00b7 \u03c6 {ph:+.2f} \u00b7 "
                f"press 1\u20134 to mutate")

    # ------------------------------------------------------------- library --
    def _next_lib_id(self):
        self._lib_seq += 1
        return self._lib_seq

    def _lib_entry(self):
        return {"seq": self.seq, "classes": [int(c) for c in self.classes],
                "path": [[int(x), int(y), int(z)] for x, y, z in self.path],
                "ss": list(self.ss), "score": self.score,
                "parts": dict(self.score_parts),
                "mode": self.mode_combo.currentData(), "seed": self.engine.seed,
                "step": self.engine.step_count, "len": len(self.seq),
                "gravy": sum(AA_KD[a] for a in self.seq) / max(1, len(self.seq)),
                "pi": estimate_pi(self.seq),
                "lc": self.long_contacts}

    def _populate_row(self, e):
        t = self.lib_table
        r = t.rowCount()
        t.insertRow(r)
        ss_pct = 100 * sum(1 for s in e["ss"] if s in "he") / max(1, e["len"])
        vals = [f"{e['score']:.1f}", str(e["len"]), f"{e['gravy']:+.3f}",
                f"{e['pi']:.2f}", f"{ss_pct:.0f}", e["mode"],
                str(e["seed"]), str(e["step"])]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if c in (0, 1, 2, 3, 4, 6, 7):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if c == 0:
                it.setForeground(QColor("#7be3ae" if e["score"] >= 70 else
                                       "#ee9930" if e["score"] >= 50 else
                                       "#ee5c77"))
            tip = " \u00b7 ".join(f"{k} {v:.1f}" for k, v in e.get("parts", {}).items())
            if e.get("goal") and e["goal"] != "Balanced":
                tip = f"[{e['goal']}] " + tip
            if tip:
                it.setToolTip(tip)
            t.setItem(r, c, it)
        t.item(r, 0).setData(Qt.UserRole, e["_id"])

    def _add_lib_row(self, e):
        self.lib_table.setSortingEnabled(False)
        self._populate_row(e)
        self.lib_table.setSortingEnabled(True)
        self._refresh_scatter()

    def _sync_table(self):
        t = self.lib_table
        t.setSortingEnabled(False)
        t.setRowCount(0)
        for e in self.library:
            self._populate_row(e)
        t.setSortingEnabled(True)
        self._refresh_scatter()

    def _refresh_scatter(self):
        self.design_space.set_data(self.library, self._current_lib_id)

    def _trim_library(self):
        while len(self.library) > 200:
            self.library.pop(0)
        self._sync_table()

    def pin_current(self, auto=False):
        if not self.seq:
            self.statusBar().showMessage("Nothing to pin yet", 2000)
            return
        e = self._lib_entry()
        e["_id"] = self._next_lib_id()
        self._best_pinned = max(self._best_pinned, e["score"])
        self.library.append(e)
        self._add_lib_row(e)
        self._trim_library()
        self._current_lib_id = e["_id"]
        self._refresh_scatter()
        if not auto:
            self.right_tabs.setCurrentIndex(2)
            self.statusBar().showMessage(
                f"Pinned design #{len(self.library)} (score {e['score']:.1f})", 2500)

    def _entry_by_row(self, r):
        it = self.lib_table.item(r, 0)
        if it is None:
            return None
        lid = it.data(Qt.UserRole)
        for e in self.library:
            if e.get("_id") == lid:
                return e
        return None

    def _load_lib_by_id(self, lid):
        for e in self.library:
            if e.get("_id") == lid:
                self._load_entry(e)
                return

    def _load_entry(self, e):
        self.seq = e["seq"]
        self.classes = list(e["classes"])
        self.path = [tuple(p) for p in e["path"]]
        self.ss = list(e["ss"])
        self.score, self.score_parts = e["score"], dict(e.get("parts", {}))
        self.long_contacts = e.get("lc", 0)
        self.selected = None
        self.viewport.set_selected(None)
        self._recompute_analysis()
        self._current_lib_id = e.get("_id")
        self._refresh_scatter()
        self.right_tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"Loaded library design \u00b7 score {e['score']:.1f} \u00b7 "
            f"seed {e['seed']} step {e['step']}", 3000)

    def _load_selected(self):
        r = self.lib_table.currentRow()
        if r >= 0:
            e = self._entry_by_row(r)
            if e:
                self._load_entry(e)

    def _delete_lib(self):
        r = self.lib_table.currentRow()
        if r >= 0:
            e = self._entry_by_row(r)
            if e:
                self.library.remove(e)
            self.lib_table.removeRow(r)
            self._refresh_scatter()

    def _clear_lib(self):
        self.library.clear()
        self.lib_table.setRowCount(0)
        self._refresh_scatter()

    def _export_library(self):
        if not self.library:
            self.statusBar().showMessage("Library is empty", 2000)
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export library FASTA",
                                            "proteinforge_library.fasta",
                                            "FASTA (*.fasta *.fa *.txt)")
        if not fn:
            return
        try:
            with open(fn, "w") as fh:
                for k, e in enumerate(self.library):
                    fh.write(f">design{k + 1} rule={e['mode']} seed={e['seed']} "
                             f"step={e['step']} score={e['score']:.1f} "
                             f"len={e['len']}\n")
                    s = e["seq"]
                    for i in range(0, len(s), 60):
                        fh.write(s[i:i + 60] + "\n")
            self.statusBar().showMessage(f"Saved {len(self.library)} designs", 3000)
        except OSError as ex:
            self.statusBar().showMessage(f"Save failed: {ex}", 4000)

    def on_export_csv(self):
        if not self.library:
            self.statusBar().showMessage("Library is empty", 2000)
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export library CSV",
                                            "proteinforge_library.csv",
                                            "CSV (*.csv)")
        if not fn:
            return
        try:
            with open(fn, "w") as fh:
                fh.write("score,len,gravy,pi,ss_pct,mode,seed,step,sequence\n")
                for e in self.library:
                    ss_pct = 100 * sum(1 for s in e["ss"] if s in "he") / \
                        max(1, e["len"])
                    fh.write(f"{e['score']:.2f},{e['len']},{e['gravy']:.4f},"
                             f"{e['pi']:.3f},{ss_pct:.1f},{e['mode']},"
                             f"{e['seed']},{e['step']},{e['seq']}\n")
            self.statusBar().showMessage(f"Saved {len(self.library)} rows", 3000)
        except OSError as ex:
            self.statusBar().showMessage(f"Save failed: {ex}", 4000)

    # ----------------------------------------------------------- breeding ---
    def _batch_config(self):
        return dict(n=self.size_spin.value(), mode=self.mode_combo.currentData(),
                    sym=self.sym_combo.currentData(),
                    field=self.field_combo.currentData(),
                    pocket=float(self.pocket_spin.value())
                    if self.pocket_cb.isChecked() else 0.0,
                    occ=self.density_ps.slider.value() / 100,
                    params=self._params(),
                    target=int(self.target_spin.value()),
                    exotic=self.exotic_ps.slider.value() / 10.0,
                    exclude=self._exclude_set())

    def _merge_batch(self, entries):
        if not entries:
            self.statusBar().showMessage("Breeding Lab: no results (cancelled?)", 3000)
            return
        for e in entries:
            e["path"] = [tuple(p) for p in e["path"]]
            e["_id"] = self._next_lib_id()
            self.library.append(e)
        self._trim_library()
        self.right_tabs.setCurrentIndex(2)
        self.statusBar().showMessage(
            f"Breeding Lab kept {len(entries)} unique designs "
            f"(best {entries[0]['score']:.1f})", 3500)

    def on_breeding(self):
        dlg = BreedingDialog(self)
        self._breed_dlg = dlg
        dlg.show()

    # ----------------------------------------------------------- session ----
    def on_save_project(self):
        data = {"app": "ProteinForge", "version": 7,
                "config": self._current_config(), "library": self.library}
        fn, _ = QFileDialog.getSaveFileName(self, "Save project",
                                            "proteinforge_project.json",
                                            "JSON (*.json)")
        if not fn:
            return
        try:
            with open(fn, "w") as fh:
                json.dump(data, fh)
            self.statusBar().showMessage(f"Project saved: {fn}", 3000)
        except OSError as ex:
            self.statusBar().showMessage(f"Save failed: {ex}", 4000)

    def on_load_project(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load project", "",
                                            "JSON (*.json)")
        if not fn:
            return
        try:
            with open(fn) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as ex:
            self.statusBar().showMessage(f"Load failed: {ex}", 4000)
            return
        cfg = data.get("config", {})
        self._apply_config(cfg)
        lib = data.get("library", [])
        self.library.clear()
        self.lib_table.setRowCount(0)
        for e in lib:
            if not isinstance(e, dict) or "seq" not in e:
                continue
            e["path"] = [tuple(p) for p in e.get("path", [])]
            e.setdefault("parts", {})
            e.setdefault("ss", [])
            e["score"] = float(e.get("score", 0.0))
            e["_id"] = self._next_lib_id()
            self.library.append(e)
        self._sync_table()
        self.statusBar().showMessage(
            f"Project loaded \u00b7 {len(self.library)} library designs", 3000)

    def on_export_grid(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export class lattice",
                                            "proteinforge_lattice.npy",
                                            "NumPy (*.npy)")
        if fn:
            np.save(fn, self.engine.class_grid())
            self.statusBar().showMessage(f"Saved {fn}", 3000)

    # ------------------------------------------------------------- outputs --
    def _props_html(self):
        seq, path = self.seq, self.path
        if not seq:
            return "<span style='color:#5a6674;'>\u2014</span>"
        n = len(seq)
        mw = sum(AA_MASS[a] for a in seq) + 18.015
        gravy = sum(AA_KD[a] for a in seq) / n
        pi = estimate_pi(seq)
        q7 = net_charge(seq, 7.0)
        arom = sum(seq.count(a) for a in "FWY") / n
        rg = radius_gyration(path)
        muH = hydrophobic_moment(seq)
        ent = shannon_entropy(seq)
        ss_pct = {s: 100 * self.ss.count(s) / n for s in "hec"}
        cpr = float(self.contact.sum(axis=1).mean()) if self.contact.size else 0.0
        sap = sap_proxy(seq, path)

        color = "#7be3ae" if self.score >= 70 else ("#ee9930" if self.score >= 50
                                                    else "#ee5c77")
        html = (f'<div style="font-size:16px;font-weight:800;color:{color};">'
                f'Design score: {self.score:.1f} / 100</div>')
        for k in PART_MAX:
            v = self.score_parts.get(k, 0.0)
            html += (f'<div style="margin-top:3px;"><span style="color:#8494a6;">'
                     f'{k}</span> {bar_html(v / PART_MAX[k] * 100, PART_COL[k])} '
                     f'<span style="color:#c9d4e0;">{v:.1f}/{PART_MAX[k]}</span></div>')

        rows = [("Length", f"{n} residues"),
                ("Molecular weight", f"{mw / 1000:.2f} kDa"),
                ("GRAVY hydropathy", f"{gravy:+.3f}"),
                ("Estimated pI", f"{pi:.2f}"),
                ("Net charge (pH 7)", f"{q7:+.1f}"),
                ("Aromaticity", f"{arom * 100:.1f}% F/W/Y"),
                ("Hydrophobic moment \u03bcH", f"{muH:.2f} (max, w=10)"),
                ("Sequence entropy", f"{ent:.2f} / {math.log(20):.2f}"),
                ("SS motif (h/e/c)", f"{ss_pct['h']:.0f}% / {ss_pct['e']:.0f}% / "
                                     f"{ss_pct['c']:.0f}%"),
                ("Contacts / residue", f"{cpr:.1f}"),
                ("Long-range contacts", str(self.long_contacts)),
                ("Salt bridges (K/R\u2013D/E)", str(len(self.salt_pairs))),
                ("SAP-like surface H", f"{sap:.2f} (0\u20134.2 scale)"),
                ("N-glyco sites (N-X-S/T)", str(glyco_sites(seq))),
                ("Rg (conformation)", f"{rg:.2f} vox \u2248 {rg * 3.8:.1f} \u00c5")]
        tr = "".join(
            f'<tr><td style="color:#8494a6;padding:2px 0;">{k}</td>'
            f'<td align="right" style="color:#eaf2fb;font-weight:600;">{v}</td></tr>'
            for k, v in rows)
        html += f'<table width="100%" style="margin-top:8px;">{tr}</table>'

        if self.engine.mode == "wfc":
            st = self.engine._wfc_state_grid()
            ok = self.engine.collapsed & (st != WFC_VOID)
            if ok.any():
                cnt = Counter(WFC_STATE_NAME[s] for s in st[ok])
                tops = " \u00b7 ".join(f"{k} {v}" for k, v in cnt.most_common(5))
                html += ('<div style="color:#5ee0c8;font-weight:600;margin-top:8px;">'
                         'WFC tiles</div>'
                         f'<div style="color:#c9d4e0;">{tops}</div>')

        if self.engine.pocket_r > 0 and self.pocket_res:
            cnt = Counter(c for _, c in self.pocket_res)
            q = cnt.get(3, 0) - cnt.get(4, 0)
            html += ('<div style="color:#ff7ae0;font-weight:600;margin-top:8px;">'
                     'Pocket (ligand site)</div>'
                     f'<div style="color:#c9d4e0;">{len(self.pocket_res)} lining '
                     f'residues \u00b7 net class charge {q:+d} \u00b7 '
                     f'{cnt.get(2, 0)} polar / {cnt.get(3, 0)}+ / '
                     f'{cnt.get(4, 0)}\u2212 / {cnt.get(1, 0)} H</div>')

        mo = motif_scan(seq)
        if mo:
            html += ('<div style="color:#6fb3e0;font-weight:600;margin-top:8px;">'
                     'Motif scan</div>')
            for name, poss in mo:
                html += (f'<div style="margin-top:2px;"><span style="color:#c9d4e0;">'
                         f'{name}</span> <span style="color:#8494a6;">at '
                         f'{", ".join(map(str, poss))}</span></div>')

        html += ('<div style="color:#6fb3e0;font-weight:600;margin-top:8px;">'
                 'Quality flags</div>')
        lvl_col = {"ok": "#7be3ae", "warn": "#ffd166", "bad": "#ff6f91"}
        pocket_n = len(self.pocket_res) if self.engine.pocket_r > 0 else -1
        for lvl, txt in quality_flags(seq, self.classes, self.long_contacts,
                                      sb=len(self.salt_pairs), pocket_n=pocket_n,
                                      sap=sap):
            html += (f'<div style="margin-top:2px;">'
                     f'<span style="color:{lvl_col[lvl]};font-weight:700;">'
                     f'\u25cf</span> <span style="color:#c9d4e0;">{txt}</span></div>')

        html += ('<div style="color:#6fb3e0;font-weight:600;margin-top:8px;">'
                 'Class composition</div>')
        cnt = Counter(seq)
        for cid in (1, 2, 3, 4):
            cc = sum(v for a, v in cnt.items() if AA_CLASS[a] == cid)
            pct = cc / n * 100
            html += (f'<div style="margin-top:3px;">'
                     f'<span style="color:#8494a6;">{CLASS_NAMES[cid]}</span> '
                     f'{bar_html(pct, CLASS_HEX[cid])} '
                     f'<span style="color:#c9d4e0;">{pct:.0f}%</span></div>')
        top = " \u00b7 ".join(f"{a} {v / n * 100:.0f}%" for a, v in cnt.most_common(6))
        html += (f'<div style="color:#8494a6;margin-top:8px;">Top residues: '
                 f'<span style="color:#c9d4e0;">{top}</span></div>')
        return html

    def on_copy(self):
        if self.seq:
            QApplication.clipboard().setText(self.seq)
            self.statusBar().showMessage("Sequence copied to clipboard", 2000)

    def on_export(self):
        if not self.seq:
            self.statusBar().showMessage("Nothing to export yet", 2000)
            return
        seq = self.seq
        gravy = sum(AA_KD[a] for a in seq) / len(seq)
        pi = estimate_pi(seq)
        default = (f"proteinforge_{self.mode_combo.currentData()}"
                   f"_seed{self.engine.seed}.fasta")
        fn, _ = QFileDialog.getSaveFileName(self, "Export FASTA", default,
                                            "FASTA (*.fasta *.fa *.txt)")
        if not fn:
            return
        header = (f">ProteinForge rule={self.mode_combo.currentData()} "
                  f"seed={self.engine.seed} step={self.engine.step_count} "
                  f"len={len(seq)} gravy={gravy:.3f} pI={pi:.2f} "
                  f"score={self.score:.1f}")
        try:
            with open(fn, "w") as fh:
                fh.write(header + "\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i + 60] + "\n")
            self.statusBar().showMessage(f"Saved {fn}", 3000)
        except OSError as e:
            self.statusBar().showMessage(f"Save failed: {e}", 4000)

    def on_export_pdb(self):
        if not self.seq:
            self.statusBar().showMessage("Nothing to export yet", 2000)
            return
        default = (f"proteinforge_{self.mode_combo.currentData()}"
                   f"_seed{self.engine.seed}.pdb")
        fn, _ = QFileDialog.getSaveFileName(self, "Export PDB (CA trace)", default,
                                            "PDB (*.pdb *.txt)")
        if not fn:
            return
        meta = (self.mode_combo.currentData(), self.engine.seed,
                self.engine.step_count, self.score)
        ligand = None
        if self.engine.pocket_r > 0:
            c = (self.engine.n - 1) / 2.0
            ligand = (c, c, c, self.engine.pocket_r)
        try:
            with open(fn, "w") as fh:
                fh.write(build_pdb(self.seq, self.path, self.classes, self.ss,
                                   meta, extra_chains=self._assembly_chains(),
                                   ligand=ligand))
            self.statusBar().showMessage(f"Saved {fn} \u2014 open in PyMOL/ChimeraX", 3000)
        except OSError as e:
            self.statusBar().showMessage(f"Save failed: {e}", 4000)

    def on_png(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save viewport image",
                                            "proteinforge.png", "PNG (*.png)")
        if fn:
            self.viewport.grab().save(fn)
            self.statusBar().showMessage(f"Saved {fn}", 3000)

    # ------------------------------------------------------------- lifecycle -
    def _restore_library(self):
        s = QSettings("ProteinForge", "ProteinForge")
        raw = s.value("library")
        if not raw:
            return
        try:
            lib = json.loads(raw)
        except (ValueError, TypeError):
            return
        for e in lib:
            if not isinstance(e, dict) or "seq" not in e:
                continue
            e["path"] = [tuple(p) for p in e.get("path", [])]
            e.setdefault("parts", {})
            e.setdefault("ss", [])
            e["score"] = float(e.get("score", 0.0))
            e["_id"] = self._next_lib_id()
            self.library.append(e)
        self._sync_table()

    def closeEvent(self, e):
        s = QSettings("ProteinForge", "ProteinForge")
        s.setValue("geometry", self.saveGeometry())
        try:
            s.setValue("library", json.dumps(self.library))
        except (TypeError, ValueError):
            pass
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    timer = QTimer(win)
    timer.setInterval(55)
    timer.timeout.connect(win.tick)
    timer.start()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()