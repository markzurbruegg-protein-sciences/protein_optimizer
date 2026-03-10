#!/usr/bin/env python
"""ProtSolM helper — runs full ProtSolM inference for a single protein.

Called by the protein_characterization pipeline step via subprocess.
Takes a PDB file path, builds the protein graph, extracts features,
runs ESM2 + ProtSSN GNN + fine-tuned classifier, and outputs prediction.

ProtSolM (Tan et al., IEEE BIBM 2024) fuses sequence, structure, and
engineered features via a multimodal architecture:
  - ESM2 (650M) protein language model embeddings
  - EGNN graph neural network on Cα contact graph (ProtSSN)
  - Handcrafted features: AA composition, GRAVY, SS composition,
    H-bonds, exposed residue fractions, pLDDT

Usage:
    python scripts/protsolm_helper.py input.pdb output.json
    python scripts/protsolm_helper.py input.pdb output.json --sequence MKVL...
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROTSOLM_DIR = os.path.join(PROJECT_ROOT, "external", "ProtSolM")
GNN_CHECKPOINT = os.path.join(PROTSOLM_DIR, "model", "protssn_k20_h512.pt")
FT_CHECKPOINT = os.path.join(PROTSOLM_DIR, "ckpt",
                              "feature512_norm_pp_attention1d_k20_h512_lr5e-4.pt")
NORM_FILE = os.path.join(PROTSOLM_DIR, "norm", "cath_k20_mean_attr.pt")
GNN_CONFIG_FILE = os.path.join(PROTSOLM_DIR, "src", "config", "egnn.yaml")

# Add ProtSolM to path
sys.path.insert(0, PROTSOLM_DIR)


# ── Feature Extraction (from get_feature.py) ──────────────────────

def compute_features_from_pdb(pdb_path: str) -> dict:
    """Extract all handcrafted features from a PDB file.

    Returns a dict with keys matching ProtSolM feature columns.
    """
    from Bio.PDB import PDBParser, DSSP
    import biotite.structure.io as bsio

    try:
        import mdtraj as md
        traj = md.load(pdb_path)
        hbonds = md.kabsch_sander(traj)
        hbonds_num = hbonds[0].nnz
    except Exception:
        hbonds_num = 0
        logger.warning("mdtraj H-bond calculation failed, using 0")

    # Extract sequence from PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    one_letter = {
        'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
        'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
        'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
        'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'
    }

    aa_seq = ""
    for chain in model:
        for residue in chain:
            resname = residue.get_resname()
            if resname in one_letter:
                aa_seq += one_letter[resname]

    # DSSP for secondary structure
    ss_alphabet_dic = {
        "H": "H", "G": "H", "E": "E", "B": "E",
        "I": "C", "T": "C", "S": "C", "L": "C", "-": "C", "P": "C"
    }

    dssp_ok = False
    try:
        dssp = DSSP(model, pdb_path)
        sec_structures = []
        rsa_values = []
        for key in dssp.keys():
            dssp_res = dssp[key]
            sec_structures.append(dssp_res[2])
            rsa_values.append(dssp_res[3])
        dssp_ok = True
    except Exception as e:
        logger.warning(f"BioPython DSSP failed: {e}. Trying pydssp...")

    if not dssp_ok:
        try:
            import pydssp
            # pydssp works with coordinate arrays
            # Extract N, CA, C, O coords
            n_coords_all, ca_coords_all, c_coords_all, o_coords_all = [], [], [], []
            for chain in model:
                for residue in chain:
                    if residue.get_resname() not in one_letter:
                        continue
                    atoms = {a.name: a.get_vector() for a in residue}
                    if all(k in atoms for k in ('N', 'CA', 'C', 'O')):
                        n_coords_all.append(list(atoms['N']))
                        ca_coords_all.append(list(atoms['CA']))
                        c_coords_all.append(list(atoms['C']))
                        o_coords_all.append(list(atoms['O']))

            if n_coords_all:
                coords_array = np.stack([
                    np.array(n_coords_all),
                    np.array(ca_coords_all),
                    np.array(c_coords_all),
                    np.array(o_coords_all),
                ], axis=1).astype(np.float32)  # (L, 4, 3)
                coords_tensor = torch.from_numpy(coords_array)
                dssp_result = pydssp.assign(coords_tensor)  # returns string like "HHHCEEE..."
                sec_structures = list(dssp_result)
                rsa_values = [0.5] * len(sec_structures)  # pydssp doesn't compute RSA
                dssp_ok = True
                logger.info(f"pydssp assigned SS for {len(sec_structures)} residues")
            else:
                raise ValueError("No complete backbone coords found")
        except Exception as e2:
            logger.warning(f"pydssp also failed: {e2}. Using default SS.")
            sec_structures = ["-"] * len(aa_seq)
            rsa_values = [0.5] * len(aa_seq)

    # pLDDT from B-factors
    try:
        struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
        plddt = float(struct.b_factor.mean())
    except Exception:
        plddt = 70.0  # reasonable default

    length = len(aa_seq)
    if length == 0:
        raise ValueError("No amino acids found in PDB")

    # AA composition features
    counts = {aa: aa_seq.count(aa) for aa in "CDERHNGPS"}
    amino_acid_hydropathy = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
        'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
        'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
        'Y': -1.3, 'V': 4.2
    }
    gravy = sum(amino_acid_hydropathy.get(aa, 0) for aa in aa_seq) / length

    # SS composition (8-state and 3-state)
    ss8_seq = "".join(s.replace("-", "L") for s in sec_structures)
    ss3_seq = "".join(ss_alphabet_dic.get(s, "C") for s in sec_structures)

    # Pad/truncate SS to match aa_seq length
    if len(ss8_seq) < length:
        ss8_seq += "L" * (length - len(ss8_seq))
        ss3_seq += "C" * (length - len(ss3_seq))
    ss8_seq = ss8_seq[:length]
    ss3_seq = ss3_seq[:length]
    rsa_values = rsa_values[:length]
    while len(rsa_values) < length:
        rsa_values.append(0.5)

    ss8_counts = {s: 0 for s in "GHIBETSP L".split() if s.strip()}
    ss8_labels = list("GHIBETSPL")
    for s in ss8_seq:
        if s in ss8_labels:
            ss8_counts[s] = ss8_counts.get(s, 0) + 1

    ss3_counts = {"H": 0, "E": 0, "C": 0}
    for s in ss3_seq:
        if s in ss3_counts:
            ss3_counts[s] += 1

    # Exposed residue fractions at various cutoffs
    cutoffs = [x / 100 for x in range(5, 105, 5)]
    exposed = [0] * 20
    for r in rsa_values:
        for j in range(20):
            if r >= cutoffs[j]:
                exposed[j] += 1

    features = {
        "1-C": counts.get("C", 0) / length,
        "1-D": counts.get("D", 0) / length,
        "1-E": counts.get("E", 0) / length,
        "1-R": counts.get("R", 0) / length,
        "1-H": counts.get("H", 0) / length,
        "Turn-forming residues fraction": (
            counts.get("N", 0) + counts.get("G", 0) +
            counts.get("P", 0) + counts.get("S", 0)
        ) / length,
        "GRAVY": gravy,
    }

    for s in "GHIBETSPL":
        features[f"ss8-{s}"] = ss8_counts.get(s, 0) / length

    for s in "HEC":
        features[f"ss3-{s}"] = ss3_counts.get(s, 0) / length

    features["Hydrogen bonds"] = hbonds_num
    features["Hydrogen bonds per 100 residues"] = hbonds_num * 100 / length

    for j in range(20):
        pct = (j + 1) * 5
        features[f"Exposed residues fraction by {pct}%"] = exposed[j] / length

    features["pLDDT"] = plddt

    return features, aa_seq


# ── Graph Construction (from SuperviseDataset) ────────────────────

def build_protein_graph(pdb_path: str, c_alpha_max_neighbors: int = 20,
                        cutoff: float = 30.0, seq_dist_cut: int = 64):
    """Build a torch_geometric Data object from a PDB file.

    Replicates SuperviseDataset.get_calpha_graph() inline.
    """
    from Bio.PDB import PDBParser, ShrakeRupley
    from torch_geometric.data import Data
    import scipy.spatial as spa
    from scipy.special import softmax

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    rec = structure[0]

    # Collect coordinates
    c_alpha_coords_list = []
    n_coords_list = []
    c_coords_list = []
    residues_clean = []

    one_letter = {
        'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
        'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
        'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
        'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'
    }

    for chain in rec:
        invalid_ids = []
        for residue in chain:
            if residue.get_resname() == 'HOH':
                invalid_ids.append(residue.get_id())
                continue
            ca, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    ca = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if ca is not None and n is not None and c is not None:
                c_alpha_coords_list.append(ca)
                n_coords_list.append(n)
                c_coords_list.append(c)
                residues_clean.append(residue)
            else:
                invalid_ids.append(residue.get_id())
        for rid in invalid_ids:
            chain.detach_child(rid)

    c_alpha_coords = np.array(c_alpha_coords_list)
    n_coords = np.array(n_coords_list)
    c_coords = np.array(c_coords_list)
    num_residues = len(c_alpha_coords)

    if num_residues <= 1:
        raise ValueError("Protein has ≤1 residue")

    # Node features: dihedral angles
    num_angle_type = 2
    angles = np.zeros((num_residues, num_angle_type))
    for i in range(num_residues - 1):
        angles[i, 0] = _dihedral(c_coords[i], n_coords[i],
                                  c_alpha_coords[i], n_coords[i + 1])
        angles[i, 1] = _dihedral(n_coords[i], c_alpha_coords[i],
                                  c_coords[i], n_coords[i + 1])

    scalar_features = np.zeros((num_residues, num_angle_type * 2))
    for i in range(num_angle_type):
        scalar_features[:, 2 * i] = np.sin(angles[:, i])
        scalar_features[:, 2 * i + 1] = np.cos(angles[:, i])

    # Residue features: one-hot + SASA + B-factor + dihedrals
    allowable_aa = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP',
        'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'
    ]

    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(rec, level="R")

    num_feature = 2 + scalar_features.shape[1]
    x = torch.zeros(num_residues, 20 + num_feature)

    for i, residue in enumerate(residues_clean):
        resname = residue.get_resname()
        idx = allowable_aa.index(resname) if resname in allowable_aa else len(allowable_aa) - 1
        if idx >= 20:
            idx = 20  # map non-standard to last standard
        # One-hot (20 dim)
        if idx < 20:
            oh = [0.0] * 20
            oh[idx] = 1.0
        else:
            oh = [0.0] * 20

        sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
        bfactor = 0.0
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
                break

        feat = oh + [sasa, bfactor] + list(scalar_features[i])
        x[i] = torch.tensor(feat, dtype=torch.float32)

    # Normalize SASA and B-factor columns
    for k in [20, 21]:
        mean = x[:, k].mean()
        std = x[:, k].std()
        x[:, k] = (x[:, k] - mean) / (std + 1e-9)

    # Position features
    pos = torch.from_numpy(c_alpha_coords.astype(np.float32))

    # Build kNN graph
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list, dst_list, dist_list = [], [], []
    mean_norm_list = []

    n_i_feat = np.zeros((num_residues, 3))
    u_i_feat = np.zeros((num_residues, 3))
    v_i_feat = np.zeros((num_residues, 3))

    for i in range(num_residues):
        u_i = (n_coords[i] - c_alpha_coords[i])
        u_norm = np.linalg.norm(u_i)
        if u_norm < 1e-8:
            u_i = np.array([1.0, 0.0, 0.0])
        else:
            u_i = u_i / u_norm

        t_i = (c_coords[i] - c_alpha_coords[i])
        t_norm = np.linalg.norm(t_i)
        if t_norm < 1e-8:
            t_i = np.array([0.0, 1.0, 0.0])
        else:
            t_i = t_i / t_norm

        n_i = np.cross(u_i, t_i)
        n_norm = np.linalg.norm(n_i)
        if n_norm < 1e-8:
            n_i = np.array([0.0, 0.0, 1.0])
        else:
            n_i = n_i / n_norm

        v_i = np.cross(n_i, u_i)
        n_i_feat[i] = n_i
        u_i_feat[i] = u_i
        v_i_feat[i] = v_i

    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        if i in dst:
            dst.remove(i)
        if len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(distances[i, :]))[1:c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]

        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)

        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(-valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]
        mean_vec = weights.dot(diff_vecs)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / (denominator + 1e-10)
        mean_norm_list.append(mean_vec_ratio_norm)

    # Edge features: sequence distance one-hot + distance + contact
    seq_edge = torch.abs(torch.tensor(src_list) - torch.tensor(dst_list)).reshape(-1, 1)
    seq_edge = torch.where(seq_edge > seq_dist_cut, seq_dist_cut, seq_edge)
    seq_edge_oh = F.one_hot(seq_edge.long(), num_classes=seq_dist_cut + 1).reshape(-1, seq_dist_cut + 1).float()

    contact_sig = torch.where(torch.tensor(dist_list) <= 8, 1.0, 0.0).reshape(-1, 1)
    dist_feat = _distance_featurizer(dist_list, divisor=4)

    edge_attr = torch.cat([seq_edge_oh, dist_feat, contact_sig], dim=-1)

    # Geometric edge features
    edge_feat_ori_list = []
    for idx in range(len(dist_list)):
        s, d = src_list[idx], dst_list[idx]
        basis = np.stack((n_i_feat[d], u_i_feat[d], v_i_feat[d]), axis=0)
        diff = c_alpha_coords[s] - c_alpha_coords[d]
        p_ij = basis.dot(diff)
        q_ij = basis.dot(n_i_feat[s])
        k_ij = basis.dot(u_i_feat[s])
        t_ij = basis.dot(v_i_feat[s])
        edge_feat_ori_list.append(np.concatenate([p_ij, q_ij, k_ij, t_ij]))

    edge_feat_ori = torch.from_numpy(np.stack(edge_feat_ori_list).astype(np.float32))
    edge_attr = torch.cat([edge_attr, edge_feat_ori], dim=-1)

    graph = Data(
        x=x,
        pos=pos,
        edge_attr=edge_attr,
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
        mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)),
    )

    return graph


def _dihedral(p0, p1, p2, p3):
    """Compute dihedral angle between four points."""
    b0 = np.array(p0) - np.array(p1)
    b1 = np.array(p2) - np.array(p1)
    b2 = np.array(p3) - np.array(p2)

    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return 0.0
    b1 = b1 / b1_norm

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def _distance_featurizer(dist_list, divisor=4):
    """RBF distance features."""
    length_scale_list = [1.5 ** x for x in range(15)]
    dist_arr = np.array(dist_list)
    transformed = np.array([
        np.exp(-((dist_arr / divisor) ** 2) / ls)
        for ls in length_scale_list
    ]).T
    return torch.from_numpy(transformed.astype(np.float32))


# ── Normalization (from dataset_utils.py) ────────────────────────

def normalize_graph(graph, norm_file: str, skip_x: int = 20,
                    skip_edge_attr: int = 64, safe_domi: float = 1e-10):
    """Apply ProtSolM normalization transform to graph."""
    dic = torch.load(norm_file, map_location="cpu", weights_only=False)
    x_mean = dic['x_mean']
    x_std = dic['x_std']
    pos_std = torch.mean(dic['pos_std'])

    edge_attr_mean = dic['edge_attr_mean']
    edge_attr_std = dic['edge_attr_std']

    graph.x[:, skip_x:] = (
        graph.x[:, skip_x:] - x_mean[skip_x:]
    ).div_(x_std[skip_x:] + safe_domi)

    graph.pos = graph.pos - graph.pos.mean(dim=-2, keepdim=False)
    graph.pos = graph.pos.div_(pos_std + safe_domi)

    # Normalize edge attr beyond skip
    if graph.edge_attr.shape[1] > skip_edge_attr:
        ea = graph.edge_attr[:, skip_edge_attr:]
        ea_mean = edge_attr_mean[skip_edge_attr:]
        ea_std = edge_attr_std[skip_edge_attr:]
        # Handle dimension mismatch gracefully
        min_dim = min(ea.shape[1], ea_mean.shape[0])
        graph.edge_attr[:, skip_edge_attr:skip_edge_attr + min_dim] = (
            ea[:, :min_dim] - ea_mean[:min_dim]
        ).div_(ea_std[:min_dim] + safe_domi)

    # Remove intermediate attributes if present
    if hasattr(graph, 'distances'):
        del graph.distances
    if hasattr(graph, 'edge_dist'):
        del graph.edge_dist
    if hasattr(graph, 'mu_r_norm'):
        del graph.mu_r_norm

    return graph


# ── Model Loading and Inference ──────────────────────────────────

def run_protsolm_inference(pdb_path: str, sequence: str | None = None) -> dict:
    """Run full ProtSolM inference on a single protein.

    Returns:
        dict with keys: soluble (bool), probability (float 0-1),
        confidence (str), label (str), raw_logits (list)
    """
    device = torch.device("cpu")  # CPU inference

    logger.info("Building protein graph from PDB...")
    graph = build_protein_graph(pdb_path, c_alpha_max_neighbors=20)

    logger.info("Normalizing graph...")
    graph = normalize_graph(graph, NORM_FILE)

    logger.info("Extracting handcrafted features...")
    features, pdb_seq = compute_features_from_pdb(pdb_path)

    # Build feature vector in ProtSolM order
    feature_aa_composition = ["1-C", "1-D", "1-E", "1-R", "1-H",
                              "Turn-forming residues fraction"]
    feature_gravy = ["GRAVY"]
    feature_ss_composition = [
        "ss8-G", "ss8-H", "ss8-I", "ss8-B", "ss8-E", "ss8-T",
        "ss8-S", "ss8-P", "ss8-L", "ss3-H", "ss3-E", "ss3-C"
    ]
    feature_hydrogen_bonds = ["Hydrogen bonds", "Hydrogen bonds per 100 residues"]
    feature_exposed = [f"Exposed residues fraction by {p}%"
                       for p in range(5, 105, 5)]
    feature_plddt = ["pLDDT"]

    feature_vec = []
    for key in (feature_aa_composition + feature_gravy +
                feature_ss_composition + feature_hydrogen_bonds +
                feature_exposed + feature_plddt):
        feature_vec.append(features.get(key, 0.0))

    feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).view(1, -1)
    feature_dim = len(feature_vec)

    logger.info(f"Feature dim: {feature_dim}")

    # Set up args namespace for model constructors
    class Args:
        pass

    args = Args()
    args.gnn = "egnn"
    args.gnn_config = yaml.load(
        open(GNN_CONFIG_FILE), Loader=yaml.FullLoader
    )["egnn"]
    args.gnn_config["hidden_channels"] = 512
    args.gnn_config["mlp_num"] = 2  # default from ProtSSN
    args.gnn_hidden_dim = 512
    args.plm = "facebook/esm2_t33_650M_UR50D"
    args.plm_hidden_size = 1280
    args.pooling_method = "attention1d"
    args.pooling_dropout = 0.1
    args.num_labels = 2
    args.feature_name = ["aa_composition", "gravy", "ss_composition",
                         "hygrogen_bonds", "exposed_res_fraction", "pLDDT"]
    args.feature_dim = feature_dim
    args.feature_embed_dim = 512
    args.use_plddt_penalty = True
    args.gnn_model_path = GNN_CHECKPOINT

    # Load ESM2 PLM
    logger.info("Loading ESM2 language model...")
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    esm_model = EsmModel.from_pretrained(args.plm).to(device)
    esm_model.eval()

    # Get ESM2 embeddings
    logger.info("Computing ESM2 embeddings...")
    use_seq = sequence if sequence else pdb_seq
    with torch.no_grad():
        inputs = tokenizer([use_seq], return_tensors="pt", padding=True).to(device)
        outputs = esm_model(**inputs)
        esm_rep = outputs.last_hidden_state[0, 1:1 + len(use_seq), :]  # strip BOS/EOS

    # Attach ESM rep to graph
    # Handle length mismatch between PDB residues and sequence
    graph_len = graph.x.shape[0]
    esm_len = esm_rep.shape[0]
    if graph_len != esm_len:
        logger.warning(f"Graph has {graph_len} nodes but ESM2 produced {esm_len} "
                       f"embeddings. Adjusting.")
        if esm_len > graph_len:
            esm_rep = esm_rep[:graph_len]
        else:
            pad = torch.zeros(graph_len - esm_len, esm_rep.shape[1])
            esm_rep = torch.cat([esm_rep, pad], dim=0)

    graph.esm_rep = esm_rep
    graph.feature = feature_tensor
    graph.label = torch.tensor([0]).view(1)
    graph.aa_seq = use_seq
    graph.name = os.path.basename(pdb_path).replace(".pdb", "")
    graph.batch = torch.zeros(graph_len, dtype=torch.long)

    # Load GNN
    logger.info("Loading ProtSSN GNN...")
    from src.models import GNN_model, ProtssnClassification
    from torch_geometric.data import Batch

    gnn_model = GNN_model(args).to(device)
    gnn_state = torch.load(GNN_CHECKPOINT, map_location=device, weights_only=False)
    gnn_model.load_state_dict(gnn_state)
    gnn_model.eval()

    # Load fine-tuned classifier
    logger.info("Loading ProtSolM classifier...")
    protssn_classification = ProtssnClassification(args).to(device)
    ft_state = torch.load(FT_CHECKPOINT, map_location=device, weights_only=False)
    protssn_classification.load_state_dict(ft_state["state_dict"])
    protssn_classification.eval()

    # Run GNN forward
    logger.info("Running GNN + classifier inference...")
    with torch.no_grad():
        # Build batch graph
        batch_graph = Batch.from_data_list([graph])

        # GNN forward
        gnn_out, gnn_embeds = gnn_model(batch_graph)

        # Combine ESM + GNN embeddings with pLDDT penalty
        esm_embeds = batch_graph.esm_rep
        if args.use_plddt_penalty:
            plddt_val = batch_graph.feature[:, -1]
            num_repeats = torch.bincount(batch_graph.batch)
            plddt_scaled = plddt_val.repeat_interleave(num_repeats).view(-1, 1)
            combined = esm_embeds + plddt_scaled * gnn_embeds
        else:
            combined = esm_embeds + gnn_embeds

        # Pad for batched attention pooling
        graph_sizes = torch.unique(batch_graph.batch, return_counts=True)[1]
        max_nodes = graph_sizes.max().item()
        batch_size = 1
        padded = torch.zeros(batch_size, max_nodes, combined.shape[-1])
        attention_mask = torch.zeros(batch_size, max_nodes)
        padded[0, :graph_len] = combined
        attention_mask[0, :graph_len] = 1

        # Pooling + classification
        pooled = protssn_classification.pooling(padded, attention_mask)

        # Add features
        feat = batch_graph.feature
        feat = protssn_classification.batch_norm1(feat)
        feat = protssn_classification.feature_embed_layer(feat)
        feat = protssn_classification.batch_norm2(feat)
        pooled_with_feat = torch.cat([pooled, feat], dim=1)

        logits, ssn_embed = protssn_classification.projection(
            pooled_with_feat, return_embed=True
        )

    # Interpret results
    probs = F.softmax(logits, dim=-1).squeeze()
    pred_label = torch.argmax(probs).item()
    soluble = pred_label == 1
    prob_soluble = probs[1].item() if probs.dim() > 0 else probs.item()

    # Confidence
    max_prob = max(probs.tolist()) if probs.dim() > 0 else probs.item()
    if max_prob >= 0.85:
        confidence = "high"
    elif max_prob >= 0.65:
        confidence = "moderate"
    else:
        confidence = "low"

    result = {
        "soluble": soluble,
        "probability": round(prob_soluble, 4),
        "probability_insoluble": round(1 - prob_soluble, 4),
        "confidence": confidence,
        "label": "Soluble" if soluble else "Insoluble",
        "score_pct": round(prob_soluble * 100, 1),
        "raw_logits": [round(l, 4) for l in logits.squeeze().tolist()],
        "features_used": list(features.keys()),
        "sequence_length": len(use_seq),
        "model": "ProtSolM (Tan et al., IEEE BIBM 2024)",
        "method": "ESM2 + ProtSSN EGNN + feature fusion",
    }

    logger.info(f"Prediction: {result['label']} "
                f"(P(soluble)={prob_soluble:.3f}, {confidence} confidence)")

    return result


def main():
    parser = argparse.ArgumentParser(description="ProtSolM inference helper")
    parser.add_argument("pdb_path", help="Input PDB file")
    parser.add_argument("output_json", help="Output JSON file")
    parser.add_argument("--sequence", default=None, help="Override sequence")
    args = parser.parse_args()

    if not os.path.exists(args.pdb_path):
        logger.error(f"PDB file not found: {args.pdb_path}")
        sys.exit(1)

    result = run_protsolm_inference(args.pdb_path, args.sequence)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result written to {args.output_json}")


if __name__ == "__main__":
    main()
