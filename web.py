import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from joblib import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GraphConv,
    NNConv,
    global_mean_pool,
    AttentiveFP
)
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, MaskType


# --- Unified Styling ---
st.markdown("""
<style>
/* ===== PAGE BACKGROUND ===== */
.stApp {
background-color: #eef6fa !important;
font-family: 'sans-serif' !important;
}
            
/* ===== GLOBAL TEXT ===== */
p, span, div, label, h1, h2, h3, h4, h5 {
    color: #002244 !important;
}
            
/* ===== AUTHOR SECTION ===== */
.author {
    background-color: #cce0ff !important;
    color: #003366 !important;
    font-style: italic;
    font-size: 16px;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    margin-top: 30px;
}

/* ===== TABS ===== */
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button span,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button div {
    color: #002244 !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    opacity: 1 !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stTabs"] div[data-baseweb="tab-panel"] * {
    color: #002244 !important;
}

/* ===== INPUT BOXES ===== */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px solid #4da6ff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #666666 !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px dashed #4da6ff !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}

/* ===== ALL BUTTONS ===== */
div.stButton > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
    }
/* Hover/focus state */ 
div.stButton > button:hover, 
div.stButton > button:focus { 
    background-color: #f0f0f0 !important; /* 
    Slight gray on hover */ 
    color: #000000 !important; 
    }
            
   
</style>
""", unsafe_allow_html=True)

# =========================
# Page config
# =========================
st.set_page_config(page_title="MAPK1 Inhibitor Screening", layout="centered")
st.title("🔬 MAPK1 Inhibitor Screening")

# --- Introduction Section ---
st.info("""
This platform predicts **mitogen-activated protein kinase 1 (MAPK1)** inhibitors using graph-based molecular features within a **stacking ensemble framework**.
""")

# --- Model Architecture Section ---
st.markdown("#### Model architectures and features")
st.markdown("""
* **Baseline models:** Attentive fingerprints (AttentiveFP), graph convolutional neural network (GCNN), and graph neural network (GNN).
* **Stacked model:** Logistic regression (LR).
* **Features:** Molecular graphs.
""")

# --- Prediction Input (Placeholder) ---
st.markdown("#### Run Prediction")
# =========================
# Load models & scalers
# =========================
class GCNNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GraphConv(node_dim, hidden_dim)] +
                                   [GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)

class GNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(node_dim, hidden_dim, self.edge_net, aggr='mean')
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.nnconv(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)

class GNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, batch):
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, batch=batch)
        return self.model(data)


@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_dim = 8
    edge_dim = 4

    gcnn = GCNNClassifier(node_dim).to(device)
    gcnn.load_state_dict(torch.load("gcnn_model.pth", map_location=device))
    gcnn.eval()

    gnn = GNNClassifier(node_dim, edge_dim).to(device)
    gnn.load_state_dict(torch.load("gnn_model.pth", map_location=device))
    gnn.eval()

    attentivefp = AttentiveFP(
        in_channels=node_dim,
        hidden_channels=64,
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=3,
        num_timesteps=2
    ).to(device)
    attentivefp.load_state_dict(torch.load("attentivefp_model.pth", map_location=device))
    attentivefp.eval()

    # Meta model (logistic regression)
    meta = load("StackGNN_LR_model.pkl")

    return {
        "device": device,
        "gcnn": gcnn,
        "gnn": gnn,
        "attfp": attentivefp,
        "meta": meta
    }

res = load_resources()


# =========================
# Graph encoder (RDKit → PyTorch Geometric)
# =========================
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = bond_features(bond)

        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(x.size(0), dtype=torch.long)
    )

    return data

# =========================
# Full inference pipeline
# =========================
def predict_full_system(smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    device = res["device"]
    graph = graph.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        p1 = torch.sigmoid(res["gcnn"](graph)).item()
        p2 = torch.sigmoid(res["gnn"](graph)).item()
        p3 = torch.sigmoid(
            res["attfp"](graph.x, graph.edge_index, graph.edge_attr, batch)
        ).item()

    stack = np.array([[p1, p2, p3]])
    final = res["meta"].predict_proba(stack)[0, 1]

    # Determine consensus
    baseline_probs = [p1, p2, p3]
    all_probs = baseline_probs + [final]
    
    if all(p > 0.5 for p in all_probs):
        consensus = "Active"
    elif all(p < 0.5 for p in all_probs):
        consensus = "Inactive"
    else:
        consensus = "Inconclusive"

    return final, stack, graph, consensus

def explain_graph(model, graph_data, device, threshold=0.7):
    model.eval()
    wrapped = GNNWrapper(model).to(device)

    graph_data = graph_data.to(device)
    batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=200),
        explanation_type=ExplanationType.model,
        model_config=dict(
            mode="binary_classification",
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object,
    )

    explanation = explainer(
        graph_data.x,
        graph_data.edge_index,
        edge_attr=graph_data.edge_attr,
        batch=batch
    )

    edge_mask = explanation.edge_mask
    edge_mask_np = edge_mask.sigmoid().cpu().numpy()
    edge_index = graph_data.edge_index.cpu().numpy()
    important_edges = np.where(edge_mask_np > threshold)[0]

    return important_edges, edge_index


def plot_molecule_with_highlights(smiles, important_edges, edge_index, pred_class):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Could not parse SMILES for plotting.")
        return None

    important_atoms = set()
    important_bonds = set()

    for idx in important_edges:
        a1, a2 = int(edge_index[0, idx]), int(edge_index[1, idx])
        important_atoms.add(a1)
        important_atoms.add(a2)
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            important_bonds.add(bond.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(important_atoms),
        highlightBonds=list(important_bonds)
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    st.markdown(f"### Predicted class: {'Active' if pred_class else 'Inactive'}")
    st.image(svg, use_container_width=True)

    return svg


# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["Single SMILES", "CSV Batch"])

with tab1:
    smiles_input = st.text_input("🔹 Enter SMILES:")

    if st.button("Predict SMILES"):
        result = predict_full_system(smiles_input)

        if result is None:
            st.error("Invalid SMILES")
        else:
            st.session_state["last_result"] = result
            st.session_state["last_smiles"] = smiles_input

    if "last_result" in st.session_state:
        prob, stack, graph, consensus = st.session_state["last_result"]

        st.metric("Consensus Prediction", consensus)

        st.write("Base model outputs:")
        st.write(f"GCNN: {stack[0][0]:.4f}")
        st.write(f"GNN: {stack[0][1]:.4f}")
        st.write(f"AttentiveFP: {stack[0][2]:.4f}")
        st.write(f"Stack Model: {prob:.4f}")

        if st.button("Explain prediction"):
            important_edges, edge_index = explain_graph(
                res["gnn"],
                graph.cpu(),
                res["device"]
            )

            pred_class = consensus == "Active"
            svg = plot_molecule_with_highlights(
                st.session_state["last_smiles"],
                important_edges,
                edge_index,
                pred_class
            )

            if svg:
                st.download_button(
                    label="Download structure (SVG)",
                    data=svg,
                    file_name="molecule_structure.svg",
                    mime="image/svg+xml", type="secondary"
                )


with tab2:
    uploaded_file = st.file_uploader("📂 Upload CSV with SMILES column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        final_preds = []
        gcnn_preds = []
        gnn_preds = []
        attfp_preds = []

        for smiles in df["SMILES"]:
            result = predict_full_system(smiles)
            if result:
                final, stack, _ = result
                final_preds.append(final)
                gcnn_preds.append(stack[0][0])
                gnn_preds.append(stack[0][1])
                attfp_preds.append(stack[0][2])
            else:
                final_preds.append(None)
                gcnn_preds.append(None)
                gnn_preds.append(None)
                attfp_preds.append(None)

        df["AttentiveFP_Prob"] = attfp_preds
        df["GCNN_Prob"] = gcnn_preds
        df["GNN_Prob"] = gnn_preds
        df["Stack_Model_Probability"] = final_preds

        # Compute consensus
        consensuses = []
        for i in range(len(final_preds)):
            if final_preds[i] is None:
                consensuses.append("Invalid")
            else:
                baseline = [gcnn_preds[i], gnn_preds[i], attfp_preds[i]]
                all_p = baseline + [final_preds[i]]
                if all(p > 0.5 for p in all_p):
                    consensuses.append("Active")
                elif all(p < 0.5 for p in all_p):
                    consensuses.append("Inactive")
                else:
                    consensuses.append("Inconclusive")
        
        df["Consensus_Label"] = consensuses

        st.dataframe(df.style.set_properties(
            background_color="white",
            color="black"
        ))

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv", type="secondary"
        )

        st.markdown(
            "**Prediction Interpretation:**  \n"
            "Consensus based on agreement:  \n"
            "All models agree >0.5 → Active  \n"
            "All models agree <0.5 → Inactive  \n"
            "Any disagreement → Inconclusive"
        )



# =========================
# Footer
# =========================
# --- Spacer before author section ---
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --- Author Section ---
st.markdown("""
<div class="author">
Authors\n
Tarapong Srisongkram<sup>1*</sup>, Darlene Nabila Zetta<sup>2</sup>, Sastiya Kampaengsri<sup>1</sup>, and Natthida Weerapreeyakul<sup>1</sup>

<sup>1</sup>*Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*
            
<sup>2</sup>*Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*
</div>
""", unsafe_allow_html=True)