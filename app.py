import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Neural Network Learning Lab", page_icon="🧠", layout="wide")

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
header[data-testid="stHeader"] { background: rgba(11,16,32,0.9); }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc; }
[data-testid="stSidebar"] { background: #111a2e; border-right: 1px solid #26324b; }
[data-testid="stMetric"] { background: #111a2e; padding: 16px 18px; border: 1px solid #26324b; border-radius: 16px; }
div[data-testid="stTabs"] button { font-weight: 600; }
.lab-card { background: linear-gradient(135deg,#151f37,#10182b); border: 1px solid #293752; padding: 20px; border-radius: 18px; }
.kicker { color:#8ba8ff; text-transform:uppercase; letter-spacing: .12em; font-size: .75rem; font-weight:700; }
.subtle { color:#a7b3cc; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#1c2d4d; color:#a9c3ff; font-size:.8rem; margin-right:5px; }
div.stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- Math ----------
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def activation(name, z):
    if name == "Sigmoid":
        return sigmoid(z)
    if name == "Tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0, z)
    if name == "Leaky ReLU":
        return np.where(z > 0, z, 0.01*z)
    if name == "Linear":
        return z
    if name == "Softmax":
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(np.clip(z, -60, 60))
        return e / np.sum(e, axis=-1, keepdims=True)
    return z

def derivative(name, z):
    if name == "Sigmoid":
        s = sigmoid(z); return s * (1-s)
    if name == "Tanh":
        t = np.tanh(z); return 1-t*t
    if name == "ReLU":
        return (z > 0).astype(float)
    if name == "Leaky ReLU":
        return np.where(z > 0, 1.0, 0.01)
    if name == "Linear":
        return np.ones_like(z)
    return np.ones_like(z)

def make_curve(name, x):
    if name == "Softmax":
        # Softmax is vector-valued; illustrate a 3-class score vector with one varying logit.
        scores = np.stack([x, np.zeros_like(x), -x], axis=1)
        return activation("Softmax", scores)[:, 0]
    return activation(name, x)

def fmt(v):
    return f"{float(v):.4f}"

# ---------- Header ----------
st.markdown('<div class="kicker">INTERACTIVE AI EDUCATION · MODULE 04</div>', unsafe_allow_html=True)
st.title("Neural Network Learning Lab")
st.markdown('<p class="subtle">Explore what neurons compute, how activation functions shape signals, and how backpropagation updates a network.</p>', unsafe_allow_html=True)
st.markdown('<span class="pill">Activation functions</span><span class="pill">Forward propagation</span><span class="pill">Backpropagation</span><span class="pill">Training dynamics</span>', unsafe_allow_html=True)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("## 🧠 Lab controls")
    st.caption("Adjust the network and experiment live.")
    activation_name = st.selectbox("Hidden-layer activation", ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Linear"])
    output_name = st.selectbox("Output activation", ["Sigmoid", "Softmax"], index=0)
    learning_rate = st.select_slider("Learning rate", options=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2], value=0.05, format_func=lambda v: f"{v:g}")
    epochs = st.slider("Training epochs", 1, 100, 25)
    hidden_units = st.slider("Hidden neurons (concept view)", 2, 6, 4)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
    st.divider()
    st.caption("The training demo uses a tiny NumPy neural network so the calculations remain inspectable.")

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Activation Explorer", "🕸️ Network & Forward Pass", "↩️ Backpropagation", "📈 Training Lab"])

# ---------- Activation Explorer ----------
with tab1:
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown("### See the curve change")
        chosen = st.selectbox("Activation function", ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Linear", "Softmax"], key="curve_activation")
        xs = np.linspace(-6, 6, 400)
        ys = make_curve(chosen, xs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=chosen,
                                 line=dict(color="#7c9cff", width=4),
                                 fill="tozeroy", fillcolor="rgba(124,156,255,0.10)"))
        fig.add_vline(x=0, line_dash="dot", line_color="#53627e")
        fig.add_hline(y=0, line_dash="dot", line_color="#53627e")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.65)",
                          margin=dict(l=10,r=10,t=15,b=10), height=390,
                          xaxis_title="Input z", yaxis_title="Output f(z)",
                          font=dict(family="DM Sans", color="#dbe5ff"),
                          legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Function playground")
        z_value = st.slider("Neuron input z", -6.0, 6.0, 1.5, 0.1, key="z_value")
        if chosen == "Softmax":
            scores = np.array([[z_value, 0.0, -1.0]])
            out = activation("Softmax", scores)[0]
            st.metric("Class 1 probability", fmt(out[0]))
            st.write(pd.DataFrame({"Class": ["Class 1", "Class 2", "Class 3"], "Probability": out}).style.format({"Probability":"{:.4f}"}))
            formula = "softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)"
            description = "Softmax acts on a vector of class scores, turning them into probabilities that sum to 1."
        else:
            out = float(activation(chosen, np.array(z_value)))
            der = float(derivative(chosen, np.array(z_value)))
            st.metric("Activation output", fmt(out))
            st.metric("Derivative at z", fmt(der))
            formula = {
                "ReLU":"f(z) = max(0, z)",
                "Sigmoid":"f(z) = 1 / (1 + e⁻ᶻ)",
                "Tanh":"f(z) = tanh(z)",
                "Leaky ReLU":"f(z) = max(0.01z, z)",
                "Linear":"f(z) = z"
            }[chosen]
            description = {
                "ReLU":"Common in hidden layers. Negative inputs become zero; positive inputs pass through.",
                "Sigmoid":"Maps values to (0, 1). Often used for binary classification outputs.",
                "Tanh":"Maps values to (-1, 1), centered around zero.",
                "Leaky ReLU":"Keeps a small gradient for negative inputs.",
                "Linear":"Does not add non-linearity; commonly used for regression outputs."
            }[chosen]
        st.markdown(f'<div class="lab-card"><div class="kicker">FORMULA</div><h3>{formula}</h3><p class="subtle">{description}</p></div>', unsafe_allow_html=True)
        st.info("Backpropagation uses derivatives to determine how changing a neuron's input or weights changes the loss.")

# ---------- Network visualization ----------
with tab2:
    st.markdown("### Follow one input through the network")
    st.caption("Illustrative network layout; the numeric forward-pass example below uses a small, explicit network.")
    rng = np.random.default_rng(int(seed))
    layer_sizes = [3, hidden_units, 2]
    positions = []
    for li, size in enumerate(layer_sizes):
        ys_pos = np.linspace(-1, 1, size) if size > 1 else np.array([0.0])
        for ni, yy in enumerate(ys_pos):
            positions.append((li, ni, li*2.4, yy))
    edges_x, edges_y = [], []
    for li in range(len(layer_sizes)-1):
        left_nodes = [p for p in positions if p[0] == li]
        right_nodes = [p for p in positions if p[0] == li+1]
        for a in left_nodes:
            for b in right_nodes:
                edges_x += [a[2], b[2], None]
                edges_y += [a[3], b[3], None]
    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines", line=dict(color="rgba(124,156,255,.25)", width=1.5), hoverinfo="skip", showlegend=False))
    labels = []
    for li, ni, xx, yy in positions:
        labels.append(["Input", "Hidden", "Output"][li] + (f" {ni+1}" if li != 2 else f" {ni+1}"))
    colors = ["#55d6be" if p[0] == 0 else "#7c9cff" if p[0] == 1 else "#ffb86b" for p in positions]
    fig_net.add_trace(go.Scatter(x=[p[2] for p in positions], y=[p[3] for p in positions], mode="markers+text",
        text=[str(p[1]+1) for p in positions], textposition="middle center",
        marker=dict(size=35, color=colors, line=dict(color="#e7edff", width=1.2)),
        hovertext=labels, hoverinfo="text", showlegend=False))
    for xx, label in [(0,"INPUT LAYER"),(2.4,"HIDDEN LAYER"),(4.8,"OUTPUT LAYER")]:
        fig_net.add_annotation(x=xx, y=1.35, text=label, showarrow=False, font=dict(size=11,color="#a7b3cc"))
    fig_net.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.5)",
        height=360, margin=dict(l=15,r=15,t=35,b=15), xaxis=dict(visible=False,range=[-0.8,5.6]),
        yaxis=dict(visible=False,range=[-1.45,1.55]), showlegend=False)
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("#### Numeric forward pass")
    c1,c2,c3 = st.columns(3)
    x1 = c1.number_input("Input x₁", value=0.8, step=0.1)
    x2 = c2.number_input("Input x₂", value=-0.4, step=0.1)
    bias = c3.number_input("Bias b", value=0.2, step=0.1)
    w1,w2 = 0.7,-0.5
    z = w1*x1 + w2*x2 + bias
    a = float(activation(activation_name, np.array(z)))
    st.latex(r"z = w_1x_1 + w_2x_2 + b")
    cols = st.columns(4)
    cols[0].metric("w₁x₁", fmt(w1*x1))
    cols[1].metric("w₂x₂", fmt(w2*x2))
    cols[2].metric("Weighted sum z", fmt(z))
    cols[3].metric(f"Activation a ({activation_name})", fmt(a))
    st.success(f"The neuron calculates z = {w1:.1f}×{x1:.1f} + ({w2:.1f})×({x2:.1f}) + {bias:.1f} = {z:.4f}, then applies {activation_name} to produce {a:.4f}.")

# ---------- Backpropagation ----------
with tab3:
    st.markdown("### Backpropagation, one step at a time")
    st.write("This worked example uses one sigmoid neuron with binary cross-entropy loss. The gradients are calculated analytically.")
    cc1,cc2,cc3,cc4 = st.columns(4)
    bx1 = cc1.number_input("Input x", value=1.0, step=0.1, key="bp_x")
    bw = cc2.number_input("Weight w", value=0.8, step=0.1, key="bp_w")
    bb = cc3.number_input("Bias b", value=-0.2, step=0.1, key="bp_b")
    by = cc4.selectbox("True label y", [0,1], index=1, key="bp_y")
    bp_z = bw*bx1 + bb
    bp_pred = float(sigmoid(np.array(bp_z)))
    eps = 1e-12
    bp_loss = -(by*np.log(bp_pred+eps) + (1-by)*np.log(1-bp_pred+eps))
    # For sigmoid + BCE, dL/dz = prediction - label
    dL_dz = bp_pred - by
    dL_dw = dL_dz * bx1
    dL_db = dL_dz
    updated_w = bw - learning_rate*dL_dw
    updated_b = bb - learning_rate*dL_db
    step_names = ["1 · Forward", "2 · Loss", "3 · Gradient", "4 · Update"]
    step = st.radio("Reveal calculation", step_names, horizontal=True, key="bp_step")
    st.markdown('<div class="lab-card">', unsafe_allow_html=True)
    if step == step_names[0]:
        st.latex(r"z = wx+b")
        st.latex(f"z = ({bw:.3f})({bx1:.3f}) + ({bb:.3f}) = {bp_z:.4f}")
        st.latex(r"\\hat{y}=\\sigma(z)")
        st.latex(f"\\hat{{y}} = {bp_pred:.4f}")
        st.write("The neuron combines its input, weight and bias, then applies sigmoid to produce a probability.")
    elif step == step_names[1]:
        st.latex(r"L=-[y\\log(\\hat{y})+(1-y)\\log(1-\\hat{y})]")
        st.latex(f"L = {bp_loss:.4f}")
        st.write("Binary cross-entropy penalizes predictions that assign low probability to the correct label.")
    elif step == step_names[2]:
        st.latex(r"\\frac{\\partial L}{\\partial z}=\\hat{y}-y")
        st.latex(f"\\frac{{\\partial L}}{{\\partial z}} = {bp_pred:.4f} - {by} = {dL_dz:.4f}")
        st.latex(r"\\frac{\\partial L}{\\partial w}=(\\hat{y}-y)x")
        st.latex(f"\\frac{{\\partial L}}{{\\partial w}} = {dL_dw:.4f}")
        st.latex(r"\\frac{\\partial L}{\\partial b}=\\hat{y}-y")
        st.latex(f"\\frac{{\\partial L}}{{\\partial b}} = {dL_db:.4f}")
        st.write("The gradient tells us how the loss changes when the weight or bias changes.")
    else:
        st.latex(r"w_{new}=w-\\eta\\frac{\\partial L}{\\partial w}")
        st.latex(f"w_{{new}} = {bw:.4f} - {learning_rate:g}({dL_dw:.4f}) = {updated_w:.4f}")
        st.latex(r"b_{new}=b-\\eta\\frac{\\partial L}{\\partial b}")
        st.latex(f"b_{{new}} = {bb:.4f} - {learning_rate:g}({dL_db:.4f}) = {updated_b:.4f}")
        st.write("The optimizer moves parameters in the direction that should reduce loss. A new forward pass checks the result.")
    st.markdown('</div>', unsafe_allow_html=True)
    new_pred = float(sigmoid(np.array(updated_w*bx1+updated_b)))
    new_loss = -(by*np.log(new_pred+eps)+(1-by)*np.log(1-new_pred+eps))
    m1,m2,m3 = st.columns(3)
    m1.metric("Loss before update", fmt(bp_loss))
    m2.metric("Loss after one update", fmt(new_loss), delta=f"{new_loss-bp_loss:+.4f}", delta_color="inverse")
    m3.metric("Updated weight", fmt(updated_w))

# ---------- Training lab ----------
with tab4:
    st.markdown("### Train a tiny neural network")
    st.caption("A small 2–4–1 network learns a simple XOR dataset. Training uses NumPy and manual backpropagation.")
    if st.button("▶ Run training", type="primary", use_container_width=False):
        rng = np.random.default_rng(int(seed))
        X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        y = np.array([[0.],[1.],[1.],[0.]])
        W1 = rng.normal(0, 0.8, (2,4)); b1 = np.zeros((1,4))
        W2 = rng.normal(0, 0.8, (4,1)); b2 = np.zeros((1,1))
        history = []
        for ep in range(epochs):
            z1 = X @ W1 + b1
            a1 = activation(activation_name, z1)
            z2 = a1 @ W2 + b2
            # binary output uses sigmoid regardless of output selector
            pred = sigmoid(z2)
            loss = -np.mean(y*np.log(pred+1e-9)+(1-y)*np.log(1-pred+1e-9))
            history.append(loss)
            dz2 = (pred-y)/len(X)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ W2.T
            dz1 = da1 * derivative(activation_name, z1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            W2 -= learning_rate*dW2; b2 -= learning_rate*db2
            W1 -= learning_rate*dW1; b1 -= learning_rate*db1
        z1 = X @ W1 + b1
        a1 = activation(activation_name, z1)
        final_pred = sigmoid(a1 @ W2 + b2)
        final_loss = history[-1] if history else float("nan")
        st.session_state["training_result"] = (history, final_pred, X, y, activation_name, learning_rate, epochs)
    if "training_result" in st.session_state:
        history, final_pred, X, y, used_activation, used_lr, used_epochs = st.session_state["training_result"]
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(range(1,len(history)+1)), y=history, mode="lines+markers",
            line=dict(color="#55d6be", width=3), marker=dict(size=5), name="Binary cross-entropy"))
        fig_loss.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.65)",
            height=320, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="Epoch", yaxis_title="Loss",
            font=dict(family="DM Sans", color="#dbe5ff"))
        st.plotly_chart(fig_loss, use_container_width=True)
        results = pd.DataFrame({"x₁":X[:,0].astype(int), "x₂":X[:,1].astype(int), "True label":y[:,0].astype(int),
                                "Predicted probability":final_pred[:,0], "Predicted class":(final_pred[:,0]>=0.5).astype(int)})
        st.dataframe(results.style.format({"Predicted probability":"{:.4f}"}), use_container_width=True, hide_index=True)
        st.caption(f"Run settings: activation = {used_activation}, learning rate = {used_lr:g}, epochs = {used_epochs}. XOR is a tiny teaching dataset; results vary with initialization and settings.")
    else:
        st.markdown('<div class="lab-card"><h3>Ready to train?</h3><p class="subtle">Press “Run training” to initialize a small network, calculate gradients manually, update weights across epochs, and inspect the loss curve and predictions.</p></div>', unsafe_allow_html=True)

st.divider()
st.markdown('<p class="subtle">Neural Network Learning Lab · Built with Streamlit, NumPy, Pandas and Plotly.</p>', unsafe_allow_html=True)
