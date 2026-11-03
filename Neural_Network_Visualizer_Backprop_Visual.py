import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Neural Network Learning Lab", page_icon="🧠", layout="wide")

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
header[data-testid="stHeader"] { background: rgba(11,16,32,0.9); }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
h1,h2,h3,.stApp h1,.stApp h2,.stApp h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc !important; }
p, label, li, .stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] { color: #e5eaf5; }
.stSelectbox [data-baseweb="select"] *, .stMultiSelect [data-baseweb="select"] * { color: #172033 !important; }
.stSelectbox [data-baseweb="select"] { background: #f8fafc !important; }
[data-testid="stSidebar"] { background: #111a2e; border-right: 1px solid #26324b; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #e5eaf5 !important; }
.stSelectbox div, .stSlider div, .stNumberInput input { color: #f8fafc; }
[data-testid="stMetric"] { background: #111a2e; padding: 14px 16px; border: 1px solid #26324b; border-radius: 14px; }
.lab-card { background: linear-gradient(135deg,#151f37,#10182b); border: 1px solid #293752; padding: 18px; border-radius: 16px; margin: 8px 0; }
.kicker { color:#8ba8ff; text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:700; }
.subtle { color:#a7b3cc; }
</style>
""", unsafe_allow_html=True)

# ---------- Math helpers ----------
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def activation(name, z):
    if name == "Sigmoid":
        return sigmoid(z)
    if name == "Tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0.0, z)
    if name == "Leaky ReLU":
        return np.where(z > 0, z, 0.01 * z)
    if name == "Linear":
        return z
    return sigmoid(z)

def activation_derivative(name, z):
    if name == "Sigmoid":
        a = sigmoid(z)
        return a * (1 - a)
    if name == "Tanh":
        a = np.tanh(z)
        return 1 - a * a
    if name == "ReLU":
        return (z > 0).astype(float)
    if name == "Leaky ReLU":
        return np.where(z > 0, 1.0, 0.01)
    if name == "Linear":
        return np.ones_like(z)
    return sigmoid(z) * (1 - sigmoid(z))

def fmt(value):
    return f"{float(value):.5f}"

def activation_formula(name):
    return {
        "ReLU": "f(z) = max(0, z)",
        "Sigmoid": "f(z) = 1 / (1 + e⁻ᶻ)",
        "Tanh": "f(z) = tanh(z)",
        "Leaky ReLU": "f(z) = max(z, 0.01z)",
        "Linear": "f(z) = z",
    }[name]

def activation_description(name):
    return {
        "ReLU": "Negative inputs become 0; positive inputs pass through. Its gradient is 0 for negative z and 1 for positive z.",
        "Sigmoid": "Maps a scalar to (0, 1). It can represent a probability, but gradients can become small for very large positive or negative inputs.",
        "Tanh": "Maps inputs to (-1, 1) and is centered at zero. Its gradient becomes small when the output saturates near -1 or 1.",
        "Leaky ReLU": "Passes positive inputs unchanged and keeps a small 0.01 slope for negative inputs, so the negative-side gradient is not zero.",
        "Linear": "Returns the input unchanged. It adds no non-linearity, so stacking only linear layers is still equivalent to a linear transformation.",
    }[name]

# ---------- Header ----------
st.markdown('<div class="kicker">INTERACTIVE AI EDUCATION · MODULE 04</div>', unsafe_allow_html=True)
st.title("Neural Network Learning Lab")
st.markdown('<p class="subtle">One consistent example from input to output: inspect every weighted sum, activation, loss, gradient, and parameter update.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 Network controls")
    st.caption("The same two-input, one-output example is used throughout the lab.")
    hidden_layers = st.slider("Number of hidden layers", min_value=1, max_value=3, value=2)
    hidden_units = st.slider("Neurons in each hidden layer", min_value=1, max_value=3, value=2)
    hidden_activation = st.selectbox("Hidden-layer activation", ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Linear"])
    learning_rate = st.slider("Learning rate η", min_value=0.001, max_value=0.200, value=0.050, step=0.001, format="%.3f")
    st.caption("The output is one sigmoid neuron for binary classification. Hidden-layer activation can be changed.")
    st.divider()
    st.markdown("### Example data")
    x1 = st.slider("Input feature x₁", -2.0, 2.0, 0.8, 0.1)
    x2 = st.slider("Input feature x₂", -2.0, 2.0, -0.4, 0.1)
    true_label = st.selectbox("True label y", [0, 1], index=1)
    st.caption("Change x₁, x₂, or y and all calculations below update together.")

tabs = st.tabs(["⚡ Activation Explorer", "🕸️ Forward Pass · All Layers", "↩️ Backpropagation · Full Update"])

# ---------- Activation Explorer ----------
with tabs[0]:
    st.markdown("### Activation function explorer")
    st.write("Move the input slider or switch functions. The curve, highlighted coordinate, output, derivative, and explanation all update together.")

    c1, c2 = st.columns([1.3, 0.9], gap="large")
    with c2:
        chosen_activation = st.selectbox(
            "Activation function",
            ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Linear"],
            key="explore_activation"
        )
        z_value = st.slider("Neuron input z", -6.0, 6.0, 1.5, 0.1, key="explore_z")
        current_output = float(activation(chosen_activation, np.array(z_value)))
        current_derivative = float(activation_derivative(chosen_activation, np.array(z_value)))

        st.metric("Input to activation (z)", f"{z_value:.2f}")
        st.metric("Output after activation f(z)", fmt(current_output))
        st.metric("Local slope f′(z)", fmt(current_derivative))

        # Interpret the derivative in plain language, including special cases.
        if chosen_activation == "ReLU":
            slope_meaning = (
                "The local slope is 0, so this ReLU unit passes no gradient through this point."
                if z_value < 0 else
                "The local slope is 1, so the gradient passes through this activation unchanged."
                if z_value > 0 else
                "At z = 0, ReLU has a corner and its mathematical derivative is undefined. This demo uses 0 as the conventional derivative."
            )
            region_meaning = "Negative z → output 0." if z_value < 0 else "Positive z → output equals z." if z_value > 0 else "At z = 0, the output is 0 and the curve changes direction."
        elif chosen_activation == "Leaky ReLU":
            slope_meaning = "The local slope is 0.01, so a small gradient can still pass through the negative side." if z_value < 0 else "The local slope is 1, so the positive input and gradient pass through unchanged."
            region_meaning = "Negative z → output is 0.01 × z." if z_value < 0 else "Non-negative z → output equals z."
        elif chosen_activation == "Sigmoid":
            slope_meaning = "The derivative is largest around z = 0 and becomes small when the output is close to 0 or 1. Very small gradients can slow learning."
            region_meaning = "Sigmoid converts any real input into a value strictly between 0 and 1, often interpreted as a probability for a binary output."
        elif chosen_activation == "Tanh":
            slope_meaning = "The derivative is strongest near z = 0 and approaches 0 for very large positive or negative inputs, where the curve saturates."
            region_meaning = "Tanh maps the input into the range −1 to 1, with output 0 when z = 0."
        else:
            slope_meaning = "The derivative is always 1, so this activation neither shrinks nor expands the gradient."
            region_meaning = "Linear returns the input unchanged. It does not add non-linearity to a network."

        st.markdown(
            f'<div class="lab-card"><div class="kicker">FUNCTION AND RULE</div>'
            f'<h3>{activation_formula(chosen_activation)}</h3>'
            f'<p>{activation_description(chosen_activation)}</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="lab-card"><div class="kicker">READ YOUR CURRENT POINT</div>'
            f'<p><b>1. Input:</b> z = {z_value:.2f} is the value calculated by the neuron before activation.</p>'
            f'<p><b>2. Apply the function:</b> f({z_value:.2f}) = <b>{current_output:.5f}</b>.</p>'
            f'<p><b>3. Read the slope:</b> f′({z_value:.2f}) = <b>{current_derivative:.5f}</b>. {slope_meaning}</p>'
            f'<p><b>4. What this function is doing:</b> {region_meaning}</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="lab-card"><div class="kicker">WHY THIS MATTERS IN LEARNING</div>'
            '<p>During the backward pass, the gradient arriving from later layers is multiplied by this local derivative. '
            'A derivative near 0 can greatly reduce the gradient; a derivative near 1 lets it pass through at about the same size. '
            'The derivative is a local slope, not the loss itself.</p></div>',
            unsafe_allow_html=True
        )

    with c1:
        xs = np.linspace(-6, 6, 500)
        ys = activation(chosen_activation, xs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{chosen_activation} curve",
            line=dict(color="#7c9cff", width=3),
            hovertemplate="z = %{x:.2f}<br>f(z) = %{y:.5f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[z_value], y=[current_output], mode="markers+text",
            text=[f"({z_value:.2f}, {current_output:.2f})"],
            textposition="top center", name="Current value",
            marker=dict(color="#ffb86b", size=14, line=dict(color="#ffffff", width=2)),
            hovertemplate="Current input z = %{x:.2f}<br>Output f(z) = %{y:.5f}<extra></extra>"
        ))
        fig.add_vline(x=z_value, line_dash="dash", line_color="#ffb86b")
        fig.add_hline(y=current_output, line_dash="dot", line_color="#ffb86b")
        fig.add_vline(x=0, line_dash="dot", line_color="#53627e")
        fig.add_hline(y=0, line_dash="dot", line_color="#53627e")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,22,40,0.65)",
            margin=dict(l=10, r=10, t=25, b=10), height=460,
            xaxis_title="Input z", yaxis_title="Output f(z)",
            font=dict(family="DM Sans", color="#e5eaf5"),
            xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor="#71809a", ticks="outside"),
            yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor="#71809a", ticks="outside"),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#e5eaf5"))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"The orange marker is the live point ({z_value:.2f}, {current_output:.5f}). "
            "The dashed lines show its input and output coordinates. Hover over the curve to inspect other values."
        )

# ---------- Build one consistent network ----------
# Deterministic, explicit starting weights keep the example repeatable and easy to inspect.
# Each hidden layer has `hidden_units` neurons. Output layer has exactly one sigmoid neuron.
def make_weights(layer_sizes):
    weights, biases = [], []
    for layer_index in range(1, len(layer_sizes)):
        fan_in = layer_sizes[layer_index - 1]
        fan_out = layer_sizes[layer_index]
        W = np.zeros((fan_in, fan_out), dtype=float)
        for i in range(fan_in):
            for j in range(fan_out):
                W[i, j] = 0.25 * (i + 1) - 0.12 * (j + 1) + 0.08 * layer_index
        b = np.array([0.05 * (j + 1) - 0.03 * layer_index for j in range(fan_out)], dtype=float)
        weights.append(W)
        biases.append(b)
    return weights, biases

layer_sizes = [2] + [hidden_units] * hidden_layers + [1]
weights, biases = make_weights(layer_sizes)

# Optional direct controls for every parameter. Values persist across Streamlit reruns.
with st.sidebar.expander("Advanced: edit every weight and bias", expanded=False):
    st.caption("Each slider changes the exact parameter used in the forward and backward calculations.")
    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        is_output = layer_idx == hidden_layers
        label = "Output layer" if is_output else f"Hidden layer {layer_idx + 1}"
        st.markdown(f"**{label}**")
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = st.slider(
                    f"W{layer_idx+1}[{i+1},{j+1}]",
                    min_value=-2.0, max_value=2.0,
                    value=float(W[i, j]), step=0.05,
                    key=f"weight_{layer_idx}_{i}_{j}"
                )
        for j in range(len(b)):
            b[j] = st.slider(
                f"b{layer_idx+1}[{j+1}]",
                min_value=-2.0, max_value=2.0,
                value=float(b[j]), step=0.05,
                key=f"bias_{layer_idx}_{j}"
            )

x_vec = np.array([x1, x2], dtype=float)
activations = [x_vec]
zs = []
for layer_idx, (W, b) in enumerate(zip(weights, biases)):
    z = activations[-1] @ W + b
    a = activation(hidden_activation, z) if layer_idx < hidden_layers else sigmoid(z)
    zs.append(z)
    activations.append(a)

prediction = float(activations[-1][0])
eps = 1e-12
loss = float(-(true_label * np.log(prediction + eps) + (1 - true_label) * np.log(1 - prediction + eps)))

def layer_name(index, total_hidden):
    if index == 0:
        return "Input layer"
    if index <= total_hidden:
        return f"Hidden layer {index}"
    return "Output layer"

# ---------- Forward pass ----------
with tabs[1]:
    st.markdown("### Network view: follow the same example through every layer")
    architecture_text = " → ".join(map(str, layer_sizes))
    st.markdown(
        f'<div class="lab-card"><b>Inputs:</b> x₁ = {x1:.2f}, x₂ = {x2:.2f} '
        f'&nbsp; · &nbsp; <b>Target:</b> y = {true_label} '
        f'&nbsp; · &nbsp; <b>Network:</b> {architecture_text} '
        f'&nbsp; · &nbsp; <b>Hidden activation:</b> {hidden_activation} '
        f'&nbsp; · &nbsp; <b>Output activation:</b> Sigmoid</div>',
        unsafe_allow_html=True
    )
    st.write("The diagram shows the architecture. Below it, follow each neuron's exact calculation from the two inputs to the single output.")

    # Network diagram: input layer, 1–3 hidden layers (up to 3 neurons each), and one output.
    node_positions = []
    layer_x = np.linspace(0.0, 1.0, len(layer_sizes))
    for li, size in enumerate(layer_sizes):
        y_positions = np.linspace(-1.0, 1.0, size) if size > 1 else np.array([0.0])
        for ni, yy in enumerate(y_positions):
            node_positions.append({"layer": li, "neuron": ni, "x": float(layer_x[li]), "y": float(yy)})
    edge_x, edge_y = [], []
    for li in range(len(layer_sizes) - 1):
        left_nodes = [n for n in node_positions if n["layer"] == li]
        right_nodes = [n for n in node_positions if n["layer"] == li + 1]
        for left_node in left_nodes:
            for right_node in right_nodes:
                edge_x += [left_node["x"], right_node["x"], None]
                edge_y += [left_node["y"], right_node["y"], None]
    net_fig = go.Figure()
    net_fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(124,156,255,0.30)", width=1.5),
        hoverinfo="skip", showlegend=False
    ))
    node_colors, node_labels, node_hover = [], [], []
    for node in node_positions:
        li, ni = node["layer"], node["neuron"]
        if li == 0:
            node_colors.append("#55d6be")
            node_labels.append(f"x{ni+1}")
            node_hover.append(f"Input x{ni+1} = {x_vec[ni]:.5f}")
        elif li == len(layer_sizes) - 1:
            node_colors.append("#ffb86b")
            node_labels.append("ŷ")
            node_hover.append(f"Output probability ŷ = {prediction:.5f}")
        else:
            node_colors.append("#7c9cff")
            node_labels.append(f"H{li}-{ni+1}")
            node_hover.append(f"Hidden layer {li}, neuron {ni+1}<br>z = {zs[li-1][ni]:.5f}<br>a = {activations[li][ni]:.5f}")
    net_fig.add_trace(go.Scatter(
        x=[n["x"] for n in node_positions], y=[n["y"] for n in node_positions],
        mode="markers+text", text=node_labels, textposition="middle center",
        textfont=dict(color="#08111f", size=11),
        marker=dict(size=42, color=node_colors, line=dict(color="#f8fafc", width=1.4)),
        hovertext=node_hover, hoverinfo="text", showlegend=False
    ))
    for li, xx in enumerate(layer_x):
        layer_label = "INPUT" if li == 0 else "OUTPUT · SIGMOID" if li == len(layer_sizes)-1 else f"HIDDEN {li} · {hidden_activation.upper()}"
        net_fig.add_annotation(x=float(xx), y=1.35, text=layer_label, showarrow=False,
                               font=dict(size=10, color="#dbe5ff"))
    net_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,22,40,0.45)", height=370,
        margin=dict(l=15, r=15, t=35, b=10),
        xaxis=dict(visible=False, range=[-0.12, 1.12]),
        yaxis=dict(visible=False, range=[-1.45, 1.55]),
        showlegend=False
    )
    st.plotly_chart(net_fig, use_container_width=True)
    st.caption("Teal = inputs · Blue = hidden neurons · Orange = the single output. Hover over a neuron to inspect its current value.")

    st.markdown("### Calculations: one layer at a time")
    st.markdown("For each neuron, **weighted sum** means multiply each incoming value by its connecting weight, then add the bias. That gives z. The activation function converts z into the value passed to the next layer.")
    for layer_idx, (W, b, z, a) in enumerate(zip(weights, biases, zs, activations[1:])):
        is_output = layer_idx == hidden_layers
        name = "Output layer" if is_output else f"Hidden layer {layer_idx + 1}"
        act_name = "Sigmoid" if is_output else hidden_activation
        with st.expander(f"{name} — {act_name} activation", expanded=True):
            for neuron_idx in range(W.shape[1]):
                neuron_label = "Output neuron" if is_output else f"Neuron H{layer_idx+1}-{neuron_idx+1}"
                st.markdown(f"**{neuron_label}**")
                terms = [f"({activations[layer_idx][i]:.5f})({W[i, neuron_idx]:.5f})" for i in range(W.shape[0])]
                equation = " + ".join(terms) + f" + ({b[neuron_idx]:.5f})"
                st.latex(r"z = \sum_i a_iw_i+b")
                st.write(f"Substitute the values: {equation}")
                st.latex(f"z = {float(z[neuron_idx]):.5f}")
                if act_name == "Sigmoid":
                    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
                elif act_name == "ReLU":
                    st.latex(r"\mathrm{ReLU}(z)=\max(0,z)")
                elif act_name == "Tanh":
                    st.latex(r"\tanh(z)")
                elif act_name == "Leaky ReLU":
                    st.latex(r"f(z)=\max(z,0.01z)")
                else:
                    st.latex(r"f(z)=z")
                st.write(f"Apply {act_name}: output a = **{float(a[neuron_idx]):.5f}**.")
                if not is_output:
                    st.caption("This activation becomes an input to every connected neuron in the next layer.")
                else:
                    st.caption("This single sigmoid output is the predicted probability for class 1.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted probability ŷ", fmt(prediction))
    m2.metric("Predicted class (threshold 0.5)", str(int(prediction >= 0.5)))
    m3.metric("Binary cross-entropy loss", fmt(loss))
    st.markdown("#### Interpret the output")
    st.write(
        f"The output neuron gives ŷ = {prediction:.5f}. With a 0.5 threshold, the predicted class is "
        f"{int(prediction >= 0.5)}. The target is y = {true_label}. The loss below compares the probability "
        "with that target; it is not simply the difference between the predicted class and the target."
    )
    st.markdown(
        '<div class="lab-card"><div class="kicker">BINARY CROSS-ENTROPY LOSS</div>'
        '<h3>L = −[y ln(ŷ) + (1 − y) ln(1 − ŷ)]</h3>'
        '<p><b>y</b> is the correct label; <b>ŷ</b> is the predicted probability. '
        '<b>ln</b> is the natural logarithm. The loss is small when the model gives high probability to the correct class, '
        'and large when it is confidently wrong.</p></div>',
        unsafe_allow_html=True
    )


# ---------- Full backpropagation ----------
with tabs[2]:
    st.markdown("### Backpropagation — one learning step, clearly explained")
    st.markdown(
        f'<div class="lab-card"><b>We are continuing the exact same example</b><br>'
        f'Inputs: x₁ = {x1:.2f}, x₂ = {x2:.2f} &nbsp; · &nbsp; Correct answer: y = {true_label} '
        f'&nbsp; · &nbsp; Current prediction: ŷ = {prediction:.5f} '
        f'&nbsp; · &nbsp; Learning rate: η = {learning_rate:.3f}</div>',
        unsafe_allow_html=True
    )
    st.write("Think of backpropagation as four connected actions: measure the prediction error, send that error backwards through the network, use it to adjust each parameter, and check the new prediction.")

    # Step 1: show the actual forward values compactly.
    with st.container(border=True):
        st.markdown("#### 1. Start with the forward-pass result")
        st.markdown("The network has already processed the two inputs through its hidden layers and produced one probability.")
        st.markdown(
            f"- Input features: **({x1:.2f}, {x2:.2f})**  \n"
            f"- Target (correct class): **{true_label}**  \n"
            f"- Predicted probability: **{prediction:.5f}**  \n"
            f"- Predicted class at threshold 0.5: **{int(prediction >= 0.5)}**"
        )
        st.caption("This is the same output shown in the Forward Pass tab. We do not create a new example here.")

    # Step 2: BCE with natural log, expanded with actual values.
    with st.container(border=True):
        st.markdown("#### 2. Measure the error — binary cross-entropy")
        st.markdown("Because this network has one sigmoid output and a binary target, we use binary cross-entropy (BCE).")
        st.latex(r"L=-\left[y\ln(\hat{y})+(1-y)\ln(1-\hat{y})\right]")
        st.markdown(
            f"Substitute the actual target and prediction:  \n"
            f"$L=-\\left[{true_label}\\ln({prediction:.5f})+(1-{true_label})\\ln(1-{prediction:.5f})\\right]$"
        )
        st.markdown(f"**Loss = {loss:.5f}**")
        st.caption("The natural logarithm ln penalizes confident wrong predictions strongly. A smaller loss means the predicted probability better matches the correct label.")

    # Gradients are calculated from output to input.
    deltas = [None] * len(weights)
    gradients_W = [None] * len(weights)
    gradients_b = [None] * len(weights)
    deltas[-1] = np.array([prediction - true_label], dtype=float)
    for layer_idx in range(len(weights) - 2, -1, -1):
        deltas[layer_idx] = (weights[layer_idx + 1] @ deltas[layer_idx + 1]) * activation_derivative(hidden_activation, zs[layer_idx])
    for layer_idx in range(len(weights)):
        gradients_W[layer_idx] = np.outer(activations[layer_idx], deltas[layer_idx])
        gradients_b[layer_idx] = deltas[layer_idx].copy()

    # Visual backpropagation map: activations move left-to-right; error gradients move right-to-left.
    with st.container(border=True):
        st.markdown("#### See the error travel through the network")
        st.write("The same network is shown below. Each neuron displays its forward activation (a) and backward gradient (δ). The grey connections show how neurons are connected; the arrows above and below show the two directions of computation.")
        layer_x_positions = np.linspace(0.08, 0.92, len(layer_sizes))
        node_xy = {}
        for li, size in enumerate(layer_sizes):
            ys_layer = np.linspace(-0.72, 0.72, size) if size > 1 else np.array([0.0])
            for ni, yy in enumerate(ys_layer):
                node_xy[(li, ni)] = (float(layer_x_positions[li]), float(yy))

        visual_fig = go.Figure()

        # Draw the network connections first so they sit behind the nodes.
        for li in range(len(layer_sizes) - 1):
            for i in range(layer_sizes[li]):
                for j in range(layer_sizes[li + 1]):
                    x0, y0 = node_xy[(li, i)]
                    x1, y1 = node_xy[(li + 1, j)]
                    visual_fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1], mode="lines",
                        line=dict(color="rgba(148,163,184,0.28)", width=1.3),
                        hoverinfo="skip", showlegend=False
                    ))

        # Add forward and backward direction arrows in separate lanes.
        for li in range(len(layer_sizes) - 1):
            x_left = float(layer_x_positions[li])
            x_right = float(layer_x_positions[li + 1])
            mid_x = (x_left + x_right) / 2
            visual_fig.add_annotation(
                x=x_right - 0.025, y=1.10, ax=x_left + 0.025, ay=1.10,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=2.5, arrowcolor="#35c9a5"
            )
            visual_fig.add_annotation(
                x=x_left + 0.025, y=-1.10, ax=x_right - 0.025, ay=-1.10,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=2.5, arrowcolor="#ff8a75"
            )

        node_colors, node_text, node_hover = [], [], []
        for li, size in enumerate(layer_sizes):
            for ni in range(size):
                x_pos, y_pos = node_xy[(li, ni)]
                a_value = float(activations[li][ni])
                if li == 0:
                    delta_text = "input"
                    hover = f"Input x{ni+1}<br>Value = {a_value:.5f}"
                    node_colors.append("#2dd4bf")
                    label = f"x{ni+1}<br>{a_value:.3f}"
                else:
                    delta_value = float(deltas[li - 1][ni])
                    delta_text = f"δ = {delta_value:.4f}"
                    layer_label = "Output" if li == len(layer_sizes) - 1 else f"H{li}-{ni+1}"
                    hover = (
                        f"{layer_label}<br>Activation a = {a_value:.6f}"
                        f"<br>Pre-activation z = {zs[li-1][ni]:.6f}"
                        f"<br>Backward gradient δ = {delta_value:.6f}"
                    )
                    node_colors.append("#ffb86b" if li == len(layer_sizes)-1 else "#8daeff")
                    label = f"{'ŷ' if li == len(layer_sizes)-1 else f'H{li}-{ni+1}'}<br>a={a_value:.3f}<br>δ={delta_value:.3f}"
                node_text.append(label)
                node_hover.append(hover)

        visual_fig.add_trace(go.Scatter(
            x=[node_xy[(li, ni)][0] for li, size in enumerate(layer_sizes) for ni in range(size)],
            y=[node_xy[(li, ni)][1] for li, size in enumerate(layer_sizes) for ni in range(size)],
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont=dict(color="#101827", size=10, family="DM Sans"),
            marker=dict(size=66, color=node_colors, line=dict(color="#f8fafc", width=1.5)),
            hovertext=node_hover, hoverinfo="text", showlegend=False
        ))
        for li, x_pos in enumerate(layer_x_positions):
            if li == 0:
                title_text = "INPUT"
            elif li == len(layer_sizes) - 1:
                title_text = "OUTPUT"
            else:
                title_text = f"HIDDEN {li}"
            visual_fig.add_annotation(
                x=float(x_pos), y=1.38, text=title_text, showarrow=False,
                font=dict(color="#e5eaf5", size=11, family="DM Sans")
            )
        visual_fig.add_annotation(
            x=0.50, y=1.10, text="FORWARD: activations →",
            showarrow=False, yshift=15,
            font=dict(color="#35c9a5", size=11, family="DM Sans")
        )
        visual_fig.add_annotation(
            x=0.50, y=-1.10, text="← BACKWARD: gradients",
            showarrow=False, yshift=-15,
            font=dict(color="#ff8a75", size=11, family="DM Sans")
        )
        visual_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,22,40,0.35)",
            height=470,
            margin=dict(l=15, r=15, t=55, b=45),
            xaxis=dict(visible=False, range=[-0.02, 1.02]),
            yaxis=dict(visible=False, range=[-1.48, 1.62]),
            showlegend=False
        )
        st.plotly_chart(visual_fig, use_container_width=True)
        st.markdown(
            '<div class="lab-card">'
            '<b><span style="color:#35c9a5">Green direction:</span> forward pass — values move from inputs toward the output.</b><br>'
            '<b><span style="color:#ff8a75">Coral direction:</span> backpropagation — gradients move from the output back toward the inputs.</b><br>'
            '<span style="color:#cbd5e1">Inside each hidden/output neuron, a is the value passed forward and δ is the gradient sent backward.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    with st.container(border=True):
        st.markdown("#### 3. Send the error backwards — calculate gradients")
        st.markdown("A gradient tells us how much the loss changes when a weight or bias changes. We start at the output neuron and work backwards, one hidden layer at a time.")

        st.markdown("**3A. Output neuron**")
        st.markdown("For a sigmoid output combined with binary cross-entropy, the derivative simplifies to:")
        st.latex(r"\delta_{\mathrm{out}}=\frac{\partial L}{\partial z_{\mathrm{out}}}=\hat{y}-y")
        st.markdown(f"Using this example:  \n$\\delta_{{out}}={prediction:.5f}-{true_label}=\\mathbf{{{deltas[-1][0]:.5f}}}$")
        st.markdown("Now calculate the gradient for each connection entering the output neuron:")
        st.latex(r"\frac{\partial L}{\partial w_i}=a_i\delta_{\mathrm{out}},\qquad \frac{\partial L}{\partial b}=\delta_{\mathrm{out}}")
        for i in range(len(activations[-2])):
            st.markdown(
                f"- Connection from previous neuron {i+1}: "
                f"$\\partial L/\\partial w_{{{i+1}}}={activations[-2][i]:.5f}\\times({deltas[-1][0]:.5f})"
                f"=\\mathbf{{{gradients_W[-1][i,0]:.5f}}}$"
            )
        st.markdown(f"- Output bias gradient: $\\partial L/\\partial b=\\mathbf{{{gradients_b[-1][0]:.5f}}}$")

        # Each hidden layer is presented in reverse order, as the error travels backwards.
        for layer_idx in range(len(weights) - 2, -1, -1):
            layer_number = layer_idx + 1
            derivs = activation_derivative(hidden_activation, zs[layer_idx])
            st.markdown(f"**3B. Hidden layer {layer_number}**")
            st.markdown(
                "Each neuron receives a backward signal from the next layer. "
                "We multiply that signal by the derivative of this layer's activation function."
            )
            st.latex(r"\delta^{(l)}=\left(W^{(l+1)}\delta^{(l+1)}\right)\odot f'(z^{(l)})")
            for j in range(len(deltas[layer_idx])):
                back_signal = float(weights[layer_idx + 1][j, :] @ deltas[layer_idx + 1])
                st.markdown(
                    f'<div class="lab-card"><b>Hidden neuron H{layer_number}-{j+1}</b><br>'
                    f'<span style="color:#cbd5e1">1. Signal arriving from the next layer:</span><br>'
                    f'<b>{back_signal:.5f}</b><br>'
                    f'<span style="color:#cbd5e1">2. Activation derivative at z = {zs[layer_idx][j]:.5f}:</span><br>'
                    f'<b>{derivs[j]:.5f}</b><br>'
                    f'<span style="color:#cbd5e1">3. Multiply them to get this neuron’s gradient:</span><br>'
                    f'<b>{back_signal:.5f} × {derivs[j]:.5f} = {deltas[layer_idx][j]:.5f}</b><br>'
                    f'<span style="color:#cbd5e1">4. Bias gradient:</span> <b>{gradients_b[layer_idx][j]:.5f}</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown("For every incoming connection, multiply its input activation by this neuron's gradient:")
                for i in range(gradients_W[layer_idx].shape[0]):
                    st.markdown(
                        f"- $\\partial L/\\partial w_{{{i+1},{j+1}}}="
                        f"{activations[layer_idx][i]:.5f}\\times({deltas[layer_idx][j]:.5f})"
                        f"=\\mathbf{{{gradients_W[layer_idx][i,j]:.5f}}}$"
                    )

    with st.container(border=True):
        st.markdown("#### 4. Update the weights and biases")
        st.markdown("Now use gradient descent. The gradient points toward increasing loss, so we subtract a learning-rate-sized step from each parameter.")
        st.latex(r"w_{\mathrm{new}}=w_{\mathrm{old}}-\eta\frac{\partial L}{\partial w}")
        st.latex(r"b_{\mathrm{new}}=b_{\mathrm{old}}-\eta\frac{\partial L}{\partial b}")
        st.markdown(f"**Learning rate:** η = {learning_rate:.3f}. A smaller learning rate makes smaller changes; a larger one makes bigger changes.")
        updated_weights, updated_biases = [], []
        for layer_idx in range(len(weights)):
            W_new = weights[layer_idx] - learning_rate * gradients_W[layer_idx]
            b_new = biases[layer_idx] - learning_rate * gradients_b[layer_idx]
            updated_weights.append(W_new)
            updated_biases.append(b_new)
            layer_title = "Output layer" if layer_idx == hidden_layers else f"Hidden layer {layer_idx+1}"
            with st.expander(f"See parameter updates — {layer_title}", expanded=(layer_idx == len(weights)-1)):
                for i in range(W_new.shape[0]):
                    for j in range(W_new.shape[1]):
                        st.markdown(
                            f"- **Weight W{layer_idx+1}[{i+1},{j+1}]:**  \n"
                            f"  ${weights[layer_idx][i,j]:.5f}-{learning_rate:.3f}\\times"
                            f"({gradients_W[layer_idx][i,j]:.5f})=\\mathbf{{{W_new[i,j]:.5f}}}$"
                        )
                for j in range(len(b_new)):
                    st.markdown(
                        f"- **Bias b{layer_idx+1}[{j+1}]:**  \n"
                        f"  ${biases[layer_idx][j]:.5f}-{learning_rate:.3f}\\times"
                        f"({gradients_b[layer_idx][j]:.5f})=\\mathbf{{{b_new[j]:.5f}}}$"
                    )

    # Run the same input through the updated network once.
    updated_acts = [x_vec]
    for layer_idx, (W_new, b_new) in enumerate(zip(updated_weights, updated_biases)):
        z_new = updated_acts[-1] @ W_new + b_new
        a_new = activation(hidden_activation, z_new) if layer_idx < hidden_layers else sigmoid(z_new)
        updated_acts.append(a_new)
    updated_prediction = float(updated_acts[-1][0])
    updated_loss = float(-(true_label * np.log(updated_prediction + eps) + (1 - true_label) * np.log(1 - updated_prediction + eps)))

    with st.container(border=True):
        st.markdown("#### 5. Check whether the update helped")
        col1, col2 = st.columns(2)
        col1.metric("Prediction before", fmt(prediction))
        col2.metric("Prediction after one update", fmt(updated_prediction), delta=f"{updated_prediction-prediction:+.5f}")
        col3, col4 = st.columns(2)
        col3.metric("Loss before", fmt(loss))
        col4.metric("Loss after one update", fmt(updated_loss), delta=f"{updated_loss-loss:+.5f}", delta_color="inverse")
        if updated_loss < loss:
            st.success(f"The loss decreased by {loss-updated_loss:.5f}. This update improved the model on this example.")
        elif updated_loss > loss:
            st.warning(f"The loss increased by {updated_loss-loss:.5f}. Try a smaller learning rate and compare.")
        else:
            st.info("The loss did not change at the displayed precision.")
        st.caption("The prediction is the model's estimated probability. The loss is the better measure of whether the update improved the answer: lower loss is better.")


st.divider()
st.markdown('<p class="subtle">Neural Network Learning Lab · Streamlit · NumPy · Pandas · Plotly. The training lab has been removed; this version focuses on one inspectable example through forward and backward propagation.</p>', unsafe_allow_html=True)
