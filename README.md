# Neural Network Learning Lab

An interactive Streamlit app for teaching activation functions, forward propagation, backpropagation, and basic training.

## Features
- Activation function curves and live input/output explorer
- Interactive network architecture visualization
- Numeric single-neuron forward pass
- Step-by-step sigmoid + binary cross-entropy backpropagation
- Tiny NumPy XOR training lab with loss curve and predictions

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy.

## Notes
- The network diagram is an educational architecture visualization; the numeric neuron and backpropagation examples show explicit calculations.
- The training lab uses a small XOR dataset and manual NumPy backpropagation so students can inspect the learning process.
