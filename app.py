"""Gradio demo: pick two drugs and get the predicted interaction type."""
import gradio as gr
from src.predict import Predictor

# On Hugging Face ZeroGPU hardware every app needs at least one @spaces.GPU function.
# The model here is tiny and runs on CPU; the decorator only satisfies the platform
# and does nothing when running locally or on other hardware.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

predictor = Predictor("models")
CHOICES = predictor.drug_choices()

INTRO = """
# Drug–drug interaction type predictor

Pick two drugs and the model predicts the **type** of their interaction.
It combines five views of each drug (chemical structure, targets, enzymes, pathways and a
ComplEx knowledge-graph embedding) with a multi-view attention network.

**Drug A is the drug that is affected**; swapping A and B can change the result.
"""

DISCLAIMER = (
    "*Research demo, not medical advice. The model only knows the 572 drugs it was trained on. "
    "Data source: DrugBank (CC BY-NC 4.0, non-commercial use) and DRKG.*"
)


@gpu
def predict(choice_a, choice_b):
    if not choice_a or not choice_b:
        return None, "Choose two drugs."
    try:
        out = predictor.predict(predictor.choice_to_id[choice_a],
                                predictor.choice_to_id[choice_b], top_k=3)
    except ValueError as err:
        return None, str(err)
    labels = {p["text"]: p["probability"] for p in out["predictions"]}
    if out["in_dataset"]:
        note = ("This pair is **in the training data** with the recorded type: "
                f"{out['dataset_label']} (so the prediction is not a test of generalisation).")
    else:
        note = "This pair is **not** in the training data, so this is a genuine prediction."
    return labels, note


with gr.Blocks(title="DDI type predictor") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        drug_a = gr.Dropdown(CHOICES, label="Drug A (the affected drug)")
        drug_b = gr.Dropdown(CHOICES, label="Drug B")
    button = gr.Button("Predict", variant="primary")
    result = gr.Label(num_top_classes=3, label="Predicted interaction type")
    note = gr.Markdown()
    gr.Markdown(DISCLAIMER)
    button.click(predict, [drug_a, drug_b], [result, note])

if __name__ == "__main__":
    demo.launch()