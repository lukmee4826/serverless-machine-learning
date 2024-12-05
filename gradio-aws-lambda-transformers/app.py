import gradio as gr
from transformers import pipeline

# Load transformers pipeline from a local directory (model)
clf = pipeline("sentiment-analysis", model="model/")

# Predict function used by Gradio
def sentiment(payload):
    prediction = clf(payload, return_all_scores=True)
    # Convert list to dict
    result = {}
    for pred in prediction[0]:
        result[pred["label"]] = pred["score"]
    return result

# Build the Gradio app
with gr.Blocks() as gradio_app:
    gr.Markdown("### Sentiment Analysis")
    text_input = gr.Textbox(placeholder="Enter a positive or negative sentence here...")
    output_label = gr.Label()
    examples = gr.Examples(
        examples=["I Love Serverless Machine Learning", "Running Gradio on AWS Lambda is amazing"],
        inputs=text_input
    )
    classify_button = gr.Button("Analyze Sentiment")
    
    classify_button.click(fn=sentiment, inputs=text_input, outputs=output_label)

# Run the app
gradio_app.launch(share=True)
