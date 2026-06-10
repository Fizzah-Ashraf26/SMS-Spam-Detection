from transformers import pipeline
import gradio as gr

classifier = pipeline(
    'text-classification',
    model='fizzah26/sms-spam-distilbert'
)

def predict(sms_text):
    if not sms_text.strip():
        return "Please enter a message."
    result = classifier(sms_text, truncation=True, max_length=128)[0]
    label  = result['label'].upper()
    score  = result['score']
    if label == 'SPAM':
        return f"SPAM — {score:.2%} confidence"
    else:
        return f"HAM (Not Spam) — {score:.2%} confidence"

custom_css = """
.gradio-container { max-width: 640px !important; margin: auto !important; }
#component-0 { border-radius: 12px; overflow: hidden; }
footer { display: none !important; }
.share-button { display: none !important; }
"""

demo = gr.Interface(
    fn          = predict,
    inputs      = gr.Textbox(
                    lines       = 6,
                    placeholder = "Paste your SMS message here...",
                    label       = "SMS Message"
                  ),
    outputs     = gr.Textbox(label="Prediction"),
    title       = "SMS Spam Detector",
    description = "Enter an SMS message to classify it as spam or ham",
    examples    = [
        ["Congratulations! You won a FREE iPhone. Click here to claim NOW!"],
        ["Hey, are we still meeting for lunch tomorrow?"],
        ["URGENT: Your bank account is suspended. Call 0800-FREE now!"],
        ["Can you send me the notes from today's class?"],
    ],
)

demo.launch(
    theme = gr.themes.Soft(
        primary_hue = gr.themes.colors.blue,
        neutral_hue = gr.themes.colors.slate,
        font        = gr.themes.GoogleFont("Inter"),
    ),
    css = custom_css,
)