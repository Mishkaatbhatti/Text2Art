import gradio as gr
from huggingface_hub import InferenceClient
import os
import io
import zipfile
from PIL import Image

# Hugging Face Inference Client
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

def generate_images(prompt, num_images=3):
    images = []
    for _ in range(num_images):
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        images.append(image)

    import tempfile

# Save all images into a zip file on disk
    temp_zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(temp_zip_path, "w") as zip_file:
        for idx, img in enumerate(images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zip_file.writestr(f"image_{idx+1}.png", img_bytes.getvalue())

    return images, temp_zip_path


with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Text-to-Image Generator (Stable Diffusion XL)")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Enter your prompt", placeholder="e.g. Astronaut riding a horse")
            num_images = gr.Slider(1, 6, value=3, step=1, label="Number of images")
            generate_btn = gr.Button("Generate Images 🚀")

        with gr.Column(scale=3):
            gallery = gr.Gallery(label="Generated Images", columns=3, height="auto")
            download = gr.File(label="Download All Images", file_types=[".zip"])

    generate_btn.click(
        fn=generate_images,
        inputs=[prompt, num_images],
        outputs=[gallery, download]
    )

demo.launch()
