import streamlit as st
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed, pipeline
import torch
import re

# Model loading section (same as before)
model_name = "microsoft/BioGPT-Large"
model = BioGptForCausalLM.from_pretrained(model_name)
tokenizer = BioGptTokenizer.from_pretrained(model_name)

# Check for GPU availability (same as before)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    st.info("Using GPU: " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    st.info("Using CPU")

def generate_bio_text(prompt, max_length=512, num_beams=5, temperature=1.0, top_k=50):
    """Generates text using BioGPT Large with adjustable parameters and removes artifacts.

    Args:
        prompt: The starting sentence or prompt for generation.
        max_length: The maximum length of the generated text (default: 512).
        num_beams: The number of beams to use for beam search (default: 5).
        temperature: Controls randomness of the generated text (default: 1.0).
        top_k: Samples the next token from the top k most likely ones (default: 50).

    Returns:
        The generated text continuation without HTML tags and control characters.
    """
    try:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        output = model.generate(
            input_ids,
            max_length=max_length,
            length_penalty=0.5,
            num_beams=num_beams,
            top_k=top_k,
            temperature=temperature,
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove HTML tags and control characters using a regular expression
        cleaned_text = re.sub(r"<[^>]*>|▃", "", generated_text)
        # Truncate at the last dot to avoid incomplete sentences
        # truncated_text = re.sub(r"\.(.*)$", "", cleaned_text)  # Matches and removes any text after the last dot
        return cleaned_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return "An error occurred during generation. Please try again." + e

# Streamlit app layout
st.title("BioGPT Large Text Generation with Control")
st.markdown("""
Generate text related to biology using BioGPT Large. Provide a starting sentence or prompt, and adjust various parameters to control the generated text.

* **Maximum Length:** Controls the maximum length of the generated text.
* **Number of Beams:** Affects the quality and diversity of the generated text.
* **Temperature:** Higher values increase randomness and creativity, potentially leading to less coherent text.
* **Top K:** Samples the next token from the top k most likely ones, influencing the focus of the generation.
""")

prompt = st.text_area("Prompt or Starting Sentence", lines=4)
max_length = st.slider("Maximum Length", minimum=30, maximum=400, value=512, step=1)
num_beams = st.slider("Number of Beams", minimum=1, maximum=5, value=5, step=1)
temperature = st.slider("Temperature", minimum=0.4, maximum=1.0, value=1.0, step=0.1)
top_k = st.slider("Top K", minimum=10, maximum=30, value=50, step=1)

if st.button("Generate"):
    generated_text = generate_bio_text(prompt, max_length, num_beams, temperature, top_k)
    st.success(generated_text)
