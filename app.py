import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@st.cache_resource
def load_model():
    base_model_name = "gpt2"
    adapter_path = "./humanizer-lora"
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="Text Humanizer", page_icon="💬")
st.title("💬 Text Humanizer AI")
st.caption("Make your text sound natural and human-like")

tokenizer, model = load_model()

with st.sidebar:
    max_new = st.slider("Max tokens", 16, 256, 80)
    temp = st.slider("Temperature", 0.1, 1.5, 0.8)

instruction = st.text_area("Text to humanize:", 
    "Rewrite this in a friendly tone: 'Submit your report now.'")

if st.button("✨ Humanize"):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new, 
        do_sample=True, temperature=temp, 
        pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(result.split("### Response:")[-1].strip())
