# api_service.py
# Proof-of-Concept FastAPI service for image+text query to a quantized vision LLM

import io
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration, MllamaForCausalLM
from PIL import Image
from typing import Optional

# Model and quantization config
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

DEFAULT_SYSTEM_PROMPT = "Please provide 2-3 smart suggested responses to participate in the conversation."

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MllamaForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB_CONFIG, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Initialize FastAPI
app = FastAPI(title="Conversation-Response-Suggestion API", version="0.1")

@app.post("/haiku")
async def haiku(prompt: str = Form("If I had to write a haiku, it would be:")):
    """
    Creates a haiku from the prompt.
    Used to test the LLM.

    Parameters
    ----------
    prompt : str, optional
        prompt for the model, by default Form(...)

    Returns
    -------
    JSONResponse
        This is the response from the model
    """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=256, do_sample=True, temperature=0.6)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return JSONResponse({
        "response": result
    })

@app.post("/suggest")
async def suggest(
    conversation_input: str = Form(...),
    history: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None)
):
    """Accepts text of a conversation and provides suggested responses.

    Parameters
    ----------
    conversation_input : str, optional
        This is the portion of conversation, by default Form(...)
    """
    messages = []

    if history:
        for turn in history.split("|||"):
            role, content = turn.strip().split("::", 1)
            messages.append({"role": role.strip(), "content": content.strip()})

    if system_prompt:
        messages.append({"role": "assistant", "content": system_prompt})
    else:
        messages.append({
            "role": "assistant",
            "content": "Please provide 2-3 smart suggested responses to participate in the conversation."
        })

    messages.append({
        "role": "user",
        "content": conversation_input
    })

    # Format the messages into a single prompt string
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    generate_ids = model.generate(
        inputs.input_ids,
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    # Clean and split suggestions
    suggestions = [s.strip() for s in output.split("\n") if s.strip()]

    return JSONResponse({
        f"Conversation suggestion response {i+1}": suggestion
        for i, suggestion in enumerate(suggestions[:3])
    })

@app.get("/healthz")
async def health_check():
    """
    Health check endpoint to verify model rediness.
    Returns 200 OK if the model is loaded and accessible.
    """
    try:
        _ = model.device
        return JSONResponse(status_code=200, content={"status":"ok"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error", "detail": str(e)})

if __name__ == "__main__":
    # To run locally: uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)