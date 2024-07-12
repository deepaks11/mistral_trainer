import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "models/Mistral-7b-v2-story_50_r256"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=0,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def generate_text(prompt, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1):

    response = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
    )
    try:
        return response[0]['generated_text'].split("</s>")[1].split(",.")[0].split(".")[0]

    except:

        return response[0]['generated_text']


if __name__ == "__main__":

    while True:
        question = input("Enter your question: ")
        prompt = f"<s>[INST] {question} [/INST]  </s>"
        generated_text = generate_text(prompt)
        print(f"response: {generated_text}")