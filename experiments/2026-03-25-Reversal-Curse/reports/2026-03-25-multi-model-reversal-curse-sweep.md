# Reversal curse toy sweep — concise report

**Task:** Same fact sheet: “The husband of Aleron is Bexley,” etc. **Forward:** “Who is Aleron’s husband?” → *Bexley*. **Reverse:** “Who is Bexley’s wife?” → *Aleron*.  

## Both directions look correct

| Model | Forward (headline) | Reverse (headline) |
|--------|--------------------|---------------------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Aleron's husband is Bexley. | Bexley's wife is Aleron. |
| Qwen/Qwen2.5-0.5B-Instruct | Aleron's husband is Bexley. | Bexley's wife is Aleron. |
| Qwen/Qwen2.5-3B-Instruct | *Bexley* | *Aleron* |
| Qwen/Qwen3-4B-Instruct-2507 | *Bexley* | Bexley's wife is Aleron. |
| Qwen/Qwen3.5-0.8B | *Bexley* | *Aleron* |
| Qwen/Qwen3.5-2B | *Bexley* | *Aleron* |
| ibm-granite/granite-3.1-2b-instruct | *Bexley* | *Aleron* |
| meta-llama/Llama-3.2-3B-Instruct | *Bexley.* | *Aleron.* |

## Forward OK, reverse weak or wrong

| Model | Forward | Reverse |
|--------|---------|---------|
| openai-community/gpt2-large | *Bexley* | Mentions *Aleron* but invents “son Corwin” (wrong) |
| openai-community/gpt2-xl | *Bexley* | Role error: “Bexley is the wife of Aleron” |
| EleutherAI/gpt-neo-1.3B | Aleron's husband is Bexley. | Stalls at “The answer is:” |
| meta-llama/Llama-3.2-1B-Instruct | *Bexley.* | “Bexley's wife is *Damaris*.” — wrong; *Damaris* is another fact-row spouse (bleed from context). See **verbatim log** below. |
| bigcode/starcoder2-3b | Broken (“Aleron's wife is.”) | *Aleron* |
| Qwen/Qwen3-0.6B | Aleron's husband is Bexley. | Wrong / hedge: says wife “not mentioned in the facts,” etc. |

## Forward wrong, reverse OK

| Model | Forward | Reverse |
|--------|---------|---------|
| Qwen/Qwen3-1.7B | *Haleth.* (wrong; gold *Bexley*) | Bexley's wife is Aleron. |

## Wrong or non-answers (both or forward)

| Model | What went wrong |
|--------|------------------|
| distilgpt2 | Off-topic narrative; no correct name |
| openai-community/gpt2-medium | Off-topic narrative; no correct name |
| facebook/opt-1.3b | List of names; reverse “Aleron's wife is Bexley” (vs facts) |
| facebook/opt-2.7b | Invents relations; reverse *Damaris* |
| EleutherAI/pythia-1b | Garbled roles / spouses |
| bigscience/bloom-560m | Exam-style junk (“" A: (1) The answer to this”) — not the golds |
| bigscience/bloom-1b7 | Rambling parentheticals — not the golds |
| stabilityai/stablelm-2-1_6b | Multiple-choice stub / quoted prompt — not the golds |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | **Swapped:** forward *Aleron* (want *Bexley*); reverse *Bexley* (want *Aleron*) |
| microsoft/phi-2 | Forward cites marriage loosely; reverse contradicts facts |
| togethercomputer/RedPajama-INCITE-Base-3B-v1 | Returns questions, not names |
| allenai/OLMo-1B-hf | Off-topic memorized text |
| allenai/OLMo-2-0425-1B | Echoes “Answer with only the final name.” |

## meta-llama/Llama-3.2-1B-Instruct — verbatim single-model run

Hub load line omitted; console showed `torch_dtype` deprecation from older calls (script now uses `dtype=` in `from_pretrained`). Transformers also noted unused generation flags unless `TRANSFORMERS_VERBOSITY=info`.

**Forward (expected *Bexley*):**

> Bexley.

**Reverse (expected *Aleron*):**

> Bexley's wife is Damaris.

So the 1B Instruct gets forward right but reverse collapses to another name from the fact sheet—contrast **Llama-3.2-3B-Instruct**, which matched both golds in the sweep.

**Takeaway (this rerun):** Strong instruct/chat models (TinyLlama, Qwen2.5/3.5, Granite, **Llama-3.2-3B**) still nail **both** directions; **Llama-3.2-1B** does not (reverse error above). **Qwen3-0.6B** (`enable_thinking=False`) is good forward but fails reverse; **Qwen3-1.7B** is the opposite—interesting asymmetry. BLOOM / StableLM now produce *visible* headlines (token-window + `min_new_tokens`) but answers stay **wrong** for this task. Smaller base GPT-2/OPT/Pythia/RedPajama/OLMo remain poor fits.
