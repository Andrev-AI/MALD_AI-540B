# MALD_AI-540B
Introducing MALD LLM (Advanced Language and Dialogue Model pt-br)

MALD, from the Portuguese "Modelo Avançado de Linguagem e Diálogo" (Advanced Language and Dialogue Model), is an LLM based on the Transformers architecture that has 540 billion parameters and a context window of 200k (200 thousand) tokens. Here is just the model code but without the training weights. MALD has more than 3 times more parameters than GPT-3.5 (175B), more parameters than LaMDA/BARD (137B), and the same amount as Google PaLM (540B). MALD is still a basic model with the aim of testing and training my programming skills and understanding of LLM.

# Code Structure

MALD-540B was made in python and uses the Torch framework. MALD-540B has `Transformers()` which refers to text generation Question and Answer. There are the generation parameters `top_p`, `top_k`, `frequency_penalty`, `max_tokens` that define how the model will generate the answers. There is the `summarize()` function for summarizing inputs, which also uses the inference parameters to regulate the quality of the response. MALD-540B in a timeline is the second version of MALD, its predecessor being MALD-180B. MALD-540B compared to its predecessor MALD-180B, in addition to having 3x the number of parameters and a context window much larger than the 8k (8 thousand) of 180B, has a slightly better code architecture and a function dedicated exclusively to summarize, something that the previous version did not have, and to perform summaries, in the text generation part, instruction was given to the model to summarize.

# Multimodal?

Not yet. MALD is still under development, however, it can be adapted and tuned to receive modalities beyond text but natively version 1, both 180B and 540B do not have multimodalities, initially accepting and working only with text. In the future, we will seek to introduce modalities such as image, video, and audio.

# Why 540B? Why "MALD"?

MALD because it came from a concept, from an acronym that in pt-br is: Modelo Avançado de Linguagem e Diálogo (Advanced Language and Dialogue Model). The name was chosen spontaneously but with the intention of transmitting an LLM language model but with advanced capabilities. MALD also serves for programmers who want to develop or study LLM models, no matter the size already have a base of the code ready, just needing to train. 540B arose as inspiration in Google PaLM (Pathways Language Model) which also has 540B parameters.

# Datasets:

* "HuggingFaceFW/fineweb". Dataset with an impressive 15T (15 trillion) tokens, enough to train large AI models with high quality. It is speculated that GPT-4 was trained with 13T of tokens, so this dataset has 2T more tokens. And it presented better results in model training than datasets like C4, Dolma, The Pile etc.
* "Anthropic/hh-rlhf". Dataset from the renowned Anthropic, is a reinforcement dataset based on human feedback, (RLHF) helps in issues of harmful and disrespectful messages. For "Uncensored" models it may not be recommended.
* "allenai/c4". One of the most popular datasets, very capable and its `multilingual` version with 9TB storage and 108 languages can mainly help the model in fluency and polyglot capabilities.
