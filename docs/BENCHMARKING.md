# Style Benchmarking Plan

This project should evaluate voice replication as style transfer, not as prediction of what a real person would believe or say. A good benchmark therefore needs four separate checks:

1. Style strength: does the output resemble the target author's writing habits?
2. Content control: does it answer the new prompt instead of drifting into unrelated source topics?
3. Fluency: is the text coherent and readable?
4. Copy safety: does it avoid quoting or near-copying training passages?

## Automatic Metrics

The current `evaluate` command reports:

- `word_f1`: lexical overlap with the held-out gold response. Useful for Enron-style reply tasks, but not enough by itself.
- `length_ratio`: whether output length is in the same range as gold.
- `greeting_match` and `signoff_match`: useful for email corpora.
- `profile_copy_5gram_rate`: percentage of generated 5-grams that appear in profile examples.
- `profile_longest_copy_run`: longest contiguous token sequence copied from profile examples.
- `style_distance`: lightweight distance from profile habits using length, sentence length, paragraphing, and punctuation rates.

Use copy metrics as guardrails. A model can score well on overlap by memorizing; that should count against it.

## Next Metrics To Add

For historical/literary/personality corpora, add these as the next layer:

- Function-word cosine similarity: compare rates of words like `and`, `but`, `of`, `to`, `that`, `which`, `shall`, `would`.
- Character n-gram similarity: captures punctuation, contractions, endings, capitalization, and phrase fragments.
- Most-frequent-word Delta or Cosine Delta: a standard stylometry baseline.
- Authorship classifier score: train a simple classifier to distinguish target-author passages from non-target passages, then score generated text as target/non-target.
- Embedding similarity to the prompt: checks content preservation without rewarding exact copying.
- Perplexity or grammar-check proxy: rough fluency check.

## Human Evaluation

For demos, use a small blind study:

- Show evaluators the prompt and 3-4 anonymous outputs: base model, RAG, LoRA, RAG+LoRA.
- Ask them to rate each from 1-5 on style recognizability, prompt faithfulness, fluency, and overall quality.
- Add a forced-choice question: "Which output sounds most like the target author?"
- Include a copy warning task for evaluators: "Does this look like it quotes the source rather than imitates style?"

## Progress Gates

Before training a new personality LoRA:

- Build a profile/query history from clean text.
- Run `export-lora --per-user`.
- Confirm profile examples are not mostly boilerplate, tables, links, or metadata.
- Confirm held-out query passages are excluded from profile examples.

After training:

- Compare base, RAG, LoRA, and RAG+LoRA on the same prompts.
- Style metrics should improve without a spike in `profile_copy_5gram_rate`.
- Human raters should prefer RAG+LoRA or LoRA over the base model for recognizability.
- Prompt faithfulness should not drop compared with the base model.
