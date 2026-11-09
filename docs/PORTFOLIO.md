# Portfolio notes

An audit of how AI assistants represent a title, with intervals instead of single numbers and a guard against accidental API spend.

## Bullets you can defend today (each is backed by the test suite or a screenshot in this repo)
- Built an AI-answer audit: repeated sampling of prompts, answer caching, Wilson-interval mention rates, rank and share of voice in genre lists, and release-date / cast accuracy against entered ground truth.
- Implemented Anthropic, Gemini, OpenAI, Ollama, and vLLM adapters (stdlib only) with retry/backoff and a spend guard that prints the call count and requires explicit confirmation; verified over real local HTTP sockets.
- Shipped an answer-inspector dashboard with per-answer verdicts, interval whiskers, and automated audit history timeline; 15 automated tests including a real-browser check.

## Add these only after you have measured them yourself (do not publish a number you have not measured)
- Ran [N] samples per prompt on [M] titles with [model]; release date correct [x]% ([lo]-[hi]%); mention rate in genre lists [y]%.
- Most common factual error found: [describe].

## Interview questions this project prepares you for
**Why sample each prompt several times?**  
Model answers vary run to run. One answer is an anecdote; N answers give a rate, and the Wilson interval says how much to trust it.

**How do you detect fabricated cast?**  
Roughly: capitalised name-like strings not in the ground truth. It is a heuristic with false positives, labelled as such in the UI.

**How do you stop a script from burning API budget?**  
`run` computes and prints the call count and exits unless `--yes` is passed; answers are cached by (title, model, prompt, sample).
