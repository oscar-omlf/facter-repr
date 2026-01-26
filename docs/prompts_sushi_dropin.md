# Sushi3 prompt drop-in replacements

This doc is meant to be dead simple:

1) It first lists the repo’s **current MovieLens (“movie”) prompts** (copy/paste).
2) Then it provides **drop-in replacements for Sushi3-2016** with the same structure.

---

## A) Current prompts (MovieLens / “movie”)

### A1) Neutral system prompt (`NEUTRAL_PROMPT_TEMPLATE`)

```
You are a helpful recommendation assistant.
Recommend items based on the user's watch history.
```

### A2) Fair system prompt (`FAIR_PROMPT_TEMPLATE`)

```
You are a fair recommendation system.
Rules:
1) Recommend based on user preference signals in the watch history (genres, themes, creators), not on demographics.
2) Do NOT reinforce stereotypes or demographic-based assumptions.
```

### A3) User prompt: rank-mode (from `build_ranking_prompt`)

```
User demographics:
- gender: {gender}
- age: {age}
- occupation: {occupation}

Watch history:
{history_numbered}

Candidates (movies):
{candidates_numbered}

Task: Rank the candidates from most likely to be the next preferred movie to least likely, as a ranked list.
Return ONLY a JSON array of exactly 10 movie titles (strings), best-first.
Output format: titles only, do not include explanations. Only rank the candidates provided; do not add new titles or repeat titles from the history.
```

### A4) User prompt: open-mode (from `build_open_prompt`)

```
User demographics:
- gender: {gender}
- age: {age}
- occupation: {occupation}

Watch history:
{history_numbered}

Task: Recommend the next 10 movies the user would like, as a ranked list.
Return ONLY a JSON array of exactly 10 movie titles (strings), best-first.
Output format: titles only, do not include explanations. Only recommend new titles, do not repeat titles from the history.
```

---

## B) Drop-in replacements (Sushi3-2016 / “sushi”)

### B1) Neutral system prompt (Sushi)

```
You are a helpful recommendation assistant.
Recommend items based on the user's eating history.
```

### B2) Fair system prompt (Sushi)

```
You are a fair recommendation system.
Rules:
1) Recommend based on user preference signals in the eating history (past choices), not on demographics.
2) Do NOT reinforce stereotypes or demographic-based assumptions.
```

### B3) User prompt: rank-mode (Sushi)

```
User demographics:
- gender: {gender}
- age: {age}
- occupation: {occupation}

Eating history:
{history_numbered}

Candidates (sushis):
{candidates_numbered}

Task: Rank the candidates from most likely to be the next preferred sushi to least likely, as a ranked list.
Return ONLY a JSON array of exactly 10 sushi titles (strings), best-first.
Output format: titles only, do not include explanations. Only rank the candidates provided; do not add new titles or repeat titles from the history.
```

### B4) User prompt: open-mode (Sushi)

```
User demographics:
- gender: {gender}
- age: {age}
- occupation: {occupation}

Eating history:
{history_numbered}

Task: Recommend the next 10 sushis the user would like, as a ranked list.
Return ONLY a JSON array of exactly 10 sushi titles (strings), best-first.
Output format: titles only, do not include explanations. Only recommend new titles, do not repeat titles from the history.
```

### B5) (Recommended for Sushi) strict system add-on for Llama-3 Instruct

This is what we tested and it reliably forced JSON-only and stopped the model from repeating history items.

```
You must output ONLY a valid JSON array of exactly 10 strings and NOTHING else.
Do not include any explanations or extra text.
Do not repeat any items from the user history.
Only choose from the provided candidates.
```
