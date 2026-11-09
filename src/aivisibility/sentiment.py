"""Small transparent sentiment lexicon with negation handling (used to score opinion answers)."""
import re


POS = set("love loved loving great amazing awesome incredible brilliant beautiful stunning perfect masterpiece fantastic excellent best good hyped excited epic fun hilarious funny gorgeous wow phenomenal superb impressive outstanding favorite favourite cool nice banger goosebumps worth".split())


NEG = set("hate hated terrible awful horrible bad worst boring cringe trash garbage disappointing disappointed overrated mess bland lazy ruined ruin flop waste dull weak predictable forgettable mid meh annoying pathetic greedy scam ripoff overpriced expensive boycott insulting sellout".split())


NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't", "didn't", "cannot", "can't", "won't", "hardly", "aren't", "wouldn't"}


INTENS = {"very", "really", "so", "absolutely", "super", "extremely", "incredibly"}


EMOJI_POS, EMOJI_NEG = set("😍❤🔥👏😂🤩👍"), set("😡🤮👎😴😒")


TOKEN = re.compile(r"[a-z']+|[\U0001F300-\U0001FAFF\u2764\u2600-\u27BF]")


def tokens(text):
    return TOKEN.findall(text.lower().replace("’", "'"))


def lexicon_score(text):
    toks, s = tokens(text), 0.0
    for i, t in enumerate(toks):
        v = 1.0 if (t in POS or t in EMOJI_POS) else -1.0 if (t in NEG or t in EMOJI_NEG) else 0.0
        if not v:
            continue
        window = toks[max(0, i - 3):i]
        if any(w in NEGATORS or w.endswith("n't") for w in window):
            v = -v
        if any(w in INTENS for w in window):
            v *= 1.5
        s += v
    return s
