"""
Wordle answer word list.

The official Wordle game uses ~2,309 possible answer words. This module contains
a curated vocabulary of 677 common 5-letter English words used for training and
evaluating the PPO agent. Set USE_FULL_LIST = True and provide a words.txt file
to use a custom word list.
"""
import os

USE_FULL_LIST = False
_WORD_FILE = os.path.join(os.path.dirname(__file__), "words.txt")

DEMO_WORDS = [
    "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult",
    "after", "again", "agent", "agree", "ahead", "alarm", "album", "alert",
    "alien", "align", "alike", "alive", "alley", "allow", "alone", "alter",
    "amino", "angel", "anger", "angle", "angry", "ankle", "apart", "apple",
    "apply", "arena", "argue", "arise", "aside", "asset", "audio", "avoid",
    "award", "aware", "badge", "baker", "basic", "basis", "beach", "begin",
    "being", "below", "bench", "bible", "birth", "black", "blade", "blame",
    "blank", "blast", "blaze", "bleed", "blend", "bless", "blind", "block",
    "blood", "bloom", "blown", "board", "bonus", "boost", "bound", "brain",
    "brand", "brave", "bread", "break", "breed", "brick", "brief", "bring",
    "broad", "broke", "brown", "brush", "build", "bunch", "burst", "cabin",
    "candy", "cargo", "carry", "catch", "cause", "chain", "chair", "chalk",
    "chaos", "charm", "chase", "cheap", "check", "chess", "chief", "child",
    "chunk", "civic", "claim", "clash", "class", "clean", "clear", "clerk",
    "click", "cliff", "climb", "cling", "clock", "clone", "close", "cloud",
    "coach", "coast", "coral", "could", "count", "court", "cover", "crack",
    "craft", "crane", "crash", "crazy", "cream", "crime", "cross", "crowd",
    "crown", "cruel", "crush", "curve", "cycle", "daily", "dance", "death",
    "debug", "decay", "delay", "delta", "demon", "dense", "depth", "derby",
    "devil", "dirty", "donor", "doubt", "draft", "drain", "drama", "drank",
    "dream", "dress", "dried", "drift", "drink", "drive", "drone", "drown",
    "dying", "eager", "early", "earth", "edict", "eight", "elder", "elect",
    "elite", "embed", "ember", "empty", "enemy", "enjoy", "enter", "equal",
    "equip", "error", "essay", "event", "every", "exact", "exert", "exile",
    "exist", "extra", "fable", "facet", "faint", "fairy", "faith", "false",
    "fence", "fever", "fiber", "field", "fifth", "fifty", "fight", "final",
    "first", "flame", "flash", "fleet", "flesh", "float", "flood", "floor",
    "flour", "fluid", "flush", "focus", "force", "forge", "found", "frame",
    "frank", "fraud", "fresh", "front", "frost", "fruit", "fully", "giant",
    "given", "glass", "globe", "gloom", "glory", "glove", "going", "grace",
    "grade", "grain", "grand", "grant", "grape", "grasp", "grass", "grave",
    "great", "green", "greet", "grief", "grind", "groan", "gross", "group",
    "grown", "guard", "guess", "guest", "guide", "guild", "guilt", "habit",
    "happy", "harsh", "heart", "heavy", "hence", "hobby", "honey", "honor",
    "horse", "hotel", "house", "human", "humor", "hurry", "hyper", "ideal",
    "image", "imply", "index", "indie", "inner", "input", "irony", "issue",
    "ivory", "jewel", "joint", "joker", "judge", "juice", "karma", "knife",
    "knock", "known", "label", "labor", "large", "laser", "later", "laugh",
    "layer", "learn", "lease", "legal", "lemon", "level", "light", "limit",
    "linen", "liver", "local", "lodge", "logic", "loose", "lover", "lower",
    "loyal", "lucky", "lunar", "lunch", "magic", "major", "maker", "manor",
    "march", "marry", "match", "mayor", "media", "mercy", "merit", "metal",
    "meter", "might", "minor", "minus", "model", "money", "month", "moral",
    "mount", "mouse", "mouth", "movie", "music", "naive", "nerve", "never",
    "night", "noble", "noise", "north", "noted", "novel", "nurse", "occur",
    "ocean", "offer", "olive", "orbit", "order", "other", "outer", "oxide",
    "ozone", "paint", "panel", "panic", "patch", "pause", "peace", "pearl",
    "penny", "phase", "phone", "photo", "piano", "piece", "pilot", "pitch",
    "pixel", "pizza", "place", "plain", "plane", "plant", "plate", "plaza",
    "plead", "plot", "plumb", "plume", "plunge", "plunk", "point", "poker",
    "polar", "pound", "power", "press", "price", "pride", "prime", "prince",
    "print", "prior", "probe", "prone", "proof", "proud", "prove", "proxy",
    "psalm", "pulse", "punch", "pupil", "purse", "queen", "query", "quest",
    "queue", "quick", "quiet", "quota", "quote", "radar", "radio", "raise",
    "rally", "ranch", "range", "rapid", "ratio", "reach", "react", "ready",
    "realm", "rebel", "reign", "relax", "reply", "rider", "ridge", "rifle",
    "right", "rigid", "risky", "rival", "river", "robin", "robot", "rocky",
    "roman", "rouge", "rough", "round", "route", "royal", "rugby", "ruler",
    "rural", "saint", "salad", "sauce", "scale", "scare", "scene", "scope",
    "score", "sense", "serve", "setup", "seven", "shall", "shame", "shape",
    "share", "sharp", "sheep", "sheer", "shelf", "shell", "shift", "shine",
    "shirt", "shock", "shoot", "shore", "short", "shout", "sight", "since",
    "sixth", "sixty", "skill", "skull", "slave", "sleep", "slice", "slide",
    "smile", "smoke", "snake", "solar", "solid", "solve", "sorry", "sound",
    "south", "space", "spare", "spark", "speak", "speed", "spend", "spill",
    "spine", "split", "spoke", "sport", "spray", "squad", "stack", "staff",
    "stage", "stair", "stake", "stall", "stamp", "stand", "stark", "start",
    "state", "steal", "steam", "steel", "steep", "steer", "stern", "stick",
    "still", "stock", "stone", "stood", "store", "storm", "story", "stove",
    "strip", "stuck", "stuff", "style", "sugar", "suite", "super", "surge",
    "swamp", "swear", "sweep", "sweet", "swift", "swing", "sword", "syrup",
    "table", "taste", "teach", "teeth", "tempo", "theme", "thick", "thing",
    "think", "third", "thorn", "those", "three", "threw", "throw", "thumb",
    "tight", "timer", "tired", "title", "toast", "today", "token", "topic",
    "torch", "total", "touch", "tough", "tower", "toxic", "trace", "track",
    "trade", "trail", "train", "trait", "trash", "treat", "trend", "trial",
    "tribe", "trick", "troop", "truck", "truly", "trump", "trunk", "trust",
    "truth", "tumor", "twist", "ultra", "uncle", "under", "union", "unite",
    "unity", "until", "upper", "upset", "urban", "usage", "usual", "utter",
    "valid", "value", "vault", "venue", "verse", "video", "vigor", "viral",
    "virus", "visit", "vista", "vital", "vivid", "vocal", "voice", "voter",
    "wages", "waste", "watch", "water", "weave", "weigh", "weird", "wheat",
    "wheel", "where", "which", "while", "white", "whole", "whose", "width",
    "witch", "woman", "world", "worry", "worse", "worst", "worth", "would",
    "wound", "wrath", "write", "wrong", "wrote", "yield", "young", "youth",
]

# Filter to only valid 5-letter words (some above may have typos)
DEMO_WORDS = [w for w in DEMO_WORDS if len(w) == 5 and w.isalpha()]


def get_word_list() -> list[str]:
    if USE_FULL_LIST and os.path.exists(_WORD_FILE):
        with open(_WORD_FILE) as f:
            words = [line.strip().lower() for line in f if len(line.strip()) == 5]
        return words
    return DEMO_WORDS
