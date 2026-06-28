# List of characters that define a sentence or chunk boundary
PUNCTUATION_BOUNDARIES = [
    '।', '॥',  # Bengali/Hindi Dandas
    '.', '?', '!'  # Common English/Universal punctuation (hard sentence boundaries)
]

# The minimum number of words before sending a chunk to TTS
MIN_WORDS_PER_CHUNK = 12
