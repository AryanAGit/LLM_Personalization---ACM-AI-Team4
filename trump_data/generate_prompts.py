"""
Generate diverse prompts for each tweet in the dataset, for fine-tuning a model
on tweet style. Strategy:

1. Classify each tweet into a content category (announcement, attack, brag,
   media-shoutout, scheduling, hashtag-rally-cry, thread-continuation, etc.)
2. Each category has a pool of prompt-framing templates with varied verbs,
   moods, and structures (imperative, descriptive, contextual, scenario-based).
3. Rotate through templates within a category so no single phrasing dominates.
4. Pull concrete details (people mentioned, hashtags, time references, all-caps
   phrases) out of the tweet and weave them into the prompt so prompts are
   specific to their output, not generic.
"""

import json
import re
import random
from collections import defaultdict

random.seed(7)

INPUT = '/mnt/user-data/uploads/trump_tweets_cleaned.json'
OUTPUT = '/mnt/user-data/outputs/trump_tweets_with_prompts.json'

# ---------- Feature extraction ----------

def extract_features(text):
    """Pull structural/content signals from a tweet."""
    f = {}
    f['mentions'] = re.findall(r'@\s?\w+', text)
    f['hashtags'] = re.findall(r'#\s?\w+', text)
    f['has_url'] = bool(re.search(r'(pic\.twitter|t\.co|http)', text, re.I))
    f['all_caps_phrases'] = re.findall(r'\b[A-Z]{3,}(?:\s+[A-Z&]{2,})*\b', text)
    f['has_exclaim'] = '!' in text
    f['has_question'] = '?' in text
    f['ends_ellipsis'] = text.rstrip().endswith('...') or text.rstrip().endswith('..')
    f['starts_lowercase'] = text[:1].islower() if text else False
    f['length'] = len(text)
    f['word_count'] = len(text.split())
    return f


# ---------- Classification ----------
# Order matters: more specific categories first.

ATTACK_KEYWORDS = [
    'fake news', 'failing', 'sad!', 'crooked', 'dishonest', 'disgrace',
    'witch hunt', 'hoax', 'rigged', 'lying', 'phony', 'enemy of the people',
    'do nothing', 'obstruction', 'low iq', 'loser', 'pathetic', 'incompetent',
    'corrupt', 'weak', 'fool', 'rino', 'cheating',
]
PRAISE_KEYWORDS = [
    'great job', 'thank you', 'congratulations', 'wonderful', 'amazing',
    'incredible', 'fantastic', 'tremendous', 'honored', 'proud',
]
SCHEDULE_KEYWORDS = [
    'will be interviewed', 'will be on', 'tune in', 'tonight at',
    'a.m.', 'p.m.', 'will be speaking', 'will hold', 'will be holding',
    'tomorrow', 'this evening', 'live at', 'just landed',
]
ANNOUNCEMENT_KEYWORDS = [
    'i am pleased', 'today i', 'just signed', 'i have', 'i will be announcing',
    'i hereby', 'i am announcing',
]
ECONOMIC_KEYWORDS = [
    'jobs', 'economy', 'gdp', 'stock market', 'unemployment', 'tariff',
    'trade deal', 'china', 'jobs report', '401k', 'wages',
]
IMMIGRATION_KEYWORDS = [
    'border', 'wall', 'immigration', 'illegal', 'caravan', 'daca', 'maga',
    'sanctuary', 'ice ',
]
RALLY_CRY_PATTERNS = [
    'make america', 'maga', 'america first', '#maga', '#americafirst',
]


def classify(text, feats):
    t = text.lower()

    # Continuation of a previous tweet
    if feats['starts_lowercase'] and not text.startswith('@'):
        return 'continuation'
    if feats['ends_ellipsis'] and feats['word_count'] < 35:
        return 'thread_start'

    # Schedule / appearance announcements
    if any(k in t for k in SCHEDULE_KEYWORDS):
        return 'schedule'

    # Pure rally cry / slogan tweet
    if feats['word_count'] < 15 and any(p in t for p in RALLY_CRY_PATTERNS):
        return 'rally_cry'

    # Attack on opponents/media
    if any(k in t for k in ATTACK_KEYWORDS):
        if 'media' in t or 'news' in t or 'cnn' in t or 'nyt' in t or 'times' in t or 'post' in t:
            return 'media_attack'
        return 'political_attack'

    # Praise / thank-you
    if any(k in t for k in PRAISE_KEYWORDS):
        return 'praise'

    # Self-credit / brag about results
    if any(k in t for k in ECONOMIC_KEYWORDS) and ('record' in t or 'best' in t or 'highest' in t or 'lowest' in t or 'up ' in t):
        return 'economic_brag'

    # Topic-specific
    if any(k in t for k in IMMIGRATION_KEYWORDS):
        return 'immigration'
    if any(k in t for k in ECONOMIC_KEYWORDS):
        return 'economy'

    # Formal announcement
    if any(k in t for k in ANNOUNCEMENT_KEYWORDS):
        return 'announcement'

    # Question/rhetorical
    if feats['has_question']:
        return 'rhetorical'

    # Quote/retweet style
    if text.startswith('"') or text.startswith('\u201c'):
        return 'quote_share'

    # Catch-all: opinion/commentary
    return 'commentary'


# ---------- Topic snippet extraction ----------

def topic_snippet(text):
    """Extract a short noun phrase describing what the tweet is *about*,
    for use as filler in templates. Best-effort; prompts also have fallbacks."""
    # Strip URLs and pic links
    cleaned = re.sub(r'pic\.twitter\.com/\S+', '', text)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = cleaned.strip()
    # First clause, capped
    first = re.split(r'[.!?]', cleaned, maxsplit=1)[0].strip()
    if len(first) > 90:
        first = first[:90].rsplit(' ', 1)[0] + '...'
    return first or cleaned[:80]


def named_targets(text):
    """Pull names/handles that appear to be the subject."""
    handles = re.findall(r'@\s?(\w+)', text)
    # Capitalized name pairs (very rough)
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    return handles, names


# ---------- Templates ----------
# Each category has 6-12 templates. {topic}, {target}, {hashtag}, {caps}
# are filled when available, otherwise the template is skipped.

TEMPLATES = {
    'continuation': [
        "Continue the previous thought, completing the sentence and landing on a strong note.",
        "Finish the previous tweet's argument; the sentence picks up mid-clause.",
        "Pick up where the last tweet trailed off and bring the point home.",
        "Second half of a two-part post — complete the prior sentence and add the closing flourish.",
        "Conclude the multi-tweet statement that began moments ago.",
    ],
    'thread_start': [
        "Open a multi-part statement that will continue in a follow-up tweet — leave it on a cliffhanger.",
        "Begin a thought that's too long for one tweet; trail off so a second tweet can finish it.",
        "Set up the first half of a two-tweet message, ending mid-sentence.",
        "Start a thread on this — the first tweet should end with an ellipsis.",
    ],
    'schedule': [
        "Tell followers when to tune in for an upcoming TV appearance or event.",
        "Give a heads-up about an interview later today, including the time and channel.",
        "Announce arrival at a destination ahead of a scheduled event.",
        "Promote a forthcoming media appearance — short, punchy, includes time and outlet.",
        "Share a calendar item: where you'll be, when, and on what network.",
        "Ping followers about an event happening soon.",
    ],
    'rally_cry': [
        "A short, all-caps slogan tweet — pure campaign energy, almost no other content.",
        "Distill the entire political program into a hashtag-laden one-liner.",
        "A bumper-sticker-length tweet built around the campaign slogan.",
        "Closing flourish on a longer day — slogan, hashtags, exclamation points.",
    ],
    'political_attack': [
        "Hit back at political opponents who've been criticizing your agenda.",
        "Frame the opposition as obstructionist and out of touch with voters.",
        "A jab at Democrats over their handling of a recent issue.",
        "Mock an opponent's recent statement or position.",
        "Accuse the other side of a long-running pattern of bad behavior.",
        "Dismiss a critic with a sharp nickname-and-insult combo.",
        "Push back on accusations being leveled by political rivals.",
    ],
    'media_attack': [
        "Slam a major news outlet for what you see as biased or dishonest coverage.",
        "Dismiss a critical news story as fabricated.",
        "Call out a specific network or paper by name for unfair reporting.",
        "Frame negative press as proof the media is the real problem.",
        "Vent about how the press covered a recent event.",
        "Accuse a publication of inventing sources or quotes.",
    ],
    'praise': [
        "Send public thanks to supporters or allies after a positive moment.",
        "Congratulate someone on an achievement, in characteristically effusive terms.",
        "Shout out a person or group who's been loyal or done good work.",
        "A grateful note after a successful event.",
        "Heap praise on an ally who's been getting unfair treatment.",
    ],
    'economic_brag': [
        "Take credit for strong economic numbers that just came out.",
        "Point to a record-setting economic figure as proof the agenda is working.",
        "Cite a fresh jobs or markets stat and contrast it with the previous administration.",
        "Trumpet a new high in some economic indicator.",
        "Translate a dry economic data release into a victory lap.",
    ],
    'economy': [
        "Comment on a trade or economic development affecting American workers.",
        "Tie an economic issue back to the campaign promise of putting Americans first.",
        "Argue a tough trade stance is finally paying off.",
        "Frame an economic policy fight as protecting American jobs.",
    ],
    'immigration': [
        "Make the case for stronger borders in light of a recent event.",
        "Argue that current immigration laws are dangerously weak.",
        "Connect a news story to the broader push for a border wall.",
        "Demand action from Congress on immigration enforcement.",
        "Defend a controversial immigration move as necessary for safety.",
    ],
    'announcement': [
        "Formally announce a new appointment, signing, or executive action.",
        "Roll out a policy decision in a single tweet.",
        "Reveal a personnel pick with a brief endorsement.",
        "Mark the signing of a bill or order with a short statement.",
        "Break news about an action you've just taken.",
    ],
    'rhetorical': [
        "Pose a pointed question that answers itself, aimed at critics.",
        "Use a rhetorical question to highlight what you see as opponents' hypocrisy.",
        "Ask the obvious question nobody in the press is willing to ask.",
        "Frame the issue as a question whose answer should be self-evident.",
    ],
    'quote_share': [
        "Amplify a supportive quote from a TV segment or news story.",
        "Share a flattering line from a commentator, framing it as vindication.",
        "Echo a quote that backs up your position.",
    ],
    'commentary': [
        "React to a developing news story.",
        "Offer a take on something happening in the country today.",
        "Weigh in on a current event with a characteristic mix of opinion and exclamation.",
        "Stake out a position on a hot-button issue.",
        "Share a thought provoked by something in this morning's news.",
        "Comment on a recent development in your trademark style.",
        "An off-the-cuff observation about the state of things.",
        "Drop a quick opinion on the day's events.",
    ],
}

# Add detail-specific augmentations that get spliced into templates when
# concrete features exist. These help make prompts non-generic.

def augment(template, feats, text):
    additions = []
    if feats['hashtags']:
        # Use up to 2 hashtags as a hint
        hts = ', '.join(h.replace(' ', '') for h in feats['hashtags'][:2])
        additions.append(f"End with the hashtag(s): {hts}.")
    if feats['all_caps_phrases']:
        # Pick the most distinctive caps phrase (longest)
        caps = max(feats['all_caps_phrases'], key=len)
        if len(caps) > 4 and caps not in ('USA', 'CIA', 'FBI', 'NYT'):
            additions.append(f'Include the all-caps phrase "{caps}".')
    if feats['mentions']:
        handle = feats['mentions'][0].replace(' ', '')
        additions.append(f"Mention {handle} by handle.")
    if feats['has_url'] and 'pic.twitter' in text:
        additions.append("Attach a photo.")
    if feats['ends_ellipsis']:
        additions.append("End mid-sentence with an ellipsis (continued in a follow-up tweet).")

    if not additions:
        return template
    # Don't pile on — at most 2 augmentations
    random.shuffle(additions)
    return template + " " + " ".join(additions[:2])


# ---------- Main ----------

def make_prompt(text, category_counters):
    feats = extract_features(text)
    category = classify(text, feats)
    pool = TEMPLATES[category]
    # Rotate through the pool to maximize diversity within a category
    idx = category_counters[category] % len(pool)
    category_counters[category] += 1
    base = pool[idx]
    return augment(base, feats, text), category


def main():
    with open(INPUT) as f:
        data = json.load(f)

    category_counters = defaultdict(int)
    category_totals = defaultdict(int)
    out = []
    for entry in data:
        text = entry['output']
        prompt, cat = make_prompt(text, category_counters)
        out.append({'prompt': prompt, 'output': text})
        category_totals[cat] += 1

    import os
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} entries to {OUTPUT}")
    print("\nCategory distribution:")
    for cat, n in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {n:4d}  ({n/len(out)*100:.1f}%)")

    print("\n--- Sample of 12 random prompt/output pairs ---")
    for i in random.sample(range(len(out)), 12):
        print(f"\n[{i}]")
        print(f"  prompt: {out[i]['prompt']}")
        print(f"  output: {out[i]['output'][:140]}")


if __name__ == '__main__':
    main()
