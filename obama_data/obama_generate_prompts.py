"""
generate_prompts.py
-------------------
Generate a diverse prompt for each Obama speech transcript, for fine-tuning
a model on Obama's rhetorical style. Outputs a JSONL file where each line is:

    {"prompt": "...", "output": "..."}

Strategy:
1. Fix JSON corruption (unescaped quotes introduced by manual editing).
2. Classify each speech into one of ~15 content categories based on title
   keywords and transcript signals.
3. Each category has a pool of 6-10 prompt templates with varied framing:
   imperative, descriptive, contextual, role-based, occasion-based.
4. Extract concrete details from the transcript — location, named people,
   policy topics, signature phrases — and weave them into prompts so each
   prompt is specific to its speech, not generic.
5. Rotate through templates within a category for diversity.

Usage:
    python generate_prompts.py
    python generate_prompts.py --input obama_speeches_clean.json --output obama_speeches_prompts.jsonl
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import defaultdict

random.seed(42)

INPUT  = "obama_speeches_clean.json"
OUTPUT = "obama_speeches_prompts.jsonl"


# ── JSON repair ───────────────────────────────────────────────────────────────
# Manual editing introduced unescaped " inside transcript string values.
# This parser fixes them before handing off to json.loads().

def repair_json(raw: str) -> str:
    result = []
    i = 0
    n = len(raw)
    while i < n:
        marker = '"transcript": "'
        idx = raw.find(marker, i)
        if idx == -1:
            result.append(raw[i:])
            break
        value_start = idx + len(marker)
        result.append(raw[i:value_start])
        i = value_start
        value_chars = []
        while i < n:
            c = raw[i]
            if c == '\\':
                value_chars.append(c)
                i += 1
                if i < n:
                    value_chars.append(raw[i])
                    i += 1
            elif c == '"':
                rest = raw[i + 1:i + 20].lstrip()
                if rest.startswith('}') or rest.startswith(',\n') or rest.startswith('\n    }'):
                    result.append(''.join(value_chars))
                    result.append('"')
                    i += 1
                    break
                else:
                    value_chars.append('\\"')
                    i += 1
            else:
                value_chars.append(c)
                i += 1
    return ''.join(result)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(title: str, transcript: str) -> dict:
    """Pull structural and content signals for classification and augmentation."""
    f = {}
    t  = transcript
    tl = title.lower()
    tt = t.lower()

    f['title']       = title
    f['length']      = len(t)
    f['word_count']  = len(t.split())

    # Named people: sequences of Title-Cased words (rough)
    f['named_people'] = re.findall(
        r'\b(?:President|Secretary|Senator|Governor|Chancellor|Prime\s+Minister|'
        r'General|Ambassador|Reverend|Dr\.|Professor|Mayor|Chief\s+Justice|'
        r'Justice)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
        t
    )[:6]

    # Places mentioned in title or early transcript
    f['location']    = _extract_location(title, t)

    # Policy topics from title
    f['policy_topic'] = _extract_policy_topic(tl, tt)

    # Signature opening — first non-greeting sentence
    f['opening_line'] = _first_substantive_line(t)

    # Does transcript contain audience call-and-response?
    f['has_audience_interaction'] = bool(re.search(
        r'Audience[:\s]|crowd[:\s]|\[applause\]|\[laughter\]', t, re.I
    ))

    # Presence of anecdote / personal story markers
    f['has_anecdote'] = bool(re.search(
        r'\bI (met|remember|spoke with|received a letter|want to tell you about|'
        r'talked to|visited|heard from)\b', t, re.I
    ))

    # Foreign language greeting
    m = re.match(r'^([A-ZÀ-ÿa-z\s]+[!,])', t.strip())
    if m and len(m.group(1)) < 30 and not m.group(1).lower().startswith(
        ('thank', 'good', 'hello', 'please', 'my', 'i ', 'to ')
    ):
        f['foreign_greeting'] = m.group(1).strip()
    else:
        f['foreign_greeting'] = None

    # Scriptural / literary opening (quotes scripture or Lincoln etc.)
    f['has_literary_opening'] = bool(re.match(
        r'^\s*([""\u201c]|Unto them|In giving|Four score|Scripture)', t.strip()
    ))

    # Is this delivered abroad?
    abroad_markers = ['europe', 'berlin', 'germany', 'france', 'japan', 'china',
                      'israel', 'cairo', 'ghana', 'kenya', 'india', 'cuba',
                      'greece', 'athens', 'hiroshima', 'hannover', 'ireland',
                      'malaysia', 'vietnam', 'laos', 'myanmar', 'africa']
    f['delivered_abroad'] = any(m in tl or m in tt[:500] for m in abroad_markers)

    # Press conference signal
    f['is_press_conf'] = bool(re.search(
        r"press conference|take your questions|i'll take some questions|"
        r"open it up for questions", tt[:800], re.I
    ))

    return f


def _extract_location(title: str, transcript: str) -> str:
    """Best-effort location extraction from title or early transcript."""
    # Title often contains location after 'at', 'in', 'to the', etc.
    m = re.search(
        r'\b(?:at|in|to)\s+(?:the\s+)?([A-Z][A-Za-z\s]{3,40}?)(?:\s*[,\n]|$)',
        title
    )
    if m:
        return m.group(1).strip()
    # Common proper nouns in first 300 chars
    m = re.search(
        r'\b(White House|Congress|Senate|United Nations|Oval Office|'
        r'[A-Z][a-z]+ University|[A-Z][a-z]+ Center|[A-Z][a-z]+, [A-Z]{2})\b',
        transcript[:300]
    )
    if m:
        return m.group(1)
    return ""


def _extract_policy_topic(title_lower: str, transcript_lower: str) -> str:
    topics = [
        ('health care|affordable care|obamacare', 'health care reform'),
        ('economy|jobs|unemployment|recession', 'the economy and jobs'),
        ('climate|energy|clean energy|paris', 'climate change and clean energy'),
        ('immigration|border|daca|dreamers', 'immigration'),
        ('gun|shooting|violence|newtown|orlando|sandy hook', 'gun violence'),
        ('wall street|financial|bailout|deficit|budget', 'financial policy'),
        ('war in iraq|iraq|afghanistan|military|troops|veterans', 'military and veterans'),
        ('iran|nuclear|sanctions', 'Iran and nuclear policy'),
        ('terrorism|terror|al qaeda|isis|isil', 'counterterrorism'),
        ('race|civil rights|voting rights|equality|justice', 'civil rights and equality'),
        ('education|college|student|school', 'education'),
        ('trade|tariff|tpp|nafta', 'trade policy'),
        ('foreign policy|diplomacy|nato|allies', 'foreign policy'),
    ]
    for pattern, label in topics:
        if re.search(pattern, title_lower) or re.search(pattern, transcript_lower[:600]):
            return label
    return ""


def _first_substantive_line(transcript: str) -> str:
    """Return first sentence that isn't a greeting or acknowledgment."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', transcript.replace('\n', ' '))
    skip = re.compile(
        r'^(thank you|good (morning|afternoon|evening|day)|please be seated|'
        r'hello|well,|it is (wonderful|great|good|an honor)|i want to thank|'
        r'let me begin|let me start|i\'d like to thank)',
        re.I
    )
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) >= 8 and not skip.match(sent):
            return sent[:120]
    return ""


# ── Classification ────────────────────────────────────────────────────────────

def classify(title: str, feats: dict) -> str:
    tl = title.lower()
    tt_start = feats['opening_line'].lower()

    # Ordered most-specific first
    if 'commencement' in tl or 'graduation' in tl:
        return 'commencement'
    if 'eulogy' in tl or 'memorial service' in tl:
        return 'eulogy'
    if 'farewell' in tl or 'final presidential' in tl:
        return 'farewell'
    if 'inaugural' in tl:
        return 'inaugural'
    if 'state of the union' in tl or 'sotu' in tl:
        return 'sotu'
    if 'prayer breakfast' in tl:
        return 'prayer_breakfast'
    if 'correspondents dinner' in tl or 'gridiron' in tl:
        return 'humor'
    if 'weekly address' in tl:
        return 'weekly_address'
    if 'press conference' in tl or 'press briefing' in tl or feats['is_press_conf']:
        return 'press_conference'
    if any(k in tl for k in ['eulogy', 'memorial', 'tribute', 'remembrance', 'in honor of']):
        return 'eulogy'
    if any(k in tl for k in ['medal of honor', 'presidential medal', 'award ceremony']):
        return 'award_ceremony'
    if any(k in tl for k in ['shooting', 'tragedy', 'attack', 'terror', 'disaster',
                               'hurricane', 'tornado', 'flooding', 'massacre', 'bombing']):
        return 'crisis_statement'
    if feats['delivered_abroad'] or any(k in tl for k in ['address to the people', 'address in ', 'speech in ']):
        return 'international_address'
    if any(k in tl for k in ['united nations', 'nato', 'g8', 'g20', 'asean', 'security council']):
        return 'international_address'
    if any(k in tl for k in ['announcement', 'nominat', 'appoint', 'signing', 'executive order']):
        return 'announcement'
    if any(k in tl for k in ['rally', 'campaign', 'victory', 'election night', 'dnc', 'democratic national']):
        return 'campaign'
    if any(k in tl for k in ['commencement', 'university', 'college', 'school']):
        return 'commencement'
    if feats['policy_topic']:
        return 'policy_speech'
    return 'general_address'


# ── Templates ─────────────────────────────────────────────────────────────────
# Each category has 7-12 templates. Rotate through them per category.
# {topic}, {location}, {person}, {opening} are filled when available.

TEMPLATES: dict[str, list[str]] = {

    'commencement': [
        "Write a commencement address to a graduating class, offering a vision of civic responsibility and the obligations of an educated person to the world.",
        "Deliver a graduation speech that weaves personal anecdote with a call to meet the challenges of the moment — economic, social, or political.",
        "Address a university graduating class, drawing on historical precedent to frame what public service demands of this generation.",
        "Write an inspiring commencement speech that uses the graduates' youth as a symbol of national possibility.",
        "Open a commencement address with characteristic humility, acknowledge the faculty and dignitaries, then pivot to a serious argument about what the country needs from its new graduates.",
        "Give a graduation speech that uses a recurring rhetorical motif — a phrase, an image, or a historical figure — to anchor its central argument.",
        "Write a commencement address in which the speaker connects his own improbable biography to the wider American story before challenging the class to write the next chapter.",
        "Deliver a college graduation speech that mixes warmth and humor in its opening, then turns earnest as it makes the case for a life of service over a life of comfort.",
        "Address a graduating class at a moment of national uncertainty, arguing that difficulty is not a reason for despair but a call to purpose.",
    ],

    'eulogy': [
        "Write a eulogy for a distinguished public servant, honoring both the arc of their career and the private qualities that defined their character.",
        "Deliver a funeral tribute that opens by addressing the grieving family directly, then moves outward to what the deceased meant to the nation.",
        "Write a eulogy that uses a specific memory or story to illuminate the larger character of the person being remembered.",
        "Give remarks at a memorial service that balance grief with gratitude, and close with an image that transforms loss into enduring legacy.",
        "Write a tribute speech for a lion of American public life — someone who fought hard battles and won many of them — situating their work in the sweep of history.",
        "Deliver a eulogy that is at once a personal goodbye and a public argument for why the cause this person championed must continue.",
        "Write a memorial address that quotes the deceased, describes a formative encounter with them, and closes with a benediction.",
        "Offer remarks at a memorial that are honest about complexity and loss before landing on a note of hard-won hope.",
    ],

    'farewell': [
        "Write a farewell address to the American people, reflecting on the work of eight years and the condition of democracy at the moment of departure.",
        "Deliver a closing speech to the nation that is equal parts retrospective and warning — a frank accounting of what was accomplished and what remains unfinished.",
        "Write a farewell that returns to the values articulated at the beginning of a presidency and measures how well the country has lived up to them.",
        "Close out the presidency with a speech to a home crowd that is personal in tone but civic in argument.",
        "Write a valedictory address that thanks the American people without being maudlin, and that ends with a clear-eyed statement of what democracy requires of its citizens.",
        "Deliver a final presidential address that names the threats to self-government plainly and asks the audience to take personal responsibility for meeting them.",
    ],

    'inaugural': [
        "Write a presidential inaugural address that opens with a statement of national purpose and closes with a call to shared sacrifice.",
        "Deliver an inauguration speech that places this moment in the long arc of American history, from the founders through the civil rights movement.",
        "Write an inaugural address that is deliberately spare — no oratorical flourish for its own sake — but builds to an unambiguous statement of what this presidency will stand for.",
        "Open a presidential term with a speech that explicitly acknowledges the crisis the country faces and frames the administration's work as a response to that crisis.",
        "Write an inaugural address that reaches across party lines while being clear about the governing philosophy that will guide the next four years.",
    ],

    'sotu': [
        "Write a State of the Union address that opens with an economic narrative, moves through domestic policy priorities, and closes with an optimistic appeal to national character.",
        "Deliver an annual address to Congress that uses specific Americans in the gallery to humanize abstract policy debates.",
        "Write a State of the Union that frames each policy agenda item as a response to a specific challenge the country is facing, rather than a list of proposals.",
        "Address Congress with a speech that is combative about obstruction but ultimately hopeful in its vision for what bipartisan action could achieve.",
        "Write a State of the Union that opens with the state of the economy, pivots to foreign policy, and closes with a passage about the American character that ties everything together.",
        "Write a State of the Union that deliberately avoids a long laundry list of proposals in favor of a tighter, more thematic argument about one or two defining national challenges.",
    ],

    'prayer_breakfast': [
        "Write a speech to the National Prayer Breakfast that takes faith seriously as a civic force while insisting on the value of doubt and humility.",
        "Address a gathering of religious leaders with remarks that are personally reverent but intellectually honest — acknowledging the ways religion has been used both for good and ill in history.",
        "Write a prayer breakfast speech that uses a personal moment of spiritual reckoning to ground a broader argument about leadership and service.",
        "Deliver remarks at a faith gathering that draw on scripture and on American religious history, then turn to the policy demands that faith makes on a just society.",
    ],

    'humor': [
        "Write a White House Correspondents Dinner speech — self-deprecating, sharp about the press, full of inside-Washington jokes, and structurally built around a running gag.",
        "Deliver remarks at a formal press dinner that opens with several well-timed jokes at the speaker's own expense before landing on a sincere closing note.",
        "Write a comedic after-dinner speech that mimics the cadence of a serious presidential address but keeps undercutting it with punchlines.",
        "Give a Correspondents Dinner speech in which teleprompters are themselves part of the joke, and the prepared text keeps colliding with improvised asides.",
        "Write remarks for a black-tie press event that joke about the speaker's age, the media's coverage, and the absurdity of the moment — then close with a genuine word about the importance of a free press.",
        "Write a Correspondents Dinner speech structured around a series of mock headlines, each more absurd than the last.",
    ],

    'weekly_address': [
        "Write a presidential weekly radio and internet address focused on a single domestic policy accomplishment or legislative priority.",
        "Deliver a short weekly address to the American people that opens with a concrete statistic, uses it to frame a policy argument, and closes with a direct ask.",
        "Write a weekly address that responds to something that happened in Congress or in the economy this week, explaining clearly what it means for ordinary families.",
        "Give a brief weekly presidential address on a national holiday or commemorative occasion, connecting the day's meaning to a current policy priority.",
        "Write a weekly address that is deliberately conversational — no big rhetoric, just a direct explanation of what the administration did this week and why.",
        "Write a Saturday presidential address that opens with a word of thanks to service members or first responders and then pivots to the policy topic of the week.",
    ],

    'press_conference': [
        "Write an opening statement for a presidential press conference that frames the economic situation, announces two or three actions being taken, and then invites questions.",
        "Open a press briefing with a statement about a developing international situation, laying out what is known, what actions have been ordered, and what the administration is watching.",
        "Write the opening remarks of a press conference held immediately after a major piece of legislation passed or failed, explaining what it means and what comes next.",
        "Write a press conference opening statement that addresses multiple issues at once — a domestic crisis, a foreign policy development, and a congressional standoff.",
        "Deliver opening remarks at a year-end press conference that takes stock of the past twelve months before opening the floor to reporters.",
        "Write a press conference statement in which the speaker responds to criticism, defends a decision, and frames the path forward — without being defensive in tone.",
        "Open a presidential press briefing on a crisis with a carefully measured statement that projects calm authority, describes the government's response, and asks for patience from the public.",
    ],

    'award_ceremony': [
        "Write remarks presenting a Medal of Honor, describing in specific detail the act of heroism being recognized and placing it in the tradition of American military valor.",
        "Deliver a White House ceremony speech that honors a soldier's bravery, addresses their family directly, and uses the occasion to reflect on the debt the nation owes its service members.",
        "Write remarks at an award ceremony that open with self-deprecating humor about the gravity of the occasion, then shift into a moving account of the honoree's character and deeds.",
        "Give a Presidential Medal of Freedom citation that explains why this person's life's work has enlarged what it means to be American.",
        "Write award ceremony remarks that balance military protocol with human warmth — honoring the institution and the individual in equal measure.",
        "Write a commendation speech in which the honoree's specific act of courage is described in the plain language of official military citation, then translated into what it means to the people who were there.",
    ],

    'crisis_statement': [
        "Write a presidential statement on a mass shooting — grieving without melodrama, defending the right to grieve, and making a measured but direct argument for action.",
        "Deliver a statement on a terrorist attack that is factual about what is known, honest about what is not, and clear about the government's immediate response — without inflaming fear.",
        "Write an Oval Office address in response to a domestic disaster, expressing solidarity with the affected community and detailing the federal response already underway.",
        "Write a statement on a national tragedy that deliberately avoids political language in its first half before introducing, carefully, the policy question the tragedy makes unavoidable.",
        "Deliver a short crisis statement that is primarily logistical — here is what happened, here is what we are doing — before closing with a passage that addresses the emotional weight of the moment.",
        "Write a presidential address to the nation following a week of violence, attempting to speak to grief and anger without choosing sides, and making the case for a kind of national self-examination.",
        "Write a statement on a foreign-policy crisis — a downed aircraft, a coup, a breakdown in negotiations — that is deliberate and precise, and that signals firmness without escalating rhetoric.",
    ],

    'international_address': [
        "Write a speech delivered abroad to a foreign audience, opening in their language, honoring the host nation's history, and making the case for the American alliance.",
        "Deliver an address at a foreign venue that places the U.S.-host nation relationship in historical context before pivoting to the present-day agenda.",
        "Write a speech to a European audience that acknowledges the strains in the transatlantic relationship honestly before making the argument for why the alliance matters more than ever.",
        "Write a speech in a nation with a complicated history with the United States that leads with acknowledgment and humility before moving to shared interests.",
        "Write an address at the United Nations General Assembly that lays out the American position on the defining international challenge of the moment.",
        "Deliver remarks to a foreign parliament or gathered public that balance American interests with genuine respect for the host country's sovereignty and history.",
        "Write an international address that opens with a striking historical observation about the setting — a divided city, a former battlefield, a new democracy — and uses it to frame the speech's central argument.",
        "Write a speech delivered to the youth of another country, appealing to shared values while being honest about the distance between ideals and current realities.",
    ],

    'announcement': [
        "Write a brief statement announcing a new cabinet appointment or senior nomination, praising the appointee's qualifications and explaining what the role requires at this moment.",
        "Announce the signing of a piece of major legislation with remarks that credit the long legislative effort, name the key players, and explain clearly what the law will do for ordinary people.",
        "Write an announcement statement for a new executive action, framing it as a necessary response to congressional inaction and explaining the legal basis for proceeding.",
        "Make a short, formal announcement of a diplomatic agreement, describing what was agreed, what each side gave up, and why the deal is worth it.",
        "Write remarks marking a significant military or intelligence achievement, balancing the need to inform the public with the need to protect ongoing operations.",
        "Announce a new domestic policy initiative with a speech that opens with a story, pivots to the scale of the problem, introduces the policy response, and closes with a challenge to Congress.",
    ],

    'campaign': [
        "Write a campaign rally speech that opens with the crowd's energy, grounds itself in the economic anxieties of the audience, and closes with an argument about what is at stake in this election.",
        "Deliver an election-night speech — either a victory address or a concession — that is gracious in tone and clear in its statement of what the outcome means.",
        "Write a stump speech for a battleground state that uses local economic details to frame the national argument, and closes with the core campaign contrast.",
        "Write a campaign speech to a traditionally skeptical audience, acknowledging the disagreements directly before making the case for why the speaker's coalition is broad enough to govern.",
        "Write a convention speech that is more argument than autobiography — using personal history only as prologue to a serious case for a policy agenda.",
        "Deliver a closing-argument campaign speech that names the stakes of the election plainly, appeals to unity without papering over disagreement, and ends on an image of what winning would make possible.",
    ],

    'policy_speech': [
        "Write a major policy address on {topic} that opens with the human stakes of the issue, presents the administration's position, anticipates the main objections, and answers them.",
        "Deliver a speech on {topic} at a relevant institution — a university, a think tank, a factory floor — grounding the policy argument in that specific setting.",
        "Write a policy speech on {topic} that uses a historical analogy to frame what the administration is trying to do, and explains why the current moment is an inflection point.",
        "Make the case for a specific piece of legislation on {topic}, explaining in plain language what the bill does, who it helps, and why Congress should pass it now.",
        "Write a speech on {topic} that is explicitly addressed to skeptics — acknowledging the strongest arguments against the administration's position before responding to them.",
        "Deliver a policy address on {topic} that opens with a concrete story about an individual or family affected by the status quo, then scales up to the systemic argument.",
        "Write a speech making the economic case for action on {topic}, arguing that inaction has its own costs and that the moment for incremental thinking has passed.",
    ],

    'general_address': [
        "Write a speech that opens with a direct acknowledgment of the occasion, establishes a historical frame for the present moment, and closes with a challenge to the audience.",
        "Deliver an address that uses a personal anecdote as its entry point, then zooms out to a wider argument about national character or shared responsibility.",
        "Write a speech that is structured around a single repeated phrase or rhetorical motif, building to a crescendo before a quiet, deliberate close.",
        "Give an address that is honest about what divides the country, makes the case for why division is a choice not a fate, and ends with a concrete image of unity.",
        "Write a speech that moves between the personal and the political, using moments of intimacy and moments of sweep to make its argument.",
        "Write an address in which the speaker is clearly moved by the occasion — the words more carefully chosen, the pauses longer, the imagery more deliberate — and that earns its emotion rather than asserting it.",
        "Deliver a speech at a civic institution that honors the institution's history while making an argument about what it needs to become.",
        "Write a speech that quotes Lincoln, or King, or the founders — but earns the quote by first establishing why this moment demands that kind of witness.",
    ],
}


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(template: str, feats: dict, category: str) -> str:
    """Fill template placeholders and optionally append one concrete detail."""
    # Fill {topic} placeholder
    if '{topic}' in template:
        topic = feats['policy_topic'] or 'the central policy challenge of the moment'
        template = template.replace('{topic}', topic)

    additions = []

    # Location detail
    if feats['location'] and category in ('international_address', 'commencement', 'policy_speech', 'general_address'):
        additions.append(f"The speech is delivered at {feats['location']}.")

    # Foreign greeting
    if feats['foreign_greeting']:
        additions.append(f"Open with the greeting \"{feats['foreign_greeting']}\" before shifting to English.")

    # Literary / scriptural opening
    if feats['has_literary_opening'] and category not in ('humor',):
        additions.append("Open with a literary or scriptural quotation that frames the speech's central theme.")

    # Anecdote signal
    if feats['has_anecdote'] and category in ('eulogy', 'commencement', 'crisis_statement', 'policy_speech', 'general_address'):
        additions.append("Include a specific personal encounter or story to ground the speech's argument.")

    # Named person
    if feats['named_people'] and category in ('eulogy', 'award_ceremony', 'announcement', 'policy_speech'):
        person = feats['named_people'][0]
        additions.append(f"Refer to {person} by name and role.")

    # Audience interaction
    if feats['has_audience_interaction'] and category in ('campaign', 'commencement', 'sotu'):
        additions.append("Write with a live audience in mind — the speech should breathe, with room for applause and call-and-response.")

    if not additions:
        return template

    # Cap at one augmentation — keep prompts clean
    random.shuffle(additions)
    return template + " " + additions[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def make_prompt(title: str, transcript: str, category_counters: dict) -> tuple[str, str]:
    feats    = extract_features(title, transcript)
    category = classify(title, feats)
    pool     = TEMPLATES[category]
    idx      = category_counters[category] % len(pool)
    category_counters[category] += 1
    base     = pool[idx]
    prompt   = augment(base, feats, category)
    return prompt, category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=INPUT)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()

    raw  = Path(args.input).read_text(encoding='utf-8')
    data = json.loads(repair_json(raw))
    print(f"Loaded {len(data)} speeches from {args.input}")

    category_counters = defaultdict(int)
    category_totals   = defaultdict(int)
    out = []

    for speech in data:
        title      = speech.get('title', '').strip()
        transcript = speech.get('transcript', '').strip()
        if not transcript:
            continue
        prompt, cat = make_prompt(title, transcript, category_counters)
        out.append({'prompt': prompt, 'response': transcript})
        category_totals[cat] += 1

    Path(args.output).write_text(
        '\n'.join(json.dumps(entry, ensure_ascii=False) for entry in out),
        encoding='utf-8'
    )
    print(f"Wrote {len(out)} entries to {args.output}")

    print("\nCategory distribution:")
    for cat, n in sorted(category_totals.items(), key=lambda x: -x[1]):
        pct = n / len(out) * 100
        print(f"  {cat:25s} {n:4d}  ({pct:.1f}%)")

    print("\n--- 12 sample prompt/output pairs ---")
    samples = random.sample(out, min(12, len(out)))
    for entry in samples:
        print(f"\n  prompt: {entry['prompt']}")
        print(f"  output: {entry['output'][:120]}...")


if __name__ == '__main__':
    main()