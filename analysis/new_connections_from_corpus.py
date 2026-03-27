"""Find new people and places mentioned in newly-added corpus texts that aren't in the existing gazetteer data."""
import re
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

# Load existing person data
existing_persons = set()
pi_path = ROOT / "gazetteer/person_info.json"
if pi_path.exists():
    with open(pi_path) as f:
        persons = json.load(f)
        for p in persons:
            name = p if isinstance(p, str) else p.get("name", "")
            existing_persons.add(name.lower().strip())

# Load existing correspondence graph names
corr_path = ROOT / "gazetteer/unified_correspondence_graph.csv"
if corr_path.exists():
    with open(corr_path) as f:
        for row in csv.DictReader(f):
            existing_persons.add(row.get("person_a", "").lower().strip())
            existing_persons.add(row.get("person_b", "").lower().strip())

# Load existing venues
existing_venues = set()
v_path = ROOT / "gazetteer/venues.csv"
if v_path.exists():
    with open(v_path) as f:
        for row in csv.DictReader(f):
            existing_venues.add(row.get("name", "").lower().strip())

# Load existing booksellers
bs_path = ROOT / "gazetteer/booksellers.csv"
if bs_path.exists():
    with open(bs_path) as f:
        for row in csv.DictReader(f):
            existing_persons.add(row.get("name", "").lower().strip())

print(f"Existing data: {len(existing_persons)} persons, {len(existing_venues)} venues\n")

def strip_gutenberg(text):
    for m in ["*** START OF", "***START OF"]:
        idx = text.find(m)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl+1:]
            break
    for m in ["*** END OF", "***END OF", "End of the Project Gutenberg"]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text

def load(path):
    return strip_gutenberg(path.read_text(errors="replace")).replace("ſ", "s")

# ============================================================
# 1. NAMED PERSONS in new texts — who is mentioned?
# ============================================================
print("=" * 100)
print("1. PEOPLE MENTIONED IN NEW CORPUS TEXTS")
print("=" * 100)

# Key figures to search for across the new corpus
# These are people who might appear as references/connections
# but aren't necessarily in the gazetteer yet
search_persons = {
    # Publishers not yet tracked
    "joseph johnson": "Publisher (Wollstonecraft, Priestley, Godwin, Paine)",
    "john murray": "Publisher",
    "longman": "Publisher dynasty",
    "lackington": "Bookseller (Temple of the Muses)",
    "hookham": "Bookseller/circulating library",

    # Medical figures
    "sydenham": "Physician (English Hippocrates)",
    "cheyne": "Physician (English Malady)",
    "willis": "Physician (neuroanatomist)",
    "cullen": "Physician (nosologist)",
    "monro": "Physician (Bethlem)",
    "battie": "Physician (madness)",
    "crichton": "Physician (attention)",
    "whytt": "Physician (nervous disorders)",
    "brown": "Physician (Brunonian medicine)",
    "hartley": "Philosopher-physician (associationism)",

    # Philosophers not in network
    "reid": "Philosopher (common sense)",
    "hutcheson": "Philosopher (moral sense)",
    "condillac": "Philosopher (sensationism)",
    "bentham": "Philosopher (utilitarianism)",
    "stewart": "Philosopher (Scottish school)",
    "paley": "Philosopher/theologian (natural theology)",
    "mandeville": "Philosopher (Fable of the Bees)",
    "shaftesbury": "Philosopher (politeness)",

    # Bluestockings / women writers
    "barbauld": "Poet, essayist",
    "seward": "Poet (Swan of Lichfield)",
    "macaulay": "Historian, political writer",
    "trimmer": "Educational writer",
    "wakefield": "Writer on female education",
    "hays": "Feminist writer",
    "inchbald": "Novelist, dramatist",
    "robinson": "Poet, novelist (Perdita)",
    "charlotte smith": "Novelist, poet",
    "lennox": "Novelist",

    # Key venue-associated people
    "garrick": "Actor-manager (Drury Lane)",
    "siddons": "Actress",
    "kemble": "Actor-manager",
    "sheridan": "Dramatist (Drury Lane manager)",

    # Scientists
    "priestley": "Chemist, theologian",
    "davy": "Chemist",
    "dalton": "Chemist",
    "banks": "Naturalist (Royal Society president)",
    "hunter": "Surgeon",
    "jenner": "Physician (vaccination)",

    # Politicians/public figures
    "wilkes": "Politician (liberty)",
    "fox": "Politician (Whig leader)",
    "pitt": "Politician (PM)",
    "north": "Politician (PM)",
    "rockingham": "Politician (PM)",
    "shelburne": "Politician (PM)",
}

# Search in newly added texts only
new_text_dirs = [
    ROOT / "nonfiction/philosophy",
    ROOT / "nonfiction/science",
    ROOT / "nonfiction/history",
    ROOT / "nonfiction/misc",
    ROOT / "nonfiction/poetry",
    ROOT / "LaurenceSterne",
    ROOT / "OliverGoldsmith",
    ROOT / "ElizabethInchbald",
    ROOT / "WilliamGodwin",
]

person_mentions = defaultdict(lambda: Counter())
for d in new_text_dirs:
    if not d.exists():
        continue
    for f in d.rglob("*.txt"):
        text = load(f).lower()
        total_words = len(re.findall(r'[a-z]+', text))
        if total_words < 2000:
            continue
        for person, role in search_persons.items():
            count = len(re.findall(r'\b' + re.escape(person) + r'\b', text))
            if count > 0:
                person_mentions[person][f.stem] = count

# Show people with significant cross-text presence
print("\nPeople mentioned across multiple new texts (potential network additions):")
print(f"{'Person':<25} {'Role':<45} {'Texts':>5} {'Total':>6}")
print("-" * 85)
for person, files in sorted(person_mentions.items(), key=lambda x: -sum(x[1].values())):
    total = sum(files.values())
    n_texts = len(files)
    if n_texts >= 2 or total >= 10:
        role = search_persons[person]
        top_texts = sorted(files.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{t}({c})" for t, c in top_texts)
        print(f"  {person:<25} {role:<45} {n_texts:>3}  {total:>5}  [{top_str}]")

# ============================================================
# 2. LONDON LOCATIONS mentioned in new texts
# ============================================================
print("\n\n" + "=" * 100)
print("2. LONDON LOCATIONS IN NEW TEXTS — potential venue additions")
print("=" * 100)

london_places = [
    # Venues not yet in gazetteer
    "temple of the muses", "lackington", "royal institution",
    "british museum", "somerset house", "burlington house",
    "freemasons' hall", "freemasons hall",
    "almack's", "almacks", "willis's rooms", "willis rooms",
    "exeter exchange", "exeter change",
    "king's bench", "fleet prison", "marshalsea",
    "foundling hospital", "magdalen hospital",
    "asylum", "bethlem", "bedlam", "bridewell",
    "guy's hospital", "st thomas's hospital", "bartholomew's hospital",
    "royal society", "linnean society", "antiquarian society",
    "the spectator", "will's coffee", "button's coffee",
    "child's coffee", "lloyd's coffee", "jonathan's coffee",
    "chapter coffee", "bedford coffee", "grecian coffee",
    "st james's coffee", "tom's coffee", "garraway's",
    "westminster hall", "guildhall", "mansion house",
    "bow street", "old bailey",
    "grub street", "moorfields",

    # Streets/areas
    "paternoster row", "ave maria lane", "stationers' hall",
    "st paul's churchyard", "ludgate hill",
    "chancery lane", "lincoln's inn", "gray's inn",
    "temple", "inner temple", "middle temple",
    "covent garden", "drury lane", "haymarket",
    "leicester square", "soho square", "golden square",
    "bloomsbury", "russell square", "bedford square",
    "grosvenor square", "hanover square", "cavendish square",
    "mayfair", "marylebone", "paddington",
    "islington", "hackney", "hampstead",
    "greenwich", "woolwich", "deptford",
    "southwark", "lambeth", "bermondsey",
    "wapping", "limehouse", "stepney", "whitechapel",
    "spitalfields", "shoreditch", "bethnal green",
]

place_mentions = defaultdict(lambda: Counter())
for d in new_text_dirs:
    if not d.exists():
        continue
    for f in d.rglob("*.txt"):
        text = load(f).lower()
        if len(text) < 5000:
            continue
        for place in london_places:
            count = len(re.findall(r'\b' + re.escape(place) + r'\b', text))
            if count > 0:
                place_mentions[place][f.stem] = count

print("\nLondon locations mentioned in new texts:")
print(f"{'Location':<30} {'Texts':>5} {'Total':>6}  {'Already in gazetteer?':>22}")
print("-" * 70)
for place, files in sorted(place_mentions.items(), key=lambda x: -sum(x[1].values())):
    total = sum(files.values())
    n_texts = len(files)
    if total >= 3:
        in_gaz = "YES" if place in existing_venues else "no"
        print(f"  {place:<30} {n_texts:>3}  {total:>5}  {in_gaz:>22}")

# ============================================================
# 3. CROSS-TEXT PERSON CONNECTIONS — who cites whom?
# ============================================================
print("\n\n" + "=" * 100)
print("3. CROSS-TEXT CONNECTIONS — authors citing each other")
print("=" * 100)

# Check which corpus authors mention other corpus authors
corpus_author_surnames = {
    "locke": "Locke", "newton": "Newton", "hume": "Hume",
    "smith": "Smith", "burke": "Burke", "johnson": "Johnson",
    "pope": "Pope", "swift": "Swift", "addison": "Addison",
    "steele": "Steele", "shaftesbury": "Shaftesbury",
    "hutcheson": "Hutcheson", "hartley": "Hartley",
    "cheyne": "Cheyne", "sydenham": "Sydenham",
    "mandeville": "Mandeville", "rousseau": "Rousseau",
    "voltaire": "Voltaire", "montesquieu": "Montesquieu",
    "priestley": "Priestley", "reid": "Reid",
    "condillac": "Condillac", "bentham": "Bentham",
    "godwin": "Godwin", "wollstonecraft": "Wollstonecraft",
    "gibbon": "Gibbon", "robertson": "Robertson",
}

# Focus on medical/conduct/philosophy texts where cross-citation matters
target_dirs = [ROOT / "nonfiction/philosophy", ROOT / "nonfiction/medical",
               ROOT / "nonfiction/conduct", ROOT / "nonfiction/science"]

print("\nKey citation chains (author A's text mentions author B):")
citation_chains = []
for d in target_dirs:
    if not d.exists():
        continue
    for f in d.rglob("*.txt"):
        text = load(f).lower()
        if len(text) < 3000:
            continue
        source = f.stem
        for surname, display in corpus_author_surnames.items():
            # Skip self-citations
            if surname in source.lower():
                continue
            count = len(re.findall(r'\b' + surname + r'\b', text))
            if count >= 5:
                citation_chains.append((source, display, count))

# Group by target and sort
by_cited = defaultdict(list)
for source, cited, count in citation_chains:
    by_cited[cited].append((source, count))

print(f"\n{'Cited authority':<20} {'Cited by (top sources)':}")
print("-" * 80)
for cited in sorted(by_cited.keys(), key=lambda x: -sum(c for _, c in by_cited[x])):
    sources = sorted(by_cited[cited], key=lambda x: -x[1])[:5]
    total = sum(c for _, c in by_cited[cited])
    src_str = "; ".join(f"{s}({c})" for s, c in sources)
    print(f"  {cited:<20} [{total:>4} total] {src_str}")
