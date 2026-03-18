"""
Shared Burney-family name-normalisation and community-assignment module.

Every catalogue abbreviation, married-name variant, and OUP header alias
maps to a single canonical form.  The ``normalise`` and ``community``
functions are the public API consumed by ``build_correspondent_network.py``.
"""

import re

# ── Canonical name dictionary ─────────────────────────────────────
# Maps every known variant → canonical form.
# Entries are checked in order: exact match wins.

CANONICAL_NAMES: dict[str, str] = {
    # ── Catalogue abbreviations (Hemlow / JL&L) ──────────────────
    "SBP":  "Susanna Burney Phillips",
    "SEB":  "Susanna Burney Phillips",   # Susanna Elizabeth Burney
    "HLTP": "Hester Thrale Piozzi",
    "CB":   "Charles Burney",
    "CB Jr":"Charles Burney Jr",
    "CB jr":"Charles Burney Jr",
    "CBFB": "Charlotte Broome",          # Charlotte Burney Francis Broome
    "CRFB": "Charlotte Broome",          # variant
    "EBB":  "Esther Burney",
    "JB":   "James Burney",
    "AA":   "Alexander d'Arblay",        # the son
    "M d'A":"Alexandre d'Arblay",        # the husband
    "FBA":  "Frances Burney",
    "FB":   "Frances Burney",
    "CPB":  "Charles Parr Burney",
    "SHB":  "Sarah Harriet Burney",
    "CFBt": "Charlotte Barrett",
    "CFPB": "Charlotte Barrett",
    "CAB":  "Charlotte Ann Burney",
    "CBF":  "Charlotte Broome",          # Charlotte (Broome) Francis stage
    "ERB":  "Edward Richard Burney",
    "CHFB": "Charlotte Barrett",         # Charlotte H. F. Barrett variant
    "LB":   "Lucy Burney",              # late-period correspondent
    "MF":   "Martin Charles Burney",
    "GBFB": "Charlotte Broome",          # variant
    "CHPB": "Charlotte Harriet Payne Burney",
    "CLB":  "Charles Lamb Burney",
    "CJJ":  "Charles John James",
    "LA":   "Lucy Burney",
    "GF Jr":"George Francis Jr",
    "MG":   "Mary Gwynn",

    # ── Waddington variants ───────────────────────────────────────
    "GMAPWaddington":  "Georgiana Waddington",
    "GMAFWaddington":  "Georgiana Waddington",
    "CMAPWaddington":  "Georgiana Waddington",
    "CMAPW addington": "Georgiana Waddington",
    "GMAP Waddington": "Georgiana Waddington",

    # ── OUP header aliases (from existing build script) ───────────
    "Susanna Burney":       "Susanna Burney Phillips",
    "Susanna Phillips":     "Susanna Burney Phillips",
    "Dr Burney":            "Dr Charles Burney",
    "Dr Charles Burney":    "Dr Charles Burney",
    "Hester Lynch Thrale":  "Hester Thrale Piozzi",
    "Hester Lynch Piozzi":  "Hester Thrale Piozzi",
    "Hester Maria Thrale":  "Hester Thrale Piozzi",
    "Charlotte Cambridge":  "Charlotte Broome",
    "Charlotte Francis":    "Charlotte Broome",
    "Charlotte Broome":     "Charlotte Broome",
    "Longman, Hurst, Rees, Orme and Brown": "Longman & Co",
    "Longman, Hurst, Rees, Orme":           "Longman & Co",
    "Messrs Longman and Company":            "Longman & Co",
    "Messrs Longman":                        "Longman & Co",
    "Longmans & Co":                         "Longman & Co",

    # ── Full-name aliases for catalogue entries ───────────────────
    "Charlotte Harriet":               "Charlotte Harriet Payne Burney",
    "Rosette (Rose) Burney":           "Rosette Burney",
    "Rosette [Rose] Burney":           "Rosette Burney",
    "Rosette [Hester] Burney":         "Rosette Burney",
    "Maria (Allen) Rishton":           "Maria Rishton",
    "Martin (Allen) Rishton":          "Maria Rishton",
    "Maria (Burney) Bourdois":         "Marianne Bourdois",
    "Marianne (Burney) Bourdois":      "Marianne Bourdois",
    "Marianne (Burney)":               "Marianne Bourdois",
    "Amelia (Locke) Angerstein":       "Amelia Angerstein",
    "Mary (Hamilton) Dickenson":       "Mary Dickenson",
    "Elizabeth (Allen) Burney":        "Elizabeth Allen Burney",
    "Charlotte (Allen) Burney":        "Elizabeth Allen Burney",
    "Hester (Mulso) Chapone":          "Hester Chapone",
    "Harriet (Mulso) Chapone":         "Hester Chapone",
    "Hester (Maria) Chapone [?]":      "Hester Chapone",
    "Sophia (Thrale) Hoare":           "Sophia Hoare",
    "Frances (Phillips) Raper":        "Frances Raper",
    "Frances (Phillips) Burney":       "Frances Phillips Burney",
    "Sarah (Burney) Payne":            "Sarah Payne",
    "Sarah (Payne) Burney":            "Sarah Payne Burney",
    "Cassandra (Leigh) Cooke":         "Cassandra Cooke",
    "Cecilia (Ogilvie) Lock":          "Cecilia Lock",
    "Cecilia (Ogilvie) Locke":         "Cecilia Lock",
    "Julia (Locke) Angerstein":        "Julia Angerstein",
    "Mary (Granville) Delany":         "Mary Delany",
    "Charlotte (Broome) Francis":      "Charlotte Broome",
    "Adrienne (de Noailles)":          "Adrienne de Noailles",
    "Anne-Louise-Germaine (Necker) de Stael-Holstein": "Germaine de Stael",
    "Anne-Louise (Necker) baroness de Stael-Holstein": "Germaine de Stael",
    "Louise-Germaine (Necker)":        "Germaine de Stael",
    "Elizabeth (Robinson) Montagu":    "Elizabeth Montagu",
    "Mary (Danby) Countess of Harcourt": "Countess of Harcourt",
    "Mary (Danby) Countess of Har-":   "Countess of Harcourt",
    "Mary (Danby)":                    "Countess of Harcourt",
    "Mary (Danby) Countess of Ha[rcourt]": "Countess of Harcourt",
    "Frances (Macartney) Greville":    "Frances Greville",
    "Sarah Martha Holroyd":            "Sarah Holroyd",
    "[Sarah Martha Holroyd]":          "Sarah Holroyd",
    "[Sarah Martha Holroyd?]":         "Sarah Holroyd",
    "Elizabeth Lady Templetown":       "Lady Templetown",
    "Lady Keith":                      "Viscountess Keith",
    "Anna (Dillingham) Ord":           "Anna Ord",
    "Sophia (Crisp) Gast":             "Sophia Gast",
    "Sophia (?) Gast":                 "Sophia Gast",
    "Sophia (Crisp?) Gast":            "Sophia Gast",
    "Eva Maria [Veigel] Garrick":      "Eva Garrick",
    "Eva Maria (Veigel) Garrick":      "Eva Garrick",
    "Elizabeth Lindley and Linley Sheridan": "Elizabeth Sheridan",
    "Margaret Duchess of Portland":    "Duchess of Portland",
    "Dorothy (Smelt) Godwin":          "Dorothy Godwin",
    "Jane (Smelt) Cholmeley":          "Jane Cholmeley",
    "Jane (Campbell) Smelt":           "Jane Smelt",
    "Charlotte (Jerningham) Bedingfield": "Charlotte Bedingfield",
    "Charlotte (Jerningham) Bedingfeld":  "Charlotte Bedingfield",
    "Henrietta Maria (Bannister) North":  "Henrietta North",
    "Henrietta Maria (Banister) North":   "Henrietta North",
    "Harriet (Collins) de Beauville[?]":  "Harriet de Boinville",
    "Harriet (Collins) de Boinville":     "Harriet de Boinville",
    "Harriet (Collins) de Beauvau":       "Harriet de Boinville",
    "Harriet (Collins) de Boieville":     "Harriet de Boinville",
    "Frances (Waddington) Bunsen":        "Frances Bunsen",
    "Princess (Waddington) Bunsen":       "Frances Bunsen",
    "Princess (Waddington) Bunsen(?)":    "Frances Bunsen",
    "John Montagu, 4th Earl of Sandwich": "Earl of Sandwich",
    "Garret Wesley, Earl of Mornington":  "Earl of Mornington",
    "Charlotte Boyle-Walsingham":         "Charlotte Walsingham",
    "Charlotte Boyle-Walsingham [?]":     "Charlotte Walsingham",
    "Charlotte [Boyle-Walsingham?]":      "Charlotte Walsingham",
    "Ann (Astley) Agnew":                 "Ann Agnew",
    "(Astley) Agnew":                     "Ann Agnew",
    "Anne (Leigh) Frodsham":              "Anne Frodsham",
    "Lady Lucy (Fitzgerald) Foley":       "Lady Lucy Foley",
    "Hector (Mulso) Chapone":             "Hector Chapone",
    "Catherine (de Boulogne)":            "Catherine de Boulogne",
    "Catherine (de Boullogne)":           "Catherine de Boulogne",
    "Catherine (de Bouillonne [?])":      "Catherine de Boulogne",
    "Adrienne (de Chavagnac)":            "Adrienne de Chavagnac",
    "Marie Charlotte (Bontemps)":         "Marie-Charlotte Bontemps",
    "Marie-Charlotte (Bontemps)":         "Marie-Charlotte Bontemps",
    "Henriette (Guignet de Souligne) de Maurville": "Henriette de Maurville",
    "Henriette (Guignet de Souligné) de Maurville": "Henriette de Maurville",
    "Cecile (de Riquet de Caraman) marquise de Sommery": "Cecile de Sommery",
    "Marie de Maisonneuve Ma bien chere": "Marie de Maisonneuve",
    "Marie de Maisonnneuve":              "Marie de Maisonneuve",
    "SBP mutilated":                      "Susanna Burney Phillips",
    "Dunbar (Suttanner) Fisher":          "Dunbar Fisher",
    "Dorothy Young":                      "Dorothy Young",
    "Frances Anne (Burney) Wood":         "Frances Wood",
    "[Jules, comte de] Polignac":         "Comte de Polignac",
    "Clarissa Marion":                    "Clarissa Marion Bolton",
    "Clarissa Marion ( ) Bolton":         "Clarissa Marion Bolton",
    "George Owen Cambridge":              "George Owen Cambridge",
    "Richard Owen Cambridge":             "Richard Owen Cambridge",
    "Elizabeth (Meeke?) Lamb":             "Elizabeth Lamb",
    "Mrs Crewe":                          "Mrs Crewe",
    "Lady Crewe":                         "Mrs Crewe",
    "Lady Crewes":                        "Mrs Crewe",
    "[Lady Crewe?]":                      "Mrs Crewe",
    "Lady Crewe [?]":                     "Mrs Crewe",
    "Mr Crewe":                           "Mr Crewe",
    "Lavinia (Bingham), Countess Spencer": "Countess Spencer",
    "George, 2nd Earl Spencer":           "Earl Spencer",
    "George, 2nd Earl Spencer [?]":       "Earl Spencer",
    "George, 1st Marquis Townshend":      "Marquis Townshend",
    "Sir Joseph Banks":                   "Joseph Banks",
    "Sir Joshua Reynolds":                "Joshua Reynolds",
    "Sir William Hamilton":               "William Hamilton",
    "Sir William Herschel":               "William Herschel",
    "Sir George Pretyman Tomline":        "George Pretyman Tomline",
    "Sir Thomas Lawrence":                "Thomas Lawrence",
    "Sir George Baker":                   "George Baker",
    "Dr William Bewley":                  "William Bewley",
    "[William Bewley]":                   "William Bewley",
    "[William Bewley?]":                  "William Bewley",
    "Lady Charlotte (North) Lindsay":     "Lady Charlotte Lindsay",
    "Elizabeth Anne (Smart) LeNoir":       "Elizabeth LeNoir",
    "Marie Hester (Hughes) Park":          "Marie Park",
    "Richard Twining Sr":                  "Richard Twining",
    "[Richard] Twining":                   "Richard Twining",
    "[Thomas] Twining":                    "Thomas Twining",
    "[Christian] Latrobe":                 "Christian Latrobe",
    "Caroline Princess of Wales":          "Princess Caroline",
    "Frederic, 5th Earl of Guilford":      "Earl of Guilford",
    "Henry Petty-FitzMaurice, Marquis of Lansdowne": "Marquis of Lansdowne",
    "Charles, 1st Earl Whitworth":         "Earl Whitworth",
    "Thomas, 1st Earl of Ailesbury":       "Earl of Ailesbury",
    "Steph[en] Digby":                     "Stephen Digby",
    "Lady Carmarthe[n]":                   "Lady Carmarthen",
    "[FB]":                                "Frances Burney",
    "[Princess Elizabeth to FBA]":          "Princess Elizabeth",
    "Princess [Elizabeth]":                 "Princess Elizabeth",
    "Princess [Elizabeth?]":               "Princess Elizabeth",
    "Princess [Knowles]":                  "Princess Knowles",
}


# ── Artefact filter ───────────────────────────────────────────────
# Strings that appear in the correspondent field but are not people.
_ARTEFACTS = frozenset({
    "[?]", "to", "from", "", "copy", "draft", "NYPL(B)", "BM(Bar)",
    "PML", "Comyn", "Osb", "Hyde", "London", "4 p 12mo",
    "Transcript copy", "[?] transcript copy", "?", "[illegible]",
    "SE 1835", "London (Harvey)",
})

_ARTEFACT_RE = re.compile(
    r"^(?:"
    r"\[?\?\]?"                # [?] or ?
    r"|NYPL\(B\)"
    r"|BM\(Bar\)"
    r"|PML"
    r"|Comyn"
    r"|Osb"
    r"|Hyde"
    r"|copy"
    r"|draft"
    r"|to"
    r"|from"
    r")$",
    re.IGNORECASE,
)


def is_artefact(name: str) -> bool:
    """Return True if *name* is a parsing artefact, not a real person."""
    s = name.strip()
    if s in _ARTEFACTS:
        return True
    if _ARTEFACT_RE.match(s):
        return True
    return False


# ── Community assignments ─────────────────────────────────────────

COMMUNITIES: dict[str, str] = {
    # ── Family ────────────────────────────────────────────────────
    "Charles Burney":              "Family",
    "Dr Charles Burney":           "Family",
    "Charles Burney Jr":           "Family",
    "Susanna Burney Phillips":     "Family",
    "Esther Burney":               "Family",
    "Charlotte Broome":            "Family",
    "Charlotte Ann Burney":        "Family",
    "Charlotte Barrett":           "Family",
    "James Burney":                "Family",
    "Charles Parr Burney":         "Family",
    "Maria Rishton":               "Family",
    "Alexandre d'Arblay":          "Family",
    "Alexander d'Arblay":          "Family",
    "Sarah Harriet Burney":        "Family",
    "Richard Thomas Burney":       "Family",
    "Rosette Burney":              "Family",
    "Marianne Bourdois":           "Family",
    "Elizabeth Allen Burney":       "Family",
    "Edward Richard Burney":       "Family",
    "Lucy Burney":                 "Family",
    "Martin Charles Burney":       "Family",
    "Charlotte Harriet Payne Burney": "Family",
    "Charles Lamb Burney":         "Family",
    "Frances Raper":               "Family",
    "Frances Phillips Burney":     "Family",
    "Sarah Payne":                 "Family",
    "Sarah Payne Burney":          "Family",
    "Norbury Phillips":            "Family",
    "Robert Allen Burney":         "Family",
    "Frances Wood":                "Family",
    "Sophia Burney":               "Family",
    "George Francis Jr":           "Family",
    "Frances Burney":              "Family",

    # ── Literary ──────────────────────────────────────────────────
    "Samuel Crisp":                "Literary",
    "Samuel Johnson":              "Literary",
    "Hester Thrale Piozzi":        "Literary",
    "Georgiana Waddington":        "Literary",
    "William Bewley":              "Literary",
    "Mrs Crewe":                   "Literary",
    "Horace Walpole":              "Literary",
    "Edmond Malone":               "Literary",
    "James Boswell":               "Literary",
    "Hannah More":                 "Literary",
    "Charlotte Smith":             "Literary",
    "Charlotte Lennox":            "Literary",
    "Edmund Burke":                "Literary",
    "Maria Edgeworth":             "Literary",
    "Richard Bentley":             "Literary",
    "William Seward":              "Literary",
    "Germaine de Stael":           "Literary",
    "Elizabeth Montagu":           "Literary",
    "Hester Chapone":              "Literary",
    "Frances Bowdler":             "Literary",
    "Harriet Bowdler":             "Literary",
    "George Steevens":             "Literary",
    "Sophia Hoare":                "Literary",
    "David Hume":                  "Literary",
    "Mr Crewe":                    "Literary",
    "Mary Dickenson":              "Literary",
    "Richard Owen Cambridge":      "Literary",
    "George Owen Cambridge":       "Literary",
    "Frances Greville":            "Literary",
    "Sophia Gast":                 "Literary",
    "Charles Babbage":             "Literary",
    "Robert Southey":              "Literary",
    "Sophia Streatfield":          "Literary",
    "Sophy Streatfield":           "Literary",
    "Hector Chapone":              "Literary",

    # ── Court ─────────────────────────────────────────────────────
    "Queen Charlotte":             "Court",
    "Princess Elizabeth":           "Court",
    "Margaret Planta":             "Court",
    "William Lowndes":             "Court",
    "Elizabeth Juliana Schwellenberg": "Court",
    "Charlotte Beckerdorff":       "Court",
    "Stephen Digby":               "Court",
    "Lady Carmarthen":             "Court",

    # ── Royal ─────────────────────────────────────────────────────
    "Princess Sophia":             "Royal",
    "Princess Mary":               "Royal",
    "Princess Augusta":            "Royal",
    "Princess Caroline":           "Royal",
    "Princess Knowles":            "Royal",

    # ── Publishers ────────────────────────────────────────────────
    "Thomas Lowndes":              "Publishers",
    "Longman & Co":                "Publishers",
    "Payne & Cadell":              "Publishers",
    "Ralph Griffiths":             "Publishers",

    # ── Intimate circle ───────────────────────────────────────────
    "Frederica Locke":             "Intimate circle",
    "Amelia Angerstein":           "Intimate circle",
    "William Locke":               "Intimate circle",
    "Viscountess Keith":           "Intimate circle",
    "William Wilberforce":         "Intimate circle",
    "Lady Templetown":             "Intimate circle",
    "Sarah Holroyd":               "Intimate circle",
    "Mary Delany":                 "Intimate circle",
    "Cecilia Lock":                "Intimate circle",
    "Julia Angerstein":            "Intimate circle",
    "Ann Agnew":                   "Intimate circle",
    "Cassandra Cooke":             "Intimate circle",
    "Catherine Coussmaker":        "Intimate circle",
    "Clarissa Marion Bolton":      "Intimate circle",
    "Sarah Baker":                 "Intimate circle",
    "Caroline Anna Moore":         "Intimate circle",
    "Harriet de Boinville":        "Intimate circle",
    "Henrietta North":             "Intimate circle",
    "Duchess of Portland":         "Intimate circle",
    "Charlotte Walsingham":        "Intimate circle",
    "Lady Hales":                  "Intimate circle",
    "Leonard Smelt":               "Intimate circle",
    "Jane Smelt":                  "Intimate circle",

    # ── French circle ─────────────────────────────────────────────
    "Marie de Maisonneuve":        "French circle",
    "Marquis de Lally-Tollendal":  "French circle",
    "Marquis de Lafayette":        "French circle",
    "Victor de Latour Maubourg":   "French circle",
    "Philippe de Noailles":        "French circle",
    "Adrienne de Noailles":        "French circle",
    "Duc de Luxembourg":           "French circle",
    "Louis":                       "French circle",
    "Angelique":                   "French circle",
    "Denise-Victoire":             "French circle",
    "Etiennette":                  "French circle",
    "Jean-Gabriel Peltier":        "French circle",
    "Arnail-Francois":             "French circle",
    "Antoine-Marie-Rene Terrier de Monciel": "French circle",
    "Florentin de Latour Maubourg": "French circle",
    "Mathieu":                     "French circle",
    "Laurent":                     "French circle",
    "Louise-Josephine":            "French circle",
    "Adrienne de Chavagnac":       "French circle",
    "Felicite":                    "French circle",
    "Marie-Charlotte Bontemps":    "French circle",
    "Marie-Louise":                "French circle",
    "Charles Sicard":              "French circle",
    "Jean-Baptiste-Gabriel Bazille": "French circle",
    "Louis-Alexandre Berthier":    "French circle",
    "Henriette de Maurville":      "French circle",
    "Catherine de Boulogne":       "French circle",
    "Cecile de Sommery":           "French circle",
    "Mme de la Grange":            "French circle",
    "Mme Bion":                    "French circle",
    "Jean-Baptiste Le Chevalier":  "French circle",
    "Jean-Baptiste-Antoine Suard": "French circle",
    "Francois Sastres":            "French circle",

    # ── Musical circle ────────────────────────────────────────────
    "Thomas Twining":              "Musical circle",
    "Padre Martini":               "Musical circle",
    "Gasparo Pacchierotti":        "Musical circle",
    "Christian Latrobe":           "Musical circle",
    "David Garrick":               "Musical circle",
    "John Wall Callcott":          "Musical circle",
    "William Mason":               "Musical circle",
    "Earl of Mornington":          "Musical circle",
    "Richard Cox":                 "Musical circle",
    "Denis Diderot":               "Musical circle",
    "Christoph Daniel Ebeling":    "Musical circle",
    "Johann Christian Hittner":    "Musical circle",
    "Francesco Roncaglia":         "Musical circle",
    "Joseph Cooper Walker":        "Musical circle",
    "William Herschel":            "Musical circle",
    "Eva Garrick":                 "Musical circle",

    # ── Scholarly/Church ──────────────────────────────────────────
    "Samuel Parr":                 "Scholarly/Church",
    "G I Huntingford":             "Scholarly/Church",
    "George I Huntingford":        "Scholarly/Church",
    "John Kaye":                   "Scholarly/Church",
    "Peter Paul Dobree":           "Scholarly/Church",
    "Richard Payne Knight":        "Scholarly/Church",
    "Thomas Dampier":              "Scholarly/Church",
    "George Pretyman Tomline":     "Scholarly/Church",
    "John Young":                  "Scholarly/Church",
    "C J Blomfield":               "Scholarly/Church",
    "John Philip Kemble":          "Scholarly/Church",
    "William Vincent":             "Scholarly/Church",
    "Shute Barrington":            "Scholarly/Church",
    "Patrick Kelly":               "Scholarly/Church",
    "Thomas Maurice":              "Scholarly/Church",
    "Charles Butler":              "Scholarly/Church",
    "Joseph Planta":               "Scholarly/Church",
    "Robert Finch":                "Scholarly/Church",
    "Gilbert Gerard":              "Scholarly/Church",
    "Thomas Gaisford":             "Scholarly/Church",
    "Edward Copleston":            "Scholarly/Church",
    "Henry Drury":                 "Scholarly/Church",
    "Martin Davy":                 "Scholarly/Church",
    "Thomas Pennant":              "Scholarly/Church",
    "Jacob Bryant":                "Scholarly/Church",
    "Arthur Young":                "Scholarly/Church",
    "James Hutton":                "Scholarly/Church",
    "Anthony Chamier":             "Scholarly/Church",
    "Fulke Greville":              "Scholarly/Church",
    "Joseph Banks":                "Scholarly/Church",
    "Joshua Reynolds":             "Scholarly/Church",
    "Thomas Lawrence":             "Scholarly/Church",
    "William Hamilton":            "Scholarly/Church",
    "George Baker":                "Scholarly/Church",

    # ── Misc known ────────────────────────────────────────────────
    "Earl of Sandwich":            "Literary",
    "Dorothy Young":               "Family",
    "Charlotte Bedingfield":       "Intimate circle",
    "Frances Bunsen":              "Intimate circle",
    "Lady Hales":                  "Intimate circle",
    "Ann Agnew":                   "Intimate circle",
    "George Hay":                  "Intimate circle",
    "Thomas Barlow":               "Intimate circle",
    "Mary Bruce Strange":          "Intimate circle",
    "Isabella Strange":            "Intimate circle",
    "Lady Lucy Foley":             "Intimate circle",
    "Lady Charlotte Fitzgerald":   "Intimate circle",
    "Anne Frodsham":               "Intimate circle",
    "Elizabeth Sheridan":           "Literary",
    "Richard Twining":             "Musical circle",
    "Edward Foss":                 "Scholarly/Church",
    "Countess of Harcourt":        "Intimate circle",
    "Earl Spencer":                "Scholarly/Church",
    "Countess Spencer":            "Intimate circle",
    "Comte de Polignac":           "French circle",
    "Marquis Townshend":           "Literary",
    "Earl Whitworth":              "Literary",
    "Earl of Ailesbury":           "Intimate circle",
    "Earl of Guilford":            "Scholarly/Church",
    "Marquis of Lansdowne":        "Scholarly/Church",
    "Lady Charlotte Lindsay":      "Intimate circle",
    "Elizabeth LeNoir":             "Literary",
    "Marie Park":                  "Literary",
    "Mary Gwynn":                  "Intimate circle",
    "Charles John James":          "Scholarly/Church",
    "Henry Thrale":                "Literary",
    "Benjamin Waddington":         "Intimate circle",
    "Emilia & Frances Waddington": "Intimate circle",
    "William Tudor":               "Literary",
    "Harriet Wilson":              "Intimate circle",
    "Dunbar Fisher":               "Intimate circle",
    "Christopher Smart":           "Literary",
    "Robert Burnside":             "Scholarly/Church",
    "Edward Miller":               "Musical circle",
    "Samuel Wesley":               "Musical circle",
    "William Ayrton":              "Musical circle",
    "Edmond Ayrton":               "Musical circle",
    "William Crotch":              "Musical circle",
    "George Dyer":                 "Literary",
    "Percival Stockdale":          "Literary",
    "Elizabeth Lamb":               "Intimate circle",
    "John Graham":                 "Scholarly/Church",
    "Edward Mangin":               "Literary",
    "Eleanore princess d'Henin":   "French circle",
    "Lady Mary Lowther":           "Intimate circle",
    "[Lady] Mary Lowther":         "Intimate circle",
    "Ella Cornelia Knight":        "Literary",
    "Warren Hastings":             "Literary",
    "Charles Blagden":             "Scholarly/Church",
    "William Windham":             "Literary",
    "Lady Banks":                  "Scholarly/Church",
    "Lady Bute":                   "Intimate circle",
    "Lady Rothes":                 "Intimate circle",
    "Stephen Allen":               "Intimate circle",
}


# ── Public API ────────────────────────────────────────────────────

def normalise(name: str) -> str:
    """Map *name* to its canonical form, or return it unchanged."""
    s = name.strip()
    if s in CANONICAL_NAMES:
        return CANONICAL_NAMES[s]
    return s


def community(name: str) -> str:
    """Return the community string for *name*, or ``'Unknown'``."""
    canon = normalise(name)
    if canon in COMMUNITIES:
        return COMMUNITIES[canon]
    # Try the raw name too
    if name in COMMUNITIES:
        return COMMUNITIES[name]
    return "Unknown"
