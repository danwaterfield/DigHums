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
    "Hester Maria Thrale":  "Viscountess Keith",
    "Hester Maria Thrale (Lady Keith)": "Viscountess Keith",
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

    # ── Duplicate fixes (2026-03-20) ────────────────────────────────
    "Charles Burney Sr":                  "Charles Burney",
    "Giovanni Battista Martini":          "Padre Martini",
    "Sophia Thrale Hoare":               "Sophia Hoare",
    "Dr. Brocklesby":                     "Richard Brocklesby",
    "Frances Crewe":                      "Mrs Crewe",
    "William Locke Sr":                   "William Locke",
    "William Locke [?]":                  "William Locke",
    "CB [?]":                             "Charles Burney",
    "(CB) [?]":                           "Charles Burney",
    "10th Duke of Norfolk":               "Duke of Norfolk",
    "1 . Duke of Portland":               "Duke of Portland",
    "9. Lord North":                      "Lord North",
    "27. Mr. George Crabbe":              "George Crabbe",
    "AA [postscript only]":               "Alexander d'Arblay",
    "Joseph Warton":                      "Joseph Warton",
    "Amédée duc de Duras":                "Duc de Duras",
    "Amédée":                             "Duc de Duras",
    "Andre Gretry":                       "André Grétry",
    "Peter Paul Dobree":                  "Peter Paul Dobrée",
    "Henry Fuseli":                       "Henry Fuseli",
    # ── Bracketed/inferred names (2026-03-21) ───────────────────────
    "[CB Jr]":                            "Charles Burney Jr",
    "[CB Jr?]":                           "Charles Burney Jr",
    "[CB Jr or CPB?]":                    "Charles Burney Jr",
    "[CB Jr? or CPB?]":                   "Charles Burney Jr",
    "[to CB Jr or CBP]":                  "Charles Burney Jr",
    "CB Jr?":                             "Charles Burney Jr",
    "CB Jr ":                             "Charles Burney Jr",
    "[CPB]":                              "Charles Parr Burney",
    "[CPB?]":                             "Charles Parr Burney",
    "[Edmund Malone?]":                   "Edmond Malone",
    "[James Fordyce]":                    "James Fordyce",
    "[John Young]":                       "John Young",
    "[John?] Aiken":                      "John Aiken",
    "[John?] Aitkin":                     "John Aiken",
    "[John?] Pridden":                    "John Pridden",
    "[Joseph?] White":                    "Joseph White",
    "[Lady Crewe]":                       "Mrs Crewe",
    "[Patrick Kelly]":                    "Patrick Kelly",
    "[Ralph Griffiths]":                  "Ralph Griffiths",
    "[Reginald?] Heber":                  "Reginald Heber",
    "[Richard?] Heber":                   "Richard Heber",
    "[Robert Harding?] Evans":            "Robert Evans",
    "[Rosette (Rose) Burney":             "Rosette Burney",
    "[Samuel Butler]":                    "Samuel Butler",
    "[Sir Thomas?] Lawrence":             "Thomas Lawrence",
    "[Spencer] Percival":                 "Spencer Percival",
    "[Thomas Cadell]":                    "Thomas Cadell",
    "[Thomas?] King":                     "Thomas King",
    "[William Howley]":                   "William Howley",
    "[William Parsons]":                  "William Parsons",
    "[William Windham]":                  "William Windham",
    "[William] Davies":                   "William Davies",
    "[Wrangham]":                         "Francis Wrangham",
    "[the President of the Royal Academy]": "Benjamin West",
    "ERB [?]":                            "Edward Richard Burney",
    "SBP [?]":                            "Susanna Burney Phillips",
    "B[asil] Montagu":                    "Basil Montagu",
    "R[ichard?] Heber":                   "Richard Heber",
    "Richard [?] Heber":                  "Richard Heber",
    "S[amuel] Butler":                    "Samuel Butler",
    "J[ames?] Young":                     "James Young",
    "James [Young]":                      "James Young",
    "G.E.G[riffiths]":                    "G E Griffiths",
    "Franz [Franz Josef] Haydn":          "Joseph Haydn",
    "Dr F[rancis[ Riollay":               "Francis Riollay",
    "Harriett [Hester] Burney":           "Hester Burney",
    "Martin[?] Burney":                   "Martin Charles Burney",
    "John Young [?]":                     "John Young",
    "John Philip Kemble [?]":             "John Philip Kemble",
    "George Colman [?]":                  "George Colman",
    "Richard Twining [?]":                "Richard Twining",
    "Denise-Victoire [?]":                "Denise-Victoire",
    "Lady Hales [?]":                     "Lady Hales",
    "Sarah (Payne) [?]":                  "Sarah Payne",
    "Warren Hastings [to CB]":            "Warren Hastings",
    "William Heberden [to CB Jr or CPB?]": "William Heberden",
    "William, 2nd Earl Spencer [?]":      "Earl Spencer",
    "Andrew Spottiswoode [= ALS CB]":     "Andrew Spottiswoode",
    "C J Hatcher [to CB Jr?]":            "C J Hatcher",
    "G Woodfall [to CB Jr?]":             "George Woodfall",
    "FBA; CB; CBFB":                      "Frances Burney",
    "CBFB [with FBA EBB]":               "Charlotte Broome",
    "CB (composite of extracts of partly [?] copy in CB [?] 5 May-5 Oct 1806)": "Charles Burney",
    "Princess Charlotte [post of Udney]": "Princess Charlotte",
    "Elizabeth Juliana Schwellenberg (per E Lattikens[?])": "Elizabeth Juliana Schwellenberg",
    "Comtesse Adrienne (de Noailles) de la Tour [?]": "Adrienne de Noailles",
    "Marquis de Lafayette [?] de Gorion-Saint-Cyr": "Marquis de Lafayette",
    "Marquis [?] Bourdois":               "Marianne Bourdois",
    "Louis Bonne [?]":                    "Louis Bonne",
    "Louis Bonne[?]":                     "Louis Bonne",
    "Miss Drysdale [?]":                  "Miss Drysdale",
    "Miss Drysdale[?]":                   "Miss Drysdale",
    # ── Walpole OCR fixes ───────────────────────────────────────────
    "Hon. H. S. CONV^^AY":                "Henry Seymour Conway",
    "Hon. H. S. €0NWAT":                  "Henry Seymour Conway",
    "DAVID HUME, Esq":                    "David Hume",
    "DAVID HUME, Esq,":                   "David Hume",
    "Hon. GEORGE HARDINGB":               "George Hardinge",
    "Hon. George Hardinge":               "George Hardinge",
    "JOHN CHUTH, Esq":                    "John Chute",
    "THOMAS BARRETT, Esq":                "Thomas Barrett",
    "THOMAS BRAND, Esq":                  "Thomas Brand",
    "COUNTESS of * * * ♦":                "Countess of Ailesbury",
    # ── Burke name fixes ────────────────────────────────────────────
    "Edm. Burke, Esq":                    "Edmund Burke",
    "Edm, Burke, Esq -> Rev. John Erskine": "Edmund Burke",
    "Edm. Malone, Esq":                  "Edmond Malone",
    "Edmund Malone, Esq":                "Edmond Malone",
    "David Garrick, Esq":                "David Garrick",
    "Earl Fitzwilliam":                  "Earl Fitzwilliam",
    "Eari Fitzwilliam":                  "Earl Fitzwilliam",
    "Earl FitzwiUiam":                   "Earl Fitzwilliam",
    "Dr. French Laurence":               "French Laurence",
    "Dr. Laurence":                       "French Laurence",
    "Dr. Benj. Franklin":                "Benjamin Franklin",
    "Alexander Wedderburne, Esq":        "Alexander Wedderburne",
    "Arthur Lee, Esq":                   "Arthur Lee",
    "Arthur Pigot, Esq":                "Arthur Pigot",
    "Charles Townshend, Esq":            "Charles Townshend",
    "Gen. Oglethorpe":                   "James Oglethorpe",
    "Dr. Robertson":                     "William Robertson",
    "Chev. de la Bintinnaye":            "Chevalier de la Bintinaye",
    "Abbe de la Biritinnaye":            "Chevalier de la Bintinaye",
    "Chev. de Rivarol":                  "Antoine de Rivarol",
    "Chev. de Grave":                    "Chevalier de Grave",
    "Charlotte Beckedorff":               "Charlotte Beckerdorff",
    "Edm. Malone":                        "Edmond Malone",
    "Edmund Malone":                      "Edmond Malone",
    "Lally-Tollendal":                    "Marquis de Lally-Tollendal",
    "Marquise de Lally-Tollendal":        "Marquis de Lally-Tollendal",
    "G I Huntingford":                    "George Huntingford",
    "George I Huntingford":               "George Huntingford",
    "Sophy Streatfeild":                  "Sophia Streatfield",
    "Sophy Streatfield":                  "Sophia Streatfield",
    "Sophy Streatfield [?]":              "Sophia Streatfield",
    "Sophia Streatfield":                 "Sophia Streatfield",
    "Caroline, Princess of Wales":        "Princess Caroline",
    "Caroline, Princess of Wales (per Lady Charlotte Campbell)": "Princess Caroline",
    "Elizabeth (Allen) Meeke":            "Elizabeth Meeke",
    "Esther (Sleepe) Burney":             "Esther Burney",
    "George Hay (on verso of NYPL(B))":   "George Hay",
    "('ait. James Ceib":                  "Capt. James Keir",
    "('ait.  James Ceib":                 "Capt. James Keir",
    "Brigit[a] Piez[e] [Frontoni]":       "Brigita Piez Frontoni",
    "James Boswell, Esq":                 "James Boswell",
    "Mr. George Crabbe":                  "George Crabbe",
    "Rev. George Crabbe":                 "George Crabbe",
    "Mr. James Barry":                    "James Barry",
    "Samuel Crisp (with PS by CB)":       "Samuel Crisp",
    "Sir George Beaumont":                "George Beaumont",
    "Sir William Parsons":                "William Parsons",
    "Sir William Scott":                  "William Scott",
    "Dr. Birch":                          "Rev. Dr. Birch",
    "Dr. John Douglas":                   "John Douglas",
    "Dr. Leland":                         "Rev. Dr. Leland",
    "Rev Dr. Leland":                     "Rev. Dr. Leland",
    "Chief Justice Aston":                "Chief Justice Aston",
    "Mr. Secretary Hamilton":             "William Gerard Hamilton",
    "Rich. Champion":                     "Richard Champion",
    "Rich. Champion, Esq":                "Richard Champion",
    "Rich. Burke, Esq":                   "Richard Burke Sr",
    "Rich. Burke, Sen.":                  "Richard Burke Sr",
    "Rich. Burke, Sen., Esq":             "Richard Burke Sr",
    "Richard Burke, Esq":                 "Richard Burke Sr",
    "Rich. Burke, Jun.":                  "Richard Burke Jr",
    "Rich. Burke, Jun":                   "Richard Burke Jr",
    "Rich. Burke, Jan.":                  "Richard Burke Jr",
    "Rich. Burke, Jun., Esq":             "Richard Burke Jr",
    "Mr. Rich. Burke, Jun.":              "Richard Burke Jr",
    "Wm. Burke":                          "William Burke",
    "Wm. Burke, Esq":                     "William Burke",
    "Wm. Smith":                          "Wm. Smith",
    "Wm. Smith, Esq":                     "Wm. Smith",
    "Rt. Hon. Edrn. Burke":               "Edmund Burke",
    "Rt. Hon. Henry Dundas":              "Henry Dundas",
    "Rt. Hon. Henry Grattan":             "Henry Grattan",
    "Rt. Hon. HenryGrattan":              "Henry Grattan",
    "of Rockingham":                      "Marquis of Rockingham",
    "Mr. T. King":                        "Mr. T. King",
    "T. King":                            "Mr. T. King",
    "Mrs. Chapone":                       "Hester Chapone",

    # ── Duplicates found in 2026-03-31 audit ─────────────────────
    # Punctuation / spacing
    "E H Barker":                         "E H. Barker",
    "L M Barral":                         "L M. Barral",
    'Rev. Dr. "Wilson':                   "Rev. Dr. Wilson",
    "Sarah (Payne)":                      "Sarah Payne",
    "T J Mathias":                        "TJ Mathias",
    "G E Griffiths":                      "G.E Griffiths",

    # Articles / OCR artefacts
    "the Abbe de la Bintinnaye":          "Abbe de la Bintinnaye",
    "the Comte d'Artois":                 "Comte d'Artois",
    "the King of Poland":                 "King of Poland",
    "s. Duke of Richmond":                "Duke of Richmond",
    "Su- A I. Elton, Town Clerk of Bristol": "Sir A I. Elton, Town Clerk of Bristol",

    # Spelling / typo variants
    "Frances North, 4th Earl of Guilford": "Francis North, 4th Earl of Guilford",
    "Frederic North, 5th Earl of Guilford": "Frederick North, 5th Earl of Guilford",
    "Harriett Wilson":                    "Harriet Wilson",
    "Nevile Maskelyne":                   "Nevil Maskelyne",
    "Marie-Alexendre Lenoir":             "Marie-Alexandre Lenoir",
    "Lawrence, Parsons, 2nd Earl of Rosse": "Laurence Parsons, 2nd Earl of Rosse",
    "Francesco Sastres":                  "Francois Sastres",

    # Title / name variants
    "Charles Maurice de Talleyrand-Perigord": "Charles-Maurice de Talleyrand-Perigord",
    "Rev. Mr. Mason":                     "Mr Mason",
    "Sir William Weller Pepys":           "William Weller Pepys",
    "Rev. Walker King":                   "Walker King",
    "Payne Knight":                       "Richard Payne Knight",
    "C W LeBas":                          "W LeBas",
    "Marie-Elisabeth (Bouée) de La Fite": "Marie-Elisabethe de La Fite",
    "Dame-Adelaide (de Damas d'Antigny) comtesse de Simiane": "Comtesse de Simiane",
    "Georgiana Countess Spencer":         "Countess Spencer",
    "George 2nd Earl Spencer":            "Earl Spencer",
    "William Lowther later Lord Lonsdale": "Lord Lonsdale",
    "Baron Crewe":                        "John, 1st Baron Crewe",

    # Person_info alignment
    "R B. Sheridan":                      "R.B. Sheridan",
    "Louis DeVisme":                      "Louis Devisme",
    "Rousseau, Jean-Jacques":             "Jean-Jacques Rousseau",
    "Lord Viscount Stormont":             "Lord Stormont",
    "Regina (Valentini) Mingotti":        "Regina Mingotti",
}


# ── Artefact filter ───────────────────────────────────────────────
# Strings that appear in the correspondent field but are not people.
_ARTEFACTS = frozenset({
    "[?]", "to", "from", "", "copy", "draft", "NYPL(B)", "BM(Bar)",
    "PML", "Comyn", "Osb", "Hyde", "London", "4 p 12mo",
    "Transcript copy", "[?] transcript copy", "?", "[illegible]",
    "SE 1835", "London (Harvey)", '"',
    "^uly 1. Rt. Hon. Edm. Sexton Pery",  # OCR artefact from burke_1844
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


_CHAPTER_PREFIX_RE = re.compile(r"^ch\s+\d+\.\s*", re.IGNORECASE)
_TRAILING_NOTE_PAREN_RE = re.compile(
    r"\s+\((?:"
    r"per\b|with\b|wife\b|on\s+verso\b|former\b|formerly\b|"
    r"postscript\b|[12]\d{3}\s*[-–]\s*[12]\d{3}"
    r")[^)]*\)$",
    re.IGNORECASE,
)
_DOTTED_INITIALS_RE = re.compile(r"\b([A-Z])\.\s*(?=[A-Z]\.)")
_HONORIFIC_PREFIX_RE = re.compile(
    r"^(?:(?:Rt\.?\s+Hon\.?|Rev\.?\s+Dr\.?|Rev\.?|Dr\.?|Mr\.?|Mrs\.?|Miss\.?|Sir)\s+)+"
)
_SUFFIX_RE = re.compile(r",?\s+(?:Esq|Bart|M\.D\.|LL\.D\.)(?:,|\.)?$", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")


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
    "George Huntingford":          "Scholarly/Church",
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
    "Elizabeth Meeke":              "Family",
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
    "Lady Clifford":               "Intimate circle",
    "Madame Saintmard":            "French circle",
    "Choderlos de Laclos":          "French circle",
    "Louis de Narbonne":           "French circle",
    "Clement Francis":             "Family",
    "Ralph Broome":                "Family",
    "Molesworth Phillips":         "Family",
    "Samuel Meeke":                "Family",
    "Madame Campan":               "French circle",
    "Jacques-Louis David":         "French circle",
    "Joseph Priestley":            "Scholarly/Church",
    "Carl Philipp Emanuel Bach":   "Musical circle",
    "Wolfgang Amadeus Mozart":     "Musical circle",
    "Leopold Mozart":              "Musical circle",
    "Johann Christian Bach":       "Musical circle",
    "Georg Philipp Telemann":      "Musical circle",
    "André Grétry":                "Musical circle",
    "Richard Brocklesby":          "Literary",
    "George Crabbe":               "Literary",
    "Duke of Norfolk":             "Intimate circle",
    "Duke of Portland":            "Literary",
    "Lord North":                  "Literary",
    "Joseph Warton":               "Literary",
    "William Locke":               "Intimate circle",
    "Duc de Duras":                "French circle",
    "Mary Hamilton":               "Intimate circle",
    "Anna Ord":                    "Intimate circle",
    "Thomas Gray":                 "Literary",
    "Oliver Goldsmith":            "Literary",
    "Adam Smith":                  "Literary",
    "Henry Seymour Conway":        "Literary",
    "George Montagu":              "Literary",
    "William Cole":                "Scholarly/Church",
    "John Chute":                  "Literary",
    "George Hardinge":             "Literary",
    "Richard West":                "Literary",
    "Lady Hervey":                 "Literary",
    "Charles James Fox":           "Literary",
    "Earl Fitzwilliam":            "Literary",
    "French Laurence":             "Scholarly/Church",
    "Benjamin Franklin":           "Scholarly/Church",
    "Duke of Richmond":            "Literary",
    "Duke of Newcastle":           "Literary",
    "Duke of Dorset":              "Literary",
    "Alexander Wedderburne":       "Literary",
    "James Oglethorpe":            "Literary",
    "William Robertson":           "Scholarly/Church",
    "Antoine de Rivarol":          "French circle",
    "Comte d'Artois":              "French circle",
    "Penelope Pennington":         "Literary",
    "Elizabeth Carter":             "Literary",
    "Elizabeth Vesey":              "Literary",
    "Sarah Scott":                 "Literary",
    "Frances Reynolds":            "Literary",
    "Charles Arne":                "Musical circle",
    "Robert Strange":              "Literary",
    "Isabella Strange":            "Literary",
    "Charles Walmesley":           "Scholarly/Church",
    "Dr Thomas Bever":             "Scholarly/Church",
    "Robert Hudson":               "Musical circle",
    "Mr Drummond":                 "Intimate circle",
    "Dr Seward":                   "Literary",
}


# ── Public API ────────────────────────────────────────────────────

def _is_known_name(name: str) -> bool:
    return (
        name in CANONICAL_NAMES
        or name in CANONICAL_NAMES.values()
        or name in COMMUNITIES
    )


def _clean_candidate(name: str) -> str:
    s = name.strip()
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = _CHAPTER_PREFIX_RE.sub("", s)
    s = _DOTTED_INITIALS_RE.sub(r"\1 ", s)
    s = s.strip().strip('"').strip()
    s = re.sub(r"\[\?\]", "", s)
    s = re.sub(r"\[([^\]]+?)\?\]", r"\1", s)
    s = s.replace("?", "")
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    while True:
        stripped = _TRAILING_NOTE_PAREN_RE.sub("", s).strip()
        if stripped == s:
            break
        s = stripped
    s = _SUFFIX_RE.sub("", s).strip()
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip(" ,;:")

def normalise(name: str) -> str:
    """Map *name* to its canonical form, or return it unchanged."""
    s = name.strip()
    if s in CANONICAL_NAMES:
        return CANONICAL_NAMES[s]

    cleaned = _clean_candidate(s)
    if cleaned in CANONICAL_NAMES:
        return CANONICAL_NAMES[cleaned]

    for candidate in (
        cleaned,
        _HONORIFIC_PREFIX_RE.sub("", cleaned).strip(),
        _SUFFIX_RE.sub("", cleaned).strip(),
        _SUFFIX_RE.sub("", _HONORIFIC_PREFIX_RE.sub("", cleaned)).strip(),
    ):
        if candidate in CANONICAL_NAMES:
            return CANONICAL_NAMES[candidate]
        if candidate and candidate != cleaned and _is_known_name(candidate):
            return candidate

    if cleaned and cleaned != s:
        return cleaned
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
