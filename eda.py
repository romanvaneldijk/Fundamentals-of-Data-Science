# Exploratory data analysis: UN General Debate speeches, CO2 emissions and electricity generation
#
# Question:
#   Did countries that talked more about climate change in their UN General Debate speeches around the
#   Kyoto Protocol (1997-98) and the Paris Agreement (2015-16) go on to reduce their CO2 emissions per
#   capita, and shift their electricity generation away from fossil fuels, more than countries that
#   talked less over the following decade?
#
# Steps in this file:
#   0. settings
#   1. load the speeches
#   2. clean the text
#   3. lowercase text without stopwords (for the predictive part later)
#   4. count climate mentions per speech
#   5. load the Our World in Data CO2 and electricity data
#   6. combine all datasets (speeches + UNSD codes + CO2 + electricity + Annex I list)
#   7. describe the dataset
#   8. build the Kyoto and Paris tables (one row per country)
#   9. compare the groups (numbers that answer the question)
#  10. figures
#
# Data:
#   UN General Debate Corpus: UNGDC_1946-2025.tar.gz from
#       https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y
#       (unpack it so the folder dataverse_files/TXT exists)
#   UNSD country codes: 'UNSD — Methodology.csv' from https://unstats.un.org/unsd/methodology/m49/overview/
#   Our World in Data CO2 data: https://github.com/owid/co2-data (downloaded automatically)
#   Our World in Data energy data: https://github.com/owid/energy-data (downloaded automatically)
#   Annex I countries: https://unfccc.int/parties-observers (list typed in below)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

TXT_FOLDER = "./dataverse_files/TXT"
UNSD_FILE = "UNSD — Methodology.csv"
DATA_FOLDER = "./data"
RESULTS_FOLDER = "./results"
FIGURES_FOLDER = "./figures"

CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
CO2_FILE = os.path.join(DATA_FOLDER, "owid-co2-data.csv")
ENERGY_FILE = os.path.join(DATA_FOLDER, "owid-energy-data.csv")

# words and phrases we count as talking about climate change, see report for clarification
CLIMATE_TERMS = ["climat", "global warming", "greenhouse", "carbon", "emission",
                 "paris agreement", "kyoto", "sea level", "sea-level", "rising seas",
                 "renewable", "net zero", "net-zero", "fossil fuel", "unfccc",
                 "cop21", "cop 21", "cop26", "cop 26", "cop27", "cop 27", "cop28", "cop 28"]

# terms for the two climate agreements
# Kyoto Protocol: agreed in December 1997 (already discussed in the 1996 and 1997 speeches)
# Paris Agreement: agreed in December 2015 (the September 2015 speeches talk about the "Paris conference")
# Before these years the same words refer to other things (e.g. the 1973 Paris agreement on Viet Nam),
# so we only count them from these years on.
KYOTO_TERMS = ["kyoto"]
KYOTO_START_YEAR = 1996
PARIS_TERMS = ["paris agreement", "paris accord", "paris climate", "paris conference", "cop21", "cop 21"]
PARIS_START_YEAR = 2015

# countries that no longer exist (not in UNSD file): Czechoslovakia, East Germany, Yugoslavia, South Yemen, EU
REMOVE_CODES = ["CSK", "DDR", "YUG", "YMD", "EU"]

# Annex I countries of the UN climate convention (UNFCCC)
ANNEX_I = ["AUS", "AUT", "BLR", "BEL", "BGR", "CAN", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
           "DEU", "GRC", "HUN", "ISL", "IRL", "ITA", "JPN", "LVA", "LIE", "LTU", "LUX", "MLT", "MCO",
           "NLD", "NZL", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR",
           "UKR", "GBR", "USA"]
ANNEX_LABEL = "Annex I (developed)"
NON_ANNEX_LABEL = "Non-Annex I (developing)"
COUNTRY_GROUPS = [ANNEX_LABEL, NON_ANNEX_LABEL]

# years of the speeches we look at for each agreement
KYOTO_TALK_YEARS = [1997, 1998]
PARIS_TALK_YEARS = [2015, 2016]

# time windows for the outcomes ("the following decade")
KYOTO_CO2_WINDOW = (1997, 2007)
# electricity data for most developing countries starts in 2000
KYOTO_ELEC_WINDOW = (2000, 2007)
KYOTO_CHECK_END = 2012
PARIS_START = 2015
PARIS_CHECK_END = 2019
# the end year for Paris is the latest year with data for most countries (computed in step 5)

# colours (colour-blind friendly, Okabe-Ito palette); the same group always gets the same colour
GROUP_COLOURS = {ANNEX_LABEL: "#0072B2", NON_ANNEX_LABEL: "#E69F00"}
SOURCE_COLOURS = {"Fossil fuels": "#7F7F7F", "Nuclear": "#CC79A7", "Renewables": "#009E73"}

# creates results folder
for folder in [DATA_FOLDER, RESULTS_FOLDER, FIGURES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# loads speeches
print("Loading speeches")
data = []

for folder in sorted(os.listdir(TXT_FOLDER)):
    if folder[0] == ".":  # skip hidden files like .DS_Store
        continue
    for filename in os.listdir(os.path.join(TXT_FOLDER, folder)):
        if filename[0] == ".":
            continue
        with open(os.path.join(TXT_FOLDER, folder, filename), encoding="utf-8") as f:
            splt = filename.split("_")
            data.append([int(splt[1]), int(splt[2][:4]), splt[0], f.read()])

df_speech = pd.DataFrame(data, columns=["Session", "Year", "ISO-alpha3 Code", "Speech"])
print(len(df_speech), "speeches loaded")

# clean text
def is_junk_line(line):
    s = line.strip()
    # for page numbers
    if s.isdigit():
        return True
    # for document codes
    if s.startswith("A/") and "/PV." in s:
        return True
    if len(s) >= 8 and s[:2].isdigit() and s[2] == "-" and s[3:8].isdigit():
        return True
    #page headers
    if "General Assembly" in s and len(s) < 60:
        if s[0].isdigit() and "session" in s:
            return True
        if s.startswith("General Assembly") and "meeting" in s:
            return True
    return False

# removes paragraph numbers at the start
def remove_paragraph_number(line):
    parts = line.split(maxsplit=1)
    if len(parts) == 2 and parts[0].endswith(".") and parts[0][:-1].isdigit() and len(parts[0]) <= 4:
        return parts[1]
    return line

# 2025 speeches were transcribed from audio
# removes intro
def remove_2025_intro(text):
    if not (text.startswith("The Assembly") or "give the floor" in text[:60]):
        return text
    intro_words = ["Excellency", "the floor", "Assembly will", "address the Assembly"]
    sentences = text.split(". ")
    while len(sentences) > 1 and any(word in sentences[0] for word in intro_words):
        sentences = sentences[1:]
    return ". ".join(sentences)

def clean_speech(text, year):
    lines = []
    for line in text.split("\n"):
        if is_junk_line(line):
            continue
        line = remove_paragraph_number(line).strip()
        # words split over two lines, e.g. "multi-" + "lateral"
        if len(lines) > 0 and lines[-1].endswith("-") and line[:1].islower():
            lines[-1] = lines[-1][:-1] + line
        else:
            lines.append(line)
    text = " ".join(lines)
    text = " ".join(text.split())
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    if year == 2025:
        text = remove_2025_intro(text)
    return text

print("Cleaning text")
df_speech["Speech clean"] = [clean_speech(text, year) for text, year in zip(df_speech["Speech"], df_speech["Year"])]
df_speech["Words raw"] = df_speech["Speech"].str.split().str.len()
df_speech["Words"] = df_speech["Speech clean"].str.split().str.len()
print("Removed", df_speech["Words raw"].sum() - df_speech["Words"].sum(), "words (headers, page numbers etc.)")


# make words lowercase and remove stopwords
sw = set(stopwords.words("english"))


def preprocess(text):
    words = []
    for w in text.lower().split():
        w = w.strip(".,;:!?\"'()[]")
        if w.isalpha() and w not in sw and len(w) > 2:
            words.append(w)
    return " ".join(words)


print("Making lowercase text for modelling...")
df_speech["Speech model"] = df_speech["Speech clean"].apply(preprocess)


# how much is climate change mentioned per speech
def count_terms(text, terms):
    text = text.lower()
    return sum(text.count(term) for term in terms)


def share_climate_sentences(text):
    """Fraction of sentences that mention at least one climate term."""
    sentences = [s.lower() for s in sent_tokenize(text) if len(s.split()) > 3]
    if len(sentences) == 0:
        return np.nan, 0
    n_climate = 0
    for s in sentences:
        if any(term in s for term in CLIMATE_TERMS):
            n_climate += 1
    return n_climate / len(sentences), len(sentences)


print("Counting climate mentions...")
df_speech["Climate mentions"] = [count_terms(text, CLIMATE_TERMS) for text in df_speech["Speech clean"]]
df_speech["Climate per 1000 words"] = 1000 * df_speech["Climate mentions"] / df_speech["Words"]

kyoto = []
paris = []
for text, year in zip(df_speech["Speech clean"], df_speech["Year"]):
    if year >= KYOTO_START_YEAR:
        kyoto.append(count_terms(text, KYOTO_TERMS))
    else:
        kyoto.append(0)
    if year >= PARIS_START_YEAR:
        paris.append(count_terms(text, PARIS_TERMS))
    else:
        paris.append(0)
df_speech["Kyoto mentions"] = kyoto
df_speech["Paris mentions"] = paris

results = [share_climate_sentences(text) for text in df_speech["Speech clean"]]
df_speech["Climate sentence share"] = [r[0] for r in results]
df_speech["Sentences"] = [r[1] for r in results]


# ===========================================================================
# 5. Load the Our World in Data CO2 and electricity data
# ===========================================================================
def load_owid(url, filename):
    """Download the file the first time, afterwards read the saved copy."""
    if not os.path.exists(filename):
        print("Downloading", url)
        pd.read_csv(url).to_csv(filename, index=False)
    return pd.read_csv(filename)


print("Loading Our World in Data files...")
df_co2_raw = load_owid(CO2_URL, CO2_FILE)
df_energy_raw = load_owid(ENERGY_URL, ENERGY_FILE)

co2_columns = ["iso_code", "year", "co2_per_capita", "co2", "population"]
energy_columns = ["iso_code", "year", "fossil_elec_per_capita", "nuclear_elec_per_capita",
                  "renewables_elec_per_capita"]
for columns, df in [(co2_columns, df_co2_raw), (energy_columns, df_energy_raw)]:
    missing = [c for c in columns if c not in df.columns]
    if len(missing) > 0:
        raise ValueError("Column(s) not found in Our World in Data file: " + str(missing))

# keep only real countries: rows for 'World', continents, income groups etc. have no ISO code
# or a code starting with 'OWID_'
df_co2 = df_co2_raw[co2_columns].copy()
df_co2 = df_co2[df_co2["iso_code"].notna() & ~df_co2["iso_code"].str.startswith("OWID")]
df_energy = df_energy_raw[energy_columns].copy()
df_energy = df_energy[df_energy["iso_code"].notna() & ~df_energy["iso_code"].str.startswith("OWID")].copy()

df_co2 = df_co2.rename(columns={"iso_code": "ISO-alpha3 Code", "year": "Year",
                                "co2_per_capita": "CO2 per capita (t)", "co2": "CO2 (Mt)",
                                "population": "Population"})
df_energy = df_energy.rename(columns={"iso_code": "ISO-alpha3 Code", "year": "Year",
                                      "fossil_elec_per_capita": "Fossil elec per capita (kWh)",
                                      "nuclear_elec_per_capita": "Nuclear elec per capita (kWh)",
                                      "renewables_elec_per_capita": "Renewable elec per capita (kWh)"})

# share of electricity that comes from low-carbon sources (nuclear + renewables), in %
total = (df_energy["Fossil elec per capita (kWh)"] + df_energy["Nuclear elec per capita (kWh)"]
         + df_energy["Renewable elec per capita (kWh)"])
low_carbon = df_energy["Nuclear elec per capita (kWh)"] + df_energy["Renewable elec per capita (kWh)"]
df_energy["Low-carbon share of electricity (%)"] = 100 * low_carbon / total.replace(0, np.nan)


def latest_good_year(df, column, min_countries=100):
    """Latest year in which at least min_countries countries have data (the most recent year
    is often still incomplete)."""
    counts = df.dropna(subset=[column]).groupby("Year").size()
    good = counts[counts >= min_countries]
    if len(good) == 0:
        return counts.index.max()
    return good.index.max()


co2_end = latest_good_year(df_co2, "CO2 per capita (t)")
elec_end = latest_good_year(df_energy, "Low-carbon share of electricity (%)")
PARIS_END = int(min(co2_end, elec_end))
print("Latest year with CO2 data: %d, with electricity data: %d -> Paris window %d-%d"
      % (co2_end, elec_end, PARIS_START, PARIS_END))

print("Countries with data in key years:")
for year in [1997, 2000, 2007, 2012, 2015, 2019, PARIS_END]:
    n_co2 = df_co2[(df_co2["Year"] == year)]["CO2 per capita (t)"].notna().sum()
    n_elec = df_energy[(df_energy["Year"] == year)]["Low-carbon share of electricity (%)"].notna().sum()
    print("  %d: CO2 %3d countries, electricity %3d countries" % (year, n_co2, n_elec))


# ===========================================================================
# 6. Combine all datasets
# ===========================================================================
print("Merging datasets...")
df_speech = df_speech[~df_speech["ISO-alpha3 Code"].isin(REMOVE_CODES)]

# UNSD country codes (Lab 3 assignment notebook, Q1)
df_codes = pd.read_csv(UNSD_FILE, sep=";")
df_codes = df_codes[["Country or Area", "Region Name", "Sub-region Name", "Intermediate Region Name",
                     "ISO-alpha3 Code", "Least Developed Countries (LDC)",
                     "Land Locked Developing Countries (LLDC)", "Small Island Developing States (SIDS)"]]
# the LDC / LLDC / SIDS columns contain an "x" or are empty -> make them True / False
for col in ["Least Developed Countries (LDC)", "Land Locked Developing Countries (LLDC)",
            "Small Island Developing States (SIDS)"]:
    df_codes[col] = df_codes[col] == "x"

df_all = pd.merge(df_speech, df_codes, on="ISO-alpha3 Code", how="left")
print("  speeches without a UNSD match:", df_all["Region Name"].isna().sum())

# CO2 and electricity (left merge: keep every speech, also if there is no CO2 / electricity data)
df_all = pd.merge(df_all, df_co2, on=["ISO-alpha3 Code", "Year"], how="left")
df_all = pd.merge(df_all, df_energy, on=["ISO-alpha3 Code", "Year"], how="left")

# Annex I or not
df_all["Annex I"] = df_all["ISO-alpha3 Code"].isin(ANNEX_I)
df_all["Country group"] = np.where(df_all["Annex I"], ANNEX_LABEL, NON_ANNEX_LABEL)
df_all = df_all.sort_values(["Year", "ISO-alpha3 Code"])

no_co2 = df_all[(df_all["Year"] >= 1990) & (df_all["Year"] <= co2_end) & df_all["CO2 per capita (t)"].isna()]
print("  speeches since 1990 without CO2 data:", len(no_co2),
      "- countries:", sorted(no_co2["ISO-alpha3 Code"].unique()))

# the same country groups for the Our World in Data tables (used in the figures)
speech_countries = df_all["ISO-alpha3 Code"].unique()
df_co2 = df_co2[df_co2["ISO-alpha3 Code"].isin(speech_countries)].copy()
df_energy = df_energy[df_energy["ISO-alpha3 Code"].isin(speech_countries)].copy()
df_co2["Country group"] = np.where(df_co2["ISO-alpha3 Code"].isin(ANNEX_I), ANNEX_LABEL, NON_ANNEX_LABEL)
df_energy["Country group"] = np.where(df_energy["ISO-alpha3 Code"].isin(ANNEX_I), ANNEX_LABEL, NON_ANNEX_LABEL)

# save
df_all.drop(columns=["Speech"]).to_pickle(os.path.join(DATA_FOLDER, "speeches_clean.pkl"))
df_all.drop(columns=["Speech", "Speech clean", "Speech model"]).to_csv(
    os.path.join(DATA_FOLDER, "speech_features.csv"), index=False)

# some random examples per climate term, to check by hand if the terms are really about climate
np.random.seed(42)
recent = df_all[df_all["Year"] >= 1990]
with open(os.path.join(DATA_FOLDER, "keyword_check.txt"), "w", encoding="utf-8") as f:
    for term in CLIMATE_TERMS:
        examples = []
        for code, year, text in zip(recent["ISO-alpha3 Code"], recent["Year"], recent["Speech clean"]):
            i = text.lower().find(term)
            if i != -1:
                examples.append(code + " " + str(year) + ": ..." + text[max(0, i - 90):i + len(term) + 90] + "...")
        f.write("\n=== " + term + " (" + str(len(examples)) + " speeches since 1990)\n")
        if len(examples) > 0:
            for i in np.random.choice(len(examples), size=min(15, len(examples)), replace=False):
                f.write("  " + examples[i] + "\n")
print("Saved data/speeches_clean.pkl, data/speech_features.csv and data/keyword_check.txt")


# ===========================================================================
# 7. Describe the dataset (Lecture 2: how many entries? range? mean and variance?)
# ===========================================================================
print("\n========== DESCRIBING THE DATA ==========")
print("Speeches:", len(df_all), "| countries:", df_all["ISO-alpha3 Code"].nunique(),
      "| years:", df_all["Year"].min(), "-", df_all["Year"].max())

speeches_per_year = df_all.groupby("Year").size()
speeches_per_year.to_csv(os.path.join(RESULTS_FOLDER, "describe_speeches_per_year.csv"), header=["Speeches"])
print("Speeches per year: min %d (%d), max %d (%d)" % (speeches_per_year.min(), speeches_per_year.idxmin(),
                                                       speeches_per_year.max(), speeches_per_year.idxmax()))

length_stats = df_all[["Words", "Sentences", "Climate mentions", "Climate per 1000 words"]].describe()
length_stats.to_csv(os.path.join(RESULTS_FOLDER, "describe_speech_statistics.csv"))
print("\nSpeech statistics (all years):")
print(length_stats.round(2))

missing_values = df_all.drop(columns=["Speech", "Speech clean", "Speech model"]).isna().sum()
missing_values.to_csv(os.path.join(RESULTS_FOLDER, "describe_missing_values.csv"), header=["Missing"])
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

df_all["Decade"] = (df_all["Year"] // 10) * 10
df_all["Mentions climate"] = df_all["Climate mentions"] > 0
per_decade = df_all.groupby("Decade").agg(speeches=("Year", "size"),
                                          mean_per_1000_words=("Climate per 1000 words", "mean"),
                                          median_per_1000_words=("Climate per 1000 words", "median"),
                                          std_per_1000_words=("Climate per 1000 words", "std"),
                                          share_mentioning=("Mentions climate", "mean"))
per_decade.to_csv(os.path.join(RESULTS_FOLDER, "describe_climate_talk_per_decade.csv"))
print("\nClimate talk per decade:")
print(per_decade.round(2))

outcome_rows = []
for year in [1997, 2000, 2007, 2012, 2015, 2019, PARIS_END]:
    for group in COUNTRY_GROUPS:
        co2_values = df_co2[(df_co2["Year"] == year) & (df_co2["Country group"] == group)]["CO2 per capita (t)"]
        elec_values = df_energy[(df_energy["Year"] == year) & (df_energy["Country group"] == group)][
            "Low-carbon share of electricity (%)"]
        outcome_rows.append([year, group, co2_values.count(), co2_values.mean(), co2_values.median(),
                             co2_values.std(), elec_values.count(), elec_values.mean(), elec_values.median(),
                             elec_values.std()])
outcome_stats = pd.DataFrame(outcome_rows, columns=["Year", "Country group", "CO2 n", "CO2 mean", "CO2 median",
                                                    "CO2 std", "Low-carbon share n", "Low-carbon share mean",
                                                    "Low-carbon share median", "Low-carbon share std"])
outcome_stats.to_csv(os.path.join(RESULTS_FOLDER, "describe_outcomes.csv"), index=False)
print("\nCO2 per capita (t) and low-carbon share of electricity (%) in key years:")
print(outcome_stats.round(2).to_string(index=False))


# ===========================================================================
# 8. Build the Kyoto and Paris tables (one row per country)
# ===========================================================================
def values_in_year(df, column, year):
    """Series with the value of `column` in `year`, with the ISO code as index."""
    return df[df["Year"] == year].set_index("ISO-alpha3 Code")[column]


def add_outcomes(table, co2_window, elec_window, check_end):
    co2_start = values_in_year(df_co2, "CO2 per capita (t)", co2_window[0])
    co2_end_values = values_in_year(df_co2, "CO2 per capita (t)", co2_window[1])
    co2_check = values_in_year(df_co2, "CO2 per capita (t)", check_end)
    elec_start = values_in_year(df_energy, "Low-carbon share of electricity (%)", elec_window[0])
    elec_end_values = values_in_year(df_energy, "Low-carbon share of electricity (%)", elec_window[1])
    elec_check = values_in_year(df_energy, "Low-carbon share of electricity (%)", check_end)

    co2_start = co2_start[co2_start > 0]  # avoid dividing by zero
    table["CO2 start"] = co2_start
    table["CO2 end"] = co2_end_values
    table["CO2 change (%)"] = 100 * (table["CO2 end"] / table["CO2 start"] - 1)
    table["CO2 change check (%)"] = 100 * (co2_check / table["CO2 start"] - 1)
    table["Low-carbon share start"] = elec_start
    table["Low-carbon share end"] = elec_end_values
    table["Low-carbon share change (pp)"] = table["Low-carbon share end"] - table["Low-carbon share start"]
    table["Low-carbon share change check (pp)"] = elec_check - table["Low-carbon share start"]
    return table


def talk_table(years):
    speeches = df_all[df_all["Year"].isin(years)]
    table = speeches.groupby("ISO-alpha3 Code").agg({"Climate mentions": "sum",
                                                     "Climate per 1000 words": "mean",
                                                     "Country or Area": "first"})
    table["Country group"] = np.where(table.index.isin(ANNEX_I), ANNEX_LABEL, NON_ANNEX_LABEL)
    return table


# --- Kyoto: mentioned climate in 1997 or 1998 (yes / no)
kyoto_table = talk_table(KYOTO_TALK_YEARS)
kyoto_table["Talk group"] = np.where(kyoto_table["Climate mentions"] > 0, "Mentioned climate", "Did not mention")
kyoto_table = add_outcomes(kyoto_table, KYOTO_CO2_WINDOW, KYOTO_ELEC_WINDOW, KYOTO_CHECK_END)
KYOTO_ORDER = ["Did not mention", "Mentioned climate"]

# --- Paris: almost every country mentioned climate in 2015-16, so we compare thirds of
#     'climate mentions per 1000 words', made separately for Annex I and non-Annex I countries
paris_table = talk_table(PARIS_TALK_YEARS)
paris_table["Talk group"] = ""
for group in COUNTRY_GROUPS:
    in_group = paris_table["Country group"] == group
    talk = paris_table.loc[in_group, "Climate per 1000 words"]
    low_cut = talk.quantile(1 / 3)
    high_cut = talk.quantile(2 / 3)
    for iso in talk.index:
        if talk[iso] <= low_cut:
            paris_table.loc[iso, "Talk group"] = "Low third"
        elif talk[iso] <= high_cut:
            paris_table.loc[iso, "Talk group"] = "Middle third"
        else:
            paris_table.loc[iso, "Talk group"] = "High third"
paris_table["Mentioned climate"] = paris_table["Climate mentions"] > 0   # only used for figure 8
paris_table = add_outcomes(paris_table, (PARIS_START, PARIS_END), (PARIS_START, PARIS_END), PARIS_CHECK_END)
PARIS_ORDER = ["Low third", "Middle third", "High third"]

kyoto_table.to_csv(os.path.join(RESULTS_FOLDER, "kyoto_table.csv"))
paris_table.to_csv(os.path.join(RESULTS_FOLDER, "paris_table.csv"))
print("\nKyoto table: %d countries, Paris table: %d countries" % (len(kyoto_table), len(paris_table)))
print("Kyoto group sizes:")
print(kyoto_table.groupby(["Country group", "Talk group"]).size())
print("Paris group sizes:")
print(paris_table.groupby(["Country group", "Talk group"]).size())


# ===========================================================================
# 9. Compare the groups
# ===========================================================================
def share_of_pairs_better(talkers, others, lower_is_better):
    """Share of all (talker, other) pairs in which the talker improved more.
    0.5 = no difference, above 0.5 = talkers improved more."""
    better = 0
    total = 0
    for a in talkers:
        for b in others:
            total += 1
            if (lower_is_better and a < b) or (not lower_is_better and a > b):
                better += 1
            elif a == b:
                better += 0.5
    if total == 0:
        return np.nan
    return better / total


def compare_groups(table, agreement, outcome, order, lower_is_better):
    """Summary statistics per country group and talk group, plus the pair share and the
    rank (Spearman) correlation between climate talk and the outcome."""
    rows = []
    for group in COUNTRY_GROUPS + ["All countries (pooled)"]:
        if group == "All countries (pooled)":
            sub = table.dropna(subset=[outcome])
        else:
            sub = table[table["Country group"] == group].dropna(subset=[outcome])
        for talk_group in order:
            values = sub[sub["Talk group"] == talk_group][outcome]
            rows.append([agreement, outcome, group, talk_group, values.count(), values.median(),
                         values.mean(), values.std(), np.nan, np.nan])
        talkers = sub[sub["Talk group"] == order[-1]][outcome]
        others = sub[sub["Talk group"] == order[0]][outcome]
        pairs = share_of_pairs_better(talkers.values, others.values, lower_is_better)
        spearman = sub["Climate per 1000 words"].corr(sub[outcome], method="spearman")
        rows.append([agreement, outcome, group, order[-1] + " vs " + order[0], len(talkers) + len(others),
                     np.nan, np.nan, np.nan, pairs, spearman])
    return rows


print("\n========== RESULTS ==========")
result_rows = []
result_rows += compare_groups(kyoto_table, "Kyoto", "CO2 change (%)", KYOTO_ORDER, True)
result_rows += compare_groups(kyoto_table, "Kyoto", "Low-carbon share change (pp)", KYOTO_ORDER, False)
result_rows += compare_groups(paris_table, "Paris", "CO2 change (%)", PARIS_ORDER, True)
result_rows += compare_groups(paris_table, "Paris", "Low-carbon share change (pp)", PARIS_ORDER, False)
# checks with other end years (Kyoto: 2012, Paris: 2019)
result_rows += compare_groups(kyoto_table, "Kyoto (check)", "CO2 change check (%)", KYOTO_ORDER, True)
result_rows += compare_groups(kyoto_table, "Kyoto (check)", "Low-carbon share change check (pp)", KYOTO_ORDER, False)
result_rows += compare_groups(paris_table, "Paris (check)", "CO2 change check (%)", PARIS_ORDER, True)
result_rows += compare_groups(paris_table, "Paris (check)", "Low-carbon share change check (pp)", PARIS_ORDER, False)

df_results = pd.DataFrame(result_rows, columns=["Agreement", "Outcome", "Country group", "Talk group", "n",
                                                "Median", "Mean", "Std",
                                                "Share of pairs where talkers improved more",
                                                "Spearman (talk vs outcome)"])
df_results.to_csv(os.path.join(RESULTS_FOLDER, "main_results.csv"), index=False)

print("Windows: Kyoto CO2 %d-%d (check %d), Kyoto electricity %d-%d (check %d), Paris %d-%d (check %d)"
      % (KYOTO_CO2_WINDOW[0], KYOTO_CO2_WINDOW[1], KYOTO_CHECK_END, KYOTO_ELEC_WINDOW[0], KYOTO_ELEC_WINDOW[1],
         KYOTO_CHECK_END, PARIS_START, PARIS_END, PARIS_CHECK_END))
pd.set_option("display.width", 200)
for (agreement, outcome), part in df_results.groupby(["Agreement", "Outcome"], sort=False):
    print("\n---", agreement, "|", outcome)
    print(part.drop(columns=["Agreement", "Outcome"]).round(2).to_string(index=False))

# Simpson's paradox check (Paris, mentioned yes / no): pooled vs split
simpson_rows = []
for group in ["All countries (pooled)"] + COUNTRY_GROUPS:
    if group == "All countries (pooled)":
        sub = paris_table
    else:
        sub = paris_table[paris_table["Country group"] == group]
    for mentioned in [True, False]:
        values = sub[sub["Mentioned climate"] == mentioned]["CO2 change (%)"].dropna()
        simpson_rows.append([group, "Mentioned climate" if mentioned else "Did not mention",
                             values.count(), values.median()])
df_simpson = pd.DataFrame(simpson_rows, columns=["Country group", "Talk", "n", "Median CO2 change (%)"])
df_simpson.to_csv(os.path.join(RESULTS_FOLDER, "paris_simpson_check.csv"), index=False)
print("\nParis, mentioned climate yes/no, pooled vs split (Simpson's paradox check):")
print(df_simpson.round(1).to_string(index=False))


# ===========================================================================
# 10. Figures (Lecture 3: one message per figure, labelled axes with units,
#     colour-blind friendly colours, same colour = same group, no chartjunk)
# ===========================================================================
print("\nMaking figures...")
sns.set_style("ticks")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    fig.savefig(os.path.join(FIGURES_FOLDER, name + ".pdf"))
    fig.savefig(os.path.join(FIGURES_FOLDER, name + ".png"), dpi=200)
    plt.close(fig)


def mark_agreements(ax, label=True):
    for year, name in [(1997, "Kyoto"), (2015, "Paris")]:
        ax.axvline(year, color="black", linestyle="--", linewidth=0.8)
        if label:
            ax.text(year + 0.5, ax.get_ylim()[1] * 0.95, name, va="top", fontsize=9)


# --- Figure 1: climate talk over time
yearly = df_all[df_all["Year"] >= 1970].groupby(["Year", "Country group"])["Climate per 1000 words"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 4))
for group in COUNTRY_GROUPS:
    ax.plot(yearly.index, yearly[group], color=GROUP_COLOURS[group], label=group, linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Climate mentions per 1000 words\n(mean over speeches)")
ax.set_title("Climate change became a UN General Debate topic around Kyoto and grew after Paris")
mark_agreements(ax)
ax.legend(frameon=False)
sns.despine(ax=ax)
save(fig, "fig1_climate_talk_over_time")

# --- Figure 2: distribution of climate talk (skewed -> medians)
values = df_all[df_all["Year"] >= 1990]["Climate per 1000 words"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(values, bins=60, color="#999999")
ax.axvline(values.median(), color="black", linestyle="-", linewidth=1.5, label="Median (%.1f)" % values.median())
ax.axvline(values.mean(), color="black", linestyle=":", linewidth=1.5, label="Mean (%.1f)" % values.mean())
ax.set_xlabel("Climate mentions per 1000 words")
ax.set_ylabel("Number of speeches")
ax.set_title("Climate talk is very skewed: %.0f%% of speeches (1990-2025) never mention it"
             % (100 * (values == 0).mean()))
ax.legend(frameon=False)
sns.despine(ax=ax)
save(fig, "fig2_climate_talk_distribution")

# --- Figure 3: CO2 per capita over time
co2_yearly = df_co2[(df_co2["Year"] >= 1990) & (df_co2["Year"] <= co2_end)].groupby(
    ["Year", "Country group"])["CO2 per capita (t)"].median().unstack()
fig, ax = plt.subplots(figsize=(8, 4))
for group in COUNTRY_GROUPS:
    ax.plot(co2_yearly.index, co2_yearly[group], color=GROUP_COLOURS[group], label=group, linewidth=2)
ax.set_ylim(bottom=0)
ax.set_xlabel("Year")
ax.set_ylabel("CO$_2$ per capita (tonnes)\n(median over countries)")
ax.set_title("Emissions per person fell in developed countries and rose in developing countries")
mark_agreements(ax)
ax.legend(frameon=False)
sns.despine(ax=ax)
save(fig, "fig3_co2_per_capita_over_time")

# --- Figure 4: electricity per person by source (population-weighted, same countries every year)
source_columns = {"Fossil fuels": "Fossil elec per capita (kWh)", "Nuclear": "Nuclear elec per capita (kWh)",
                  "Renewables": "Renewable elec per capita (kWh)"}
# up to PARIS_END, because the population numbers come from the CO2 file (which may end earlier)
elec = df_energy[(df_energy["Year"] >= 2000) & (df_energy["Year"] <= PARIS_END)]
elec = pd.merge(elec, df_co2[["ISO-alpha3 Code", "Year", "Population"]], on=["ISO-alpha3 Code", "Year"], how="left")
elec = elec.dropna(subset=list(source_columns.values()) + ["Population"])
# only countries with data in every year, otherwise the lines jump when countries enter the data
n_years = PARIS_END - 2000 + 1
complete = elec.groupby("ISO-alpha3 Code").size()
elec = elec[elec["ISO-alpha3 Code"].isin(complete[complete == n_years].index)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, group in zip(axes, COUNTRY_GROUPS):
    sub = elec[elec["Country group"] == group]
    years = sorted(sub["Year"].unique())
    stacks = []
    for source, column in source_columns.items():
        per_year = []
        for year in years:
            rows = sub[sub["Year"] == year]
            # population-weighted average = total generation / total population
            per_year.append((rows[column] * rows["Population"]).sum() / rows["Population"].sum())
        stacks.append(per_year)
    ax.stackplot(years, stacks, labels=list(source_columns.keys()),
                 colors=[SOURCE_COLOURS[s] for s in source_columns])
    mark_agreements(ax, label=False)
    ax.set_title("%s (%d countries)" % (group, sub["ISO-alpha3 Code"].nunique()))
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity per person (kWh)")
    sns.despine(ax=ax)
axes[0].legend(loc="lower left", frameon=False)
fig.suptitle("Electricity generation per person by source (population-weighted; dashed line = Paris)")
save(fig, "fig4_electricity_by_source")


# --- Figures 5 and 6: change per talk group (main answer)
def change_boxplots(kyoto_col, paris_col, ylabel, title, name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    for row, group in enumerate(COUNTRY_GROUPS):
        for col, (table, column, order, label) in enumerate(
                [(kyoto_table, kyoto_col, KYOTO_ORDER, "Kyoto (talk in %d-%d)" % tuple(KYOTO_TALK_YEARS)),
                 (paris_table, paris_col, PARIS_ORDER, "Paris (talk in %d-%d)" % tuple(PARIS_TALK_YEARS))]):
            ax = axes[row, col]
            sub = table[table["Country group"] == group].dropna(subset=[column])
            sns.boxplot(data=sub, x="Talk group", y=column, order=order, color=GROUP_COLOURS[group],
                        showfliers=False, width=0.5, ax=ax)
            sns.stripplot(data=sub, x="Talk group", y=column, order=order, color="black", size=3,
                          alpha=0.6, jitter=0.15, ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            counts = [(sub["Talk group"] == t).sum() for t in order]
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([t + "\n(n=%d)" % n for t, n in zip(order, counts)])
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_title(label + " - " + group, fontsize=10)
            sns.despine(ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, name)


change_boxplots("CO2 change (%)", "CO2 change (%)",
                "Change in CO$_2$ per capita (%)",
                "Change in CO$_2$ per capita over the following decade (Kyoto %d-%d, Paris %d-%d)"
                % (KYOTO_CO2_WINDOW[0], KYOTO_CO2_WINDOW[1], PARIS_START, PARIS_END),
                "fig5_co2_change_by_talk")
change_boxplots("Low-carbon share change (pp)", "Low-carbon share change (pp)",
                "Change in low-carbon share of electricity\n(percentage points)",
                "Change in the share of electricity from nuclear + renewables (Kyoto %d-%d, Paris %d-%d)"
                % (KYOTO_ELEC_WINDOW[0], KYOTO_ELEC_WINDOW[1], PARIS_START, PARIS_END),
                "fig6_low_carbon_change_by_talk")

# --- Figure 7: individual developed countries
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, table, label in [(axes[0], kyoto_table, "Kyoto: talk %d-%d, CO$_2$ change %d-%d"
                          % (KYOTO_TALK_YEARS[0], KYOTO_TALK_YEARS[1], KYOTO_CO2_WINDOW[0], KYOTO_CO2_WINDOW[1])),
                         (axes[1], paris_table, "Paris: talk %d-%d, CO$_2$ change %d-%d"
                          % (PARIS_TALK_YEARS[0], PARIS_TALK_YEARS[1], PARIS_START, PARIS_END))]:
    sub = table[table["Country group"] == ANNEX_LABEL].dropna(subset=["CO2 change (%)"])
    ax.scatter(sub["Climate per 1000 words"], sub["CO2 change (%)"], color=GROUP_COLOURS[ANNEX_LABEL])
    for iso, row in sub.iterrows():
        ax.annotate(iso, (row["Climate per 1000 words"], row["CO2 change (%)"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    spearman = sub["Climate per 1000 words"].corr(sub["CO2 change (%)"], method="spearman")
    ax.set_title(label + "\nSpearman correlation = %.2f (n=%d)" % (spearman, len(sub)), fontsize=10)
    ax.set_xlabel("Climate mentions per 1000 words (mean of the two speeches)")
    ax.set_ylabel("Change in CO$_2$ per capita (%)")
    sns.despine(ax=ax)
fig.suptitle("Developed (Annex I) countries: more climate talk did not go with larger CO$_2$ cuts")
fig.tight_layout()
save(fig, "fig7_annex1_countries_scatter")

# --- Figure 8: pooled vs split (Simpson's paradox)
fig, ax = plt.subplots(figsize=(8, 4))
groups = ["All countries (pooled)"] + COUNTRY_GROUPS
x = np.arange(len(groups))
width = 0.38
for i, (talk, colour) in enumerate([("Mentioned climate", "#555555"), ("Did not mention", "#BBBBBB")]):
    part = df_simpson[df_simpson["Talk"] == talk].set_index("Country group").loc[groups]
    bars = ax.bar(x + (i - 0.5) * width, part["Median CO2 change (%)"], width, color=colour, label=talk)
    for bar, n in zip(bars, part["n"]):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + (1 if y >= 0 else -1), "n=%d" % n,
                ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Median change in CO$_2$ per capita (%%)\n%d-%d" % (PARIS_START, PARIS_END))
ax.set_title("Paris: pooling all countries hides that talkers did not cut more within each group")
ax.legend(frameon=False)
sns.despine(ax=ax)
save(fig, "fig8_simpson_pooled_vs_split")

print("Saved figures in", FIGURES_FOLDER, "and tables in", RESULTS_FOLDER)
