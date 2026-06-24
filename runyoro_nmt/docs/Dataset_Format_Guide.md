# Dataset Format Guide for Runyoro-English Translation Model

## Purpose

This document explains how training data should be structured to produce a translation model that is **context-aware** and applies **correct grammar rules** in both Runyoro-Rutooro and English.

---

## 1. Core Principle

Every row in the dataset must be:

| Runyoro-Rutooro | English |
|-----------------|---------|
| One complete, natural sentence or phrase | Its exact equivalent in the other language |

The model learns by seeing **thousands of examples** of how one language maps to another. It picks up grammar, tense, word order, and context **from the patterns in the data itself** — not from labels or annotations.

---

## 2. What GOOD Training Data Looks Like

### 2.1 Complete Sentence Pairs (Best for Grammar & Context)

These teach the model tense, subject-verb agreement, word order, idioms, and context:

| Runyoro | English |
|---------|---------|
| Abaana nibasome | The children are studying |
| Nyineeka afeera aka ye' | The house owner dies for his home |
| Tukaija kare bulaijo | We will come early tomorrow |
| Omukazi akagura ebitooke omu katale | The woman bought bananas at the market |
| Enkuba neyija kutonnyera ijo | It was going to rain yesterday |
| Ekyaro kyaitu kikonyera abanyakuli omukwetaaga | Our village has always helped those who are in need |

**Why this works:** The model sees "Tukaija" paired with "We will come" many times across different sentences and learns that "Tukaija" = future tense, first person plural.

### 2.2 Short Phrase Pairs (Good for Vocabulary in Context)

| Runyoro | English |
|---------|---------|
| ensi yaitu | our country |
| omusiri gwaitu | our garden |
| emiti mingi | many trees |
| omuhendo murungi | a good price |
| entebe ntukura | a red chair |

**Why this works:** The model learns adjective-noun agreement and possessive structures.

### 2.3 Multi-Word Expressions & Idioms

| Runyoro | English |
|---------|---------|
| Okugwa omu mahanga | To fall into trouble |
| Okubiika omutima | To be patient |
| Enkoni n'egusha | Discipline corrects |

**Why this works:** Idioms cannot be translated word-by-word. The model learns them as whole units.

---

## 3. What BAD Training Data Looks Like

### 3.1 ❌ Dictionary Definitions (NOT translations)

| Runyoro | English |
|---------|---------|
| fuuta | To fill mouth with food |
| jabuka | To be chipped, broken |
| jaagira | To waddle, straddle, walk with legs wide open due to pain |

**Why this is bad:** "To fill mouth with food" is a **definition**, not a translation. Nobody says "To fill mouth with food" in normal English. The model learns to output "To [verb]..." for everything.

### 3.2 ❌ Grammar Annotations Mixed In

| Runyoro | English |
|---------|---------|
| jabuka (v.i.) | To be chipped, broken. Oku- |
| hya, Oku-, v.t. | To get cooked, get ripe, scalded |

**Why this is bad:** The model treats "(v.i.)", "Oku-" as part of the translation and may output them.

### 3.3 ❌ Multiple Translations in One Cell

| Runyoro | English |
|---------|---------|
| kujaakaarra; emibu ejaakaaza omuswija | The spread of HIV/AIDS is...; Mosquitoes spread malaria; The spread of HIV/AIDS is due to... |

**Why this is bad:** The model cannot learn which Runyoro maps to which English when multiple unrelated items are jammed together.

### 3.4 ❌ Single Words Without Context

| Runyoro | English |
|---------|---------|
| ekitabu | Book |
| omuti | Tree |
| amazzi | Water |

**Why this is limited:** Single words alone don't teach grammar. The model won't learn how "ekitabu" behaves in a sentence (e.g., "Ekitabu kyange" = "My book" vs "Ebitabu byange" = "My books").

**Better version:**

| Runyoro | English |
|---------|---------|
| Ekitabu kyange kiri ha meeza | My book is on the table |
| Ebitabu byange nibingi | My books are many |
| Amata maingi | A lot of milk |
| Amazzi gaitu | Our water |

---

## 4. How to Build Grammar Awareness

The model learns grammar from **seeing patterns repeat** across many examples. Here's how to ensure your data teaches specific grammar concepts:

### 4.1 Tense Coverage

Include the **same verb/concept** in multiple tenses:

| Runyoro | English | Tense |
|---------|---------|-------|
| Ninkoora | I am working | Present Continuous |
| Nkakoora | I worked | Past Simple |
| Ninkaija kukoora | I will work | Future Simple |
| Nakooire | I have worked | Present Perfect |
| Nkaba ninkoora | I was working | Past Continuous |

*Note: The "Tense" column is for YOUR reference only — it should NOT be in the training data.*

### 4.2 Noun Class Agreement

Runyoro has noun classes that affect prefixes. Include examples showing agreement:

| Runyoro | English |
|---------|---------|
| Omwana murungi | The child is good |
| Abaana barungi | The children are good |
| Ekitabu kirungi | The book is good |
| Ebitabu birungi | The books are good |
| Omuti murungi | The tree is good |
| Emiti mirungi | The trees are good |

### 4.3 Subject-Verb Agreement

| Runyoro | English |
|---------|---------|
| Ninkoora | I am working |
| Nukoora | You are working |
| Nakoora | He/she is working |
| Tukoora | We are working |
| Mukoora | You (plural) are working |
| Bakoora | They are working |

### 4.4 Questions and Negation

| Runyoro | English |
|---------|---------|
| Orikukora ki? | What are you doing? |
| Oryakugenda he? | Where are you going? |
| Tirukukoora | I am not working |
| Tatakagendayo | He/she will not go there |

### 4.5 Diverse Domains

Include sentences from multiple domains so the model doesn't just learn agriculture or religion:

- Daily conversation
- Agriculture
- Health
- Education
- Commerce/Business
- Culture/Tradition
- Government
- Nature/Environment
- Technology
- Family/Relationships

---

## 5. Dataset File Format

### Recommended Format: CSV or XLSX

Use two columns with a header row:

```
Runyoro,English
"Abaana nibasome","The children are studying"
"Tukaija kare bulaijo","We will come early tomorrow"
```

Or in Excel (XLSX) format with columns:

| Column A: Runyoro | Column B: English |
|-------------------|-------------------|

### Rules for Each Cell:

1. **One sentence or phrase per cell** — never multiple sentences separated by semicolons
2. **No annotations** — no (v.i.), (n.), Adv., Oku-, etc.
3. **No numbering** — no "1.", "2)", "a)" at the start
4. **Natural text only** — write how a person would actually speak or write
5. **Consistent punctuation** — use periods at the end of complete sentences
6. **No HTML or special formatting** — plain text only

---

## 6. Minimum Dataset Size Guidelines

| Dataset Size | Expected Quality |
|-------------|------------------|
| < 5,000 pairs | Poor — model memorizes, doesn't generalize |
| 5,000 – 15,000 pairs | Fair — handles common patterns |
| 15,000 – 50,000 pairs | Good — generalizes to new sentences |
| 50,000 – 200,000 pairs | Very good — handles nuance and rare constructions |
| > 200,000 pairs | Excellent — near-human quality on common domains |

**Current dataset: ~25,800 clean pairs** — in the "Good" range but more data will improve quality significantly.

---

## 7. Quality Checklist Before Training

Before adding data to the training pipeline, verify:

- [ ] Each row has exactly one Runyoro text and one English text
- [ ] Both sides are complete (not fragments or cut-off mid-sentence)
- [ ] No dictionary formatting ("To + verb" definitions)
- [ ] No grammar annotations or metadata
- [ ] No multiple translations crammed into one cell
- [ ] The Runyoro side is actually Runyoro (not Luganda, Rukiga, or Nyankole)
- [ ] The English side is natural English (not robotic or overly literal)
- [ ] Sentences represent diverse tenses and grammatical structures
- [ ] Data covers multiple subject domains
- [ ] No excessive duplicates (same pair repeated many times)

---

## 8. How to Contribute More Data

The most valuable new data to add:

1. **Conversational sentences** — everyday dialogue
2. **Proverbs with translations** — not definitions, but meaning-equivalent English proverbs or explanations
3. **News-style sentences** — formal register
4. **Instructions/commands** — imperative mood
5. **Questions** — interrogative forms
6. **Complex sentences** — relative clauses, conditionals ("If... then...")
7. **Sentences with pronouns** — he/she/they/we in various positions

Each new pair you add makes the model slightly better at understanding that pattern.

---

## Summary

| Do This | Not This |
|---------|----------|
| `tulima muhogo mumusiri → We grow cassava in our garden` | `muhogo → Cassava` |
| `Abaana nibasome → The children are studying` | `okusome (v.t.) → To study, read` |
| `Nyineeka afeera aka ye' → The house owner dies for his home` | `nyineeka → owner; afeera → dies; aka ye' → for his home` |
| One sentence per row | Multiple translations separated by semicolons |
| Natural speech | Dictionary metalanguage |
