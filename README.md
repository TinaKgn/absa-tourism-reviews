# Aspect-Based Sentiment Analysis of New York Hotel Reviews

## Project Overview

Online reviews are a primary signal for hospitality businesses, but a single rating rarely tells the whole story: a single star rating or overall sentiment score obscures the specific service dimensions that guests actually care about. A guest may rate a hotel four stars overall while explicitly criticising its cleanliness or front desk service — information that gets lost with a traditional approach to sentiment attribution.

This project addresses that gap with a two-stage NLP pipeline for **Aspect-Based Sentiment Analysis (ABSA)**, applied to a stratified sample of 5,000 TripAdvisor reviews (NYC, 2015–2019 and 2022–2023), balanced across VADER review-level sentiment terciles and time periods. The pipeline:

1. **Detects which service aspects are mentioned** in each review using Natural Language Inference (NLI),
2. **Evaluates the sentiment expressed towards each aspect** using a large language model (LLM).

To validate the pipeline and demonstrate its business utility, this repository also includes:
- **Exploratory analysis of ABSA outputs**, including a comparison with VADER review-level sentiment scores as a baseline,
- **An applied use case**: exploring whether complaints about specific aspects co-move with NYC visitation volume data across months.


---

## Key Findings

**Pipeline output**:

- Structured, aspect-level sentiment data that a single overall rating cannot capture.
- The NLI aspect detector identified at least one aspect in all 5,000 reviews, with an
average of 8 aspects per review across 15 service dimensions.
- The most frequently mentioned aspects are accommodation quality, value for money, and
staff friendliness — each appearing in over 4,800 reviews. Crowding, wait time, and
booking availability are the sparsest, with fewer than 450 mentions each.
- Of 39,941 total aspect evaluations, 72% are positive, 18% neutral, and 6% negative.
High overall positivity is consistent with a known ‘positivity bias’ in language and evaluation within tourism reviews.
- ABSA aspect-level sentiment shows broad agreement with VADER review-level scores,
confirming the pipeline is directionally consistent with an established baseline while
providing substantially more granular output.

**Tourism volume comparison (exploratory)**:

The analysis compares negative sentiment proportions per aspect in the busiest 30% of months against the quietest 30%. Focusing on the extremes, rather than a single correlation across all months, makes the contrast more legible with a limited dataset, and reduces the influence of mid-range months where volume differences are small and sentiment signals are noisier.


- Crowding complaints (overcrowding in hotel dining areas, breakfast rooms, lobbies, elevators) shows the clearest co-movement: negative mentions rise from 27% in low-volume months to 49% in high-volume months (+21.9 percentage points).
- Wait time and noise level show no consistent directional signal, possibly because hotels
adequately staff up for peak seasons, or because guest expectations shift during busy periods.
- High-frequency structural aspects — accommodation quality, value for money — show no co-movement with volume, which is expected: these reflect
characteristics of the hotel itself rather than operational pressure from visitor numbers.

The analysis does not attempt to explain these patterns; they are offered as directions for further investigation.

> **Note:** Visitation volume data (monthly NYC visitor counts derived from US Bureau of Transportation Statistics flight data) is sourced from the same upstream repository as the review dataset (see Data and Pre-Processing). The data span includes a pre/post COVID gap (2016–2018 and 2022–2023). Results are descriptive; no causal claims are made.

---

## Notebooks Roadmap

| # | Notebook | Environment | Description |
|---|----------|-------------|-------------|
| 01 | `notebooks/01_ABSA_NLI_Aspect_Detection.ipynb` | Colab GPU recommended | NLI-based aspect detection |
| 02 | `notebooks/02_ABSA_LLM_Sentiment_Evaluation.ipynb` | Colab GPU recommended | LLM sentiment scoring |
| 03 | `notebooks/03_ABSA_Aspect_Sentiment_Analysis.ipynb` | Suitable for running locally | Aspect frequency analysis, sentiment distribution across 15 service dimensions, word clouds for top aspects, and VADER baseline comparison |
| 04 | `notebooks/04_tourism_volume_comparison_example.ipynb` | Suitable for running locally | Tourism volume comparison use case |

---

## Design Decisions

### Why NLI for Aspect Detection?

The pipeline uses a predefined aspect dictionary (covering dimensions such as location, service, cleanliness, amenities, price, and staff) combined with NLI inference to determine whether a given review sentence entails the presence of each aspect. This approach offers several practical advantages:

- **Domain control without labelled data**: aspect coverage is governed by a human-readable dictionary, not a trained classifier, eliminating the need for annotated training sets.
- **Scalability**: new aspects can be added simply by extending the dictionary and defining an entailment template.
- **Robustness to varied phrasing**: NLI handles implicit or indirect aspect mentions that keyword matching would miss.
- **Interpretability**: the entailment decision is transparent and easy for analysts to audit.

### Why an LLM for Sentiment Scoring?

Sentiment analysis of tourism reviews is deceptively difficult. Guests frequently use irony and culture-specific expressions that confuse simpler methods. LLMs handle this complexity by reasoning over the full context of a review text rather than scoring words in isolation.

**Why not VADER?** The dataset includes review-level VADER scores, which serve as a useful sanity-check baseline: the project includes a confusion matrix to confirm broad agreement between VADER's overall sentiment and ABSA's aspect-level outputs. However, VADER is a rule-based lexicon model that assigns a single polarity score to an entire review. It cannot attribute sentiment to a specific aspect, and it struggles with negation, conditional phrasing, and informal tone. It is an appropriate benchmark, but not a substitute for aspect-level analysis.

**Why not a fine-tuned BERT classifier?** Supervised classifiers require labelled training data — in this case, thousands of review sentences manually annotated with both an aspect label and a sentiment label. That data does not exist for this domain, and creating it is out of scope of this project. Additionally, BERT-based classifiers also tend to perform poorly outside the domain they were trained on.

**Why not zero-shot classification alone?** Zero-shot classification (the NLI step) is well-suited to the binary question of whether an aspect is present. It is less reliable for graded sentiment evaluation, where nuance, degree, and context matter considerably. Combining NLI for detection with an LLM for sentiment evaluation plays to the strengths of each approach.

### Why Mistral model?

`mistral-7b-instruct` was selected after evaluating the available open-weight models on two practical criteria: it runs efficiently within Google Colab's GPU memory constraints, and it delivers a strong balance of instruction-following quality and inference speed. Among freely available models, Mistral 7B Instruct consistently produces well-structured, context-aware outputs for classification prompts, making it a good choice for sentiment evaluation at scale.

The model is loaded and served directly on Colab's GPU using the **Hugging Face `transformers` library**. This means inference runs with no external API calls and no token costs.

---

## Checkpointing and Caching

Running LLM inference over thousands of reviews is time- and resource-intensive. The pipeline processes reviews in batches, reducing GPU memory overhead, and makes the workload practical on free-tier hardware like Colab's T4. The pipeline also includes a checkpointing and caching workflow designed to make iterative development practical:

- NLI aspect detection outputs are cached after the first run,
- LLM sentiment results are cached per review,
- Rows already processed are skipped on re-runs.

This significantly reduces redundant computation and total processing time when resuming interrupted sessions.

---

## Data and Pre-Processing

This pipeline assumes the review dataset has already been downloaded and pre-processed. Data preparation is handled by an upstream repository (see [tourism_data_project](https://github.com/TinaKgn/tourism_data_project)), maintained by [David Bienvenue](https://www.linkedin.com/in/davidabienvenue/).

A Google Drive connection is required to run notebooks 01 and 02. The pipeline expects the preprocessed review dataset to be available at a specific path within your Google Drive folder: `tourism_data_project/data/silver/tripadvisor/staged_primary/`.

Notebook 01 automatically saves its NLI aspect detection results to the same Drive folder, where notebook 02 picks them up as input.

---

## Running the Notebooks

**Notebooks 01 and 02** are optimised for **Google Colab GPU** runtime.

**Notebooks 03 and 04** can be run locally, provided the output files from notebooks 01 and 02 are available.

### Validation Steps

1. Open `01_ABSA_NLI_Aspect_Detection.ipynb` in Colab, set GPU runtime, point to the `data/processed` path, and run to completion.
2. Run `02_ABSA_LLM_Sentiment_Evaluation.ipynb` to generate aspect-level sentiment results.
3. Run notebooks 03 and 04 either on Colab or locally using the generated outputs.

---

## Infrastructure

The pipeline is designed to run entirely on **free compute**:
- Google Colab GPU (T4) for the NLP inference steps,
- Mistral 7B Instruct loaded locally via Hugging Face `transformers` for sentiment evaluation.

No paid cloud infrastructure or proprietary APIs are required.

---

## Contact

**Kristina Kogan**<br>
Data Analytics & AI Enablement | AI-Driven Insights

[Connect on LinkedIn](https://www.linkedin.com/in/kristina-kogan/)