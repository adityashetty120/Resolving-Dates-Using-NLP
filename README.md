# Resolving Dates Using NLP  

**Resolving Dates Using NLP** is a Natural Language Processing project that extracts, classifies, and organizes events from unstructured text into a clean, chronological timeline.  
It combines **spaCy** for linguistic analysis, **Meta LLaMA 3.1** for temporal reasoning, and **Streamlit** for an interactive interface.  

---

## Features  
- **Event Classification**  
  - Classifies sentences as **past/current events**, **future forecasts**, or **non-events** using spaCy.  
- **Date Resolution**  
  - Handles explicit dates (*“March 10, 2024”*) and relative references (*“a week later”*) by converting them into standardized `DD-MM-YYYY` format.  
- **LLM Integration**  
  - Uses Meta LLaMA 3.1 with structured prompting to generate concise event summaries in a JSON timeline.  
- **Interactive Web App**  
  - Streamlit-based app where users can paste raw text and instantly view an ordered timeline of events.  

---

## Tech Stack  
- [spaCy](https://spacy.io/) – NLP parsing & classification  
- [Meta LLaMA 3.1](https://ai.meta.com/research/publications/llama/) (via [lmstudio](https://lmstudio.ai/)) – Temporal reasoning & event summarization  
- [Streamlit](https://streamlit.io/) – Web app interface  
- **Pandas, Regex, JSON** – Data processing and timeline generation  

---

## Example  

**Input Text:**  
The meeting was held on March 10, 2024.
A follow-up was scheduled a week later.

**Output JSON:**  
```json
[
    {
        "date": "10-03-2024",
    "event": "The meeting was held."
  },
  {
      "date": "17-03-2024",
    "event": "A follow-up was scheduled."
  }
]

Timeline Output (Streamlit):
- 10-03-2024: The meeting was held.  
- 17-03-2024: A follow-up was scheduled.  

