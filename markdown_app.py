import streamlit as st
import spacy
from event_classifier import EventClassifier
import pandas as pd
import lmstudio as lms
import re
import json

# Load the English language model (you might need to download it first)
try:
    nlp = spacy.load("en_core_web_sm")
    classifier = EventClassifier(nlp=nlp, debugging=False)
    model = lms.llm("meta-llama-3.1-8b-instruct")
    
except OSError:
    st.warning("Downloading en_core_web_sm language model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def split_into_sentences(text):
    """Splits a passage of text into sentences using spaCy."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def split_passage(passage):
    sentences = split_into_sentences(passage)
    labels = []


    for sentence in sentences:
        if classifier.is_event(sentence)['sublabel'] == 'past_or_current':
            labels.append('event')
        elif classifier.is_event(sentence)['sublabel'] == 'forecast_or_prediction':
            labels.append('forecast')
        else:
            labels.append('non_event')

    
    split_sentences_df = pd.DataFrame({
        'text': sentences,
        'label': labels
    })

    return split_sentences_df


def tag_passage(split_sentences_df):
    passage = ""
    for index, row in split_sentences_df.iterrows():
        text = row['text'].strip("\n").strip().strip("\n")
        if row['label'] == 'event' or row['label'] == 'forecast':
            passage += "<EVENT> " + text + " </EVENT> "
            passage += "\n"
        else:
            passage += text
            passage += "\n"
    return passage



def generate_response(tagged_passage):
    timeline_prompt = """Generate a chronological timeline of events as a JSON object based on the provided text.

You will be given a `Passage` containing descriptions of events, often enclosed in `<EVENT></EVENT>` tags. Your task is to extract the key events, determine their specific dates in DD-MM-YYYY format, summarize the event concisely, and present them as a chronologically ordered JSON array.

# Steps

1.  **Identify Potential Events:** Scan the entire `Passage`, paying close attention to text within `<EVENT></EVENT>` tags, but also considering event descriptions outside tags if they are clearly linked.
2.  **Extract Dates:** Identify all explicit date mentions (e.g., "March 15, 2023", "April 10") and relative date mentions (e.g., "the next morning", "two days later", "a week later", "three months later").
3.  **Determine Full Dates:**
    *   Convert all dates into DD-MM-YYYY format.
    *   For relative dates, calculate the specific date based on the context and previously established dates within the passage. If a reference date (like a publication date) is separately provided, use it *only* if no other context is available within the passage.
    *   For imprecise dates mentioning only month and year (e.g., "in July 2023"), assume the date is the 1st of that month (e.g., 01-07-2023).
4.  **Summarize Events:** For each determined date, concisely summarize the core action or event that occurred on that date. This may involve extracting key verbs and objects from the sentence describing the event. Sometimes, a single `<EVENT>` tag might contain multiple distinct events occurring on different dates; these should be treated separately.
5.  **Chronological Ordering:** Arrange the identified events strictly by date, from earliest to latest.
6.  **Format Output:** Construct a JSON array where each object represents an event. Ensure the array is ordered chronologically by date.

# Output Format

Output a single JSON array. Each object within the array should represent one event and contain the following two keys:
*   `date`: A string representing the event date in DD-MM-YYYY format.
*   `event`: A string containing the concise summary of the event.

The JSON array itself must be ordered chronologically based on the `date` field of the objects. Do not wrap the JSON output in markdown code blocks.

Example JSON Structure:
```json
[
{
    "date": "[Date of first event in DD-MM-YYYY]",
    "event": "[Summarized description of first event]"
},
{
    "date": "[Date of second event in DD-MM-YYYY]",
    "event": "[Summarized description of second event]"
}
]
```

# Example

*Input Passage:*
```
<EVENT> The robbery took place on the night of Wednesday, March 15, 2023, at around 11:45 PM. </EVENT>
<EVENT> Witnesses reported seeing a masked individual entering the bank on 5th Avenue just minutes before midnight. </EVENT>
The robery was a pretty quick and influential.
<EVENT> The police began their investigation the next morning, on Thursday, March 16, and released CCTV footage two days later, on Saturday, March 18. </EVENT>
<EVENT> A suspect was arrested a week later, on Thursday, March 23, and the trial commenced on Monday, April 10. </EVENT>
<EVENT> The final verdict was delivered three months later, in July 2023. </EVENT>
```

*Output:*
```json
[
{
    "date": "15-03-2023",
    "event": "A masked individual entered the bank on 5th Avenue."
},
{
    "date": "16-03-2023",
    "event": "The police began their investigation."
},
{
    "date": "18-03-2023",
    "event": "The police released CCTV footage."
},
{
    "date": "23-03-2023",
    "event": "The police arrested a suspect."
},
{
    "date": "10-04-2023",
    "event": "The trial commenced."
},
{
    "date": "01-07-2023",
    "event": "The final verdict was delivered."
}
]
```
IMPORTANT: Ensure all property names and string values are enclosed in DOUBLE QUOTES.
Do not use single quotes in the JSON structure.
Each object in the array must follow strict JSON format with keys and values properly quoted.

# Notes

*   Focus on summarizing the core event associated with each specific date.
*   Calculate relative dates based on the nearest preceding date context within the passage.
*   For dates specified only by month and year, use the 1st day of the month (DD=01).
*   The final JSON array must be sorted chronologically by date.
*   Internally, first identify the event details and context, then determine the precise date, and *finally* construct the JSON object and place it correctly in the chronologically ordered array.
"""

    from datetime import datetime

    # Get the current date and weekday
    current_date = datetime.today().strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
    current_weekday = datetime.today().strftime("%A")  # Get full weekday name

    # Define the input message
    input_message = f"""Publication Date: {current_date}
    Publication Weekday: {current_weekday}
    Passage:
    {tagged_passage}"""

    timeline_input_prompt = timeline_prompt + input_message
    timeline_model_output = model.respond(timeline_input_prompt)
    
    return timeline_model_output



def main():
    st.title("Timeline Generator")

    passage = st.text_area("Enter your passage of text:", height=200)

    if passage:
        split_sentences_df = split_passage(passage)

        tagged_passage = tag_passage(split_sentences_df)
        
        timeline_output = generate_response(tagged_passage)
        timeline_output = timeline_output.content
        match = re.search(r'```json(.*?)```', timeline_output, re.DOTALL)

        if match:
            extracted_json = match.group(1).strip()
            
            timeline = json.loads(extracted_json)

        extracted_and_rephrased_events_df = pd.DataFrame(timeline)
        extracted_and_rephrased_events_df['date'] = pd.to_datetime(extracted_and_rephrased_events_df['date'], format='%d-%m-%Y')
        extracted_and_rephrased_events_df = extracted_and_rephrased_events_df.sort_values(by='date')

        st.subheader("***Timeline:***")
        for index, row in extracted_and_rephrased_events_df.iterrows():
            date = row['date'].strftime('%d-%m-%Y')
            rephrased_event = row['event']                                                                               
            st.write(f"- {date}: {rephrased_event}")
    else:
        st.warning("Please enter some text to generate timeline.")


if __name__ == "__main__":
    main()