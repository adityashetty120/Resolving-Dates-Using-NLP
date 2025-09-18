import spacy

class EventClassifier():
    """A class that classifies a sentence as either having an event or a non event using spaCy.

    Parameters
    ----------
    nlp : spacy.Language
        The spaCy language model to use for processing the text. Defaults to en_core_web_lg.
    debugging : bool
        A flag to enable debugging mode. Defaults to False.
    """
    def __init__(self, nlp=spacy.load("en_core_web_sm"), debugging=False):
        self.stative_verbs = {"be", "get", "see", "locate", "include", "note", "have", "base", "mean"}
        self.intention_or_goal_verbs = {"predict", "indicate", "forecast", "target", "plan", "aim", "intend", "expect", "hope", "propose", "commit", "promise", "signal", "project", "schedule", "decide", "consider", "believe"}
        self.nlp = nlp
        self.debugging = debugging

    def is_event(self, text=None, doc=None) -> dict:
        """
        Determines if a given sentence is an event.
        
        Parameters
        ----------
        text : str
            Text to be classified.
        doc : spacy.tokens.Doc
            The spaCy Doc object of the text.

        Returns
        -------
        result : dict
            A dictionary containing the classification results:
            - label : str
                Label of the sentence, either "event" or "non_event".
            - sublabel : str
                Sublabel of the sentence, either "past_or_current", "forecast_or_prediction", or None.
            - explanation : list
                List containing the explanation of the classification.

        Examples
        --------
        >>> is_event("I am going to the park.")
        {
            'label': 'event',
            'sublabel': 'past_or_current',
            'debug_info': [
                ['text:am, lemma:be, pos:AUX, dep:aux, tag:VBP', 'text:going, lemma:go, pos:VERB, dep:ROOT, tag:VBG'],
                'Action Verb: True, Intention or Goal: False, Modal Verb: False, Stative Verb: False'
            ]
        }
        """
        if doc is None:
            doc = self.nlp(text)
        
        # Flags to identify if the sentence is an event
        has_action_verb = False
        is_intention_or_goal = False
        has_modal_verb = False
        is_stative_verb = False
        debug_info = None
        root_children = set()

        #debugging data
        verbs = []
        data = None

        # Check for verb tenses, types, and modals
        for token in doc:
            if token.pos_ == "VERB" or token.pos_ == "AUX":
                verb_data = f"text:{token.text}, lemma:{token.lemma_}, pos:{token.pos_}, dep:{token.dep_}, tag:{token.tag_}"
                verbs.append(verb_data)

                # Get the root verb and its children
                if token.dep_ == "ROOT":
                    # print(f"text:{token.text}, lemma:{token.lemma_}, pos:{token.pos_}, dep:{token.dep_}, tag:{token.tag_}")
                    root_children = {child.lemma_ for child in token.children}

                    if token.tag_ in {"VBD", "VBG", "VBN", "VBP", "VBZ", "VB"}:
                        has_action_verb = True

                    # Check for stative verbs
                    if token.lemma_ in self.stative_verbs:
                        is_stative_verb = True 

                    # Check for intention or goal-related verbs OR VB root verbs are simple present tense verbs which are considered as intention or goal verbs
                    if token.lemma_ in self.intention_or_goal_verbs or token.tag_ == "VB":
                        is_intention_or_goal = True
                    

                for token in doc:
                    # Check for modal verbs
                    if token.pos_ == "AUX" and token.dep_ == "aux" and token.tag_ == "MD" and token.text in root_children:
                        has_modal_verb = True

                data = f"Action Verb: {has_action_verb}, Intention or Goal: {is_intention_or_goal}, Modal Verb: {has_modal_verb}, Stative Verb: {is_stative_verb}"
                debug_info = [verbs, data] if self.debugging else None

        # Determine if the sentence is an event
        if has_action_verb and not is_intention_or_goal and not has_modal_verb and not is_stative_verb:

            return {"label": "event", "sublabel": "past_or_current", "debug_info": debug_info}
        elif has_action_verb and is_intention_or_goal and not has_modal_verb and not is_stative_verb:
            return {"label": "event", "sublabel": "forecast_or_prediction", "debug_info": debug_info}
        else:
            return {"label": "non_event", "sublabel": None, "debug_info": debug_info}
                
                

    def extract_events(self, texts: list) -> list:
        """
        Extracts events from a list of texts.

        Parameters
        ----------
        texts : list
            List of texts to be classified.

        Returns
        -------
        events : list
            List of events extracted from the texts.

        Examples
        --------
        >>> extract_events(["I am going to the park.", "I will go to the park.", "I am at the park.", "             "])
        [
            {
                'text': 'I am going to the park.',
                'label': 'event',
                'sublabel': 'past_or_current',
                'debug_info': [
                    ['text:am, lemma:be, pos:AUX, dep:aux, tag:VBP', 'text:going, lemma:go, pos:VERB, dep:ROOT, tag:VBG'],
                    'Action Verb: True, Intention or Goal: False, Modal Verb: False, Stative Verb: False'
                ]
            },
            {
                'text': 'I will go to the park.',
                'label': 'event',
                'sublabel': 'forecast_or_prediction',
                'debug_info': [
                    ['text:will, lemma:will, pos:AUX, dep:aux, tag:MD', 'text:go, lemma:go, pos:VERB, dep:ROOT, tag:VB'],
                    'Action Verb: True, Intention or Goal: False, Modal Verb: True, Stative Verb: False'
                ]
            },
            {
                'text': 'I am at the park.',
                'label': 'non_event',
                'sublabel': None,
                'debug_info': None
            },
            {
                'text': '             ',
                'label': 'non_event',
                'sublabel': None,
                'debug_info': None
            }
        ]
        """
        docs = self.nlp.pipe(texts, n_process=-1)
        events = []
        for text, doc in zip(texts, docs):
            result = self.is_event(doc=doc)
            result["text"] = text
            events.append(result)

        return events

if __name__ == "__main__":
    # Create an instance of the EventClassifier class
    classifier = EventClassifier()

    # Test the is_event method
    print(classifier.is_event("I am going to the park."))
    print(classifier.is_event("I will go to the park."))
    print(classifier.is_event("I am at the park."))
    print(classifier.is_event("             "))
    print(classifier.is_event("We are targeting a deadline of next year."))

    # Test the extract_events method
    print(classifier.extract_events(["I am going to the park.", "I will go to the park.", "I am at the park.", "             "]))