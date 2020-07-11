# Swedish BERT Models

**Update 2020-02-26: Huggingface BERT-base and NER are updated**

**Update 2020-04-27: Added some comparisons to NER performance of this and other BERTs**

**Update 2020-07-03: You can now [cite](#citation-in-bibtex-format) us!**

**Update 2020-07-11: TF models available through Huggingface Transformers**

The National Library of Sweden / KBLab releases three pretrained language models based on BERT and ALBERT. The models are trained on aproximately 15-20GB of text (200M sentences, 3000M tokens) from various sources (books, news, government publications, swedish wikipedia and internet forums) aiming to provide a representative BERT model for Swedish text. A more complete description is available [here](https://arxiv.org/abs/2007.01658).

The following three models are currently available:

- **bert-base-swedish-cased** (*v1.1*) - A BERT trained with the same hyperparameters as first published by Google.
- **bert-base-swedish-cased-ner** (*experimental*) - a BERT fine-tuned for NER using SUC 3.0.
- **albert-base-swedish-cased-alpha** (*alpha*) - A first attempt at an ALBERT for Swedish.

All models are cased and trained with whole word masking.

## Files

| **name**                        | **files** |
|---------------------------------|-----------|
| bert-base-swedish-cased         | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/vocab.txt), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/pytorch_model.bin), [TF checkpoint](https://data.kb.se/datasets/2020/01/tf/bert_base_swedish_cased-v1.1.tar) |
| bert-base-swedish-cased-ner     | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/vocab.txt) [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/pytorch_model.bin) |
| albert-base-swedish-cased-alpha | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/config.json), [sentencepiece model](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/spiece.model), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/pytorch_model.bin), [TF checkpoint](https://data.kb.se/datasets/2020/01/tf/albert_base_swedish_cased.tar) |

## Usage requirements / installation instructions

The examples below require Huggingface Transformers 2.4.1 and Pytorch 1.3.1 or greater. For Transformers<2.4.0 the tokenizer must be instantiated manually and the `do_lower_case` flag parameter set to `False` and `keep_accents` to `True` (for ALBERT).

To create an environment where the examples can be run, run the following in an terminal on your OS of choice.

```
# git clone https://github.com/Kungbib/swedish-bert-models
# cd swedish-bert-models
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install -r requirements.txt
```

On some platforms, notably MacOSX < 10.15, you may have to install a Rust compiler for Transformers to install.

### BERT Base Swedish

**UPDATE**: for Transformers==2.5.0 add the parameter `use_fast=False` to `AutoTokenizer.from_pretrained(...)` to retain accented characters such as Å, Ä and Ö.

A standard BERT base for Swedish trained on a variety of sources. Vocabulary size is ~50k. Using Huggingface Transformers the model can be loaded in Python as follows:

```python
from transformers import AutoModel,AutoTokenizer,TFAutoModel

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

# Using TF models
model = TFAutoModel.from_pretrained('KB/bert-base-swedish-cased')

```


### BERT base fine-tuned for Swedish NER

This model is fine-tuned on the SUC 3.0 dataset. Preliminary evaluation (F1) compared other BERTs are as follows:

| **model**  | **PER** | **ORG** | **LOC** | **TME** | **MSR** | **WRK** | **EVN** | **OBJ** | **AVG** |
|---------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| AF-AI      |  0.913  | 0.780 | 0.913 | 0.655 | 0.828 | 0.596 | 0.716 | 0.710 | 0.898 |
| BotXO      |  -- | -- | -- | -- | -- | -- | -- | -- | 0.899 | 
| BERT Multi |  0.945  | 0.834 | 0.942 | 0.888 | 0.853 | 0.631 | 0.792 | 0.761 | 0.906 |
| KB-BERT    | **0.961** | **0.884** | **0.958** | **0.906** | **0.890** | **0.720** | **0.834** | **0.770** | **0.928** |

Using the Huggingface pipeline the model can be easily instantiated. For Transformer<2.4.1 it seems the tokenizer must be loaded separately to disable lower-casing of input strings:

```python
from transformers import pipeline,TFBertForTokenClassification

nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')

nlp('Kalle och Pelle startar firman Kalle och Pelle.')

# Specifically using Tensorflow

tf = TFBertForTokenClassification.from_pretrained('KB/bert-base-swedish-cased-ner')
nlp = pipeline('ner', model=tf, tokenizer='KB/bert-base-swedish-cased-ner')
```

Running the Python code above should produce in something like the result below. Note that the model disambiguates between the names of the persons and the name of the company.

```python
[ { 'word': 'Kalle', 'score': 0.9998126029968262, 'entity': 'PER' },
  { 'word': 'Pelle', 'score': 0.9998126029968262, 'entity': 'PER' },
  { 'word': 'Kalle',   'score': 0.9814832210540771, 'entity': 'ORG' }
  { 'word': 'och',   'score': 0.9814832210540771, 'entity': 'ORG' }
  { 'word': 'Pelle',   'score': 0.9814832210540771, 'entity': 'ORG' } ]
```

Entity types used are `TME` for time, `PRS` for personal names, `LOC` for locations, `EVN` for events and `ORG` for organisations. These labels are subject to change.

The BERT tokenizer often splits words into multiple tokens, with the subparts starting with `##`, for example the string `Engelbert kör Volvo till Herrängens fotbollsklubb` gets tokenized as `Engel ##bert kör Volvo till Herr ##ängens fotbolls ##klubb`. To glue parts back together one can use something like this:

```python
text = 'Engelbert tar sin Rolls-Royce till Tele2 Arena för att titta på Djurgården IF ' +\
       'som spelar fotboll i VM klockan två på kvällen.'

l = []
t = nlp(text)
in_word=False
nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner', ignore_labels=[])
for i,token in enumerate(t):
    if token['entity'] == 'O':
        in_word = False
        continue

    if token['word'].startswith('##'):
        # deal with (one level of) orphaned ##-tokens
        if not in_word:
            l += [ t[i-1] ]
            l[-1]['entity'] = token['entity']
        
        l[-1]['word'] += token['word'][2:]
    else:
        l += [ token ]

    in_word = True

print(l)
```

Which should result in the following (though less cleanly formated):

```python
[ { 'word': 'Engelbert',     'score': 0.99..., 'entity': 'PRS'},
  { 'word': 'Rolls',         'score': 0.99..., 'entity': 'OBJ'},
  { 'word': '-',             'score': 0.99..., 'entity': 'OBJ'},
  { 'word': 'Royce',         'score': 0.99..., 'entity': 'OBJ'},
  { 'word': 'Tele2',         'score': 0.99..., 'entity': 'LOC'},
  { 'word': 'Arena',         'score': 0.99..., 'entity': 'LOC'},
  { 'word': 'Djurgården',    'score': 0.99..., 'entity': 'ORG'},
  { 'word': 'IF',            'score': 0.99..., 'entity': 'ORG'},
  { 'word': 'VM',            'score': 0.99..., 'entity': 'EVN'},
  { 'word': 'klockan',       'score': 0.99..., 'entity': 'TME'},
  { 'word': 'två',           'score': 0.99..., 'entity': 'TME'},
  { 'word': 'på',            'score': 0.99..., 'entity': 'TME'},
  { 'word': 'kvällen',       'score': 0.54..., 'entity': 'TME'} ]
```

### ALBERT base

The easisest way to do this is, again, using Huggingface Transformers:

```python
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha'),
model = AutoModel.from_pretrained('KB/albert-base-swedish-cased-alpha')
```

## Acknowledgements ❤️

- Resources from Stockholms University, Umeå University and Swedish Language Bank at Gothenburg University were used when fine-tuning BERT for NER.
- Model pretraining was made partly in-house at the KBLab and partly (for material without active copyright) with the support of Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
- Models are hosted on S3 by Huggingface 🤗

## Citation in BibTex format
If you wish to reference this work, please use to following:
```
@misc{swedish-bert,
   Author = {Martin Malmsten and Love Börjeson and Chris Haffenden},
   Title = {Playing with Words at the National Library of Sweden -- Making a Swedish BERT},
   Year = {2020},
   Eprint = {arXiv:2007.01658},
}
```

