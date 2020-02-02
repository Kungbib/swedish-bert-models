# Swedish BERT Models

The National Library of Sweden / KB lab releases three pretrained language models based on BERT and ALBERT. The models are trained on aproximately 15-20GB of text (200M sentences, 3000M tokens) from various sources (books, news, government publications, swedish wikipedia and internet forums) aiming to provide a representative BERT model for Swedish text. A more complete description will be published later on.

The following three models are currently available:

- **bert-base-swedish-cased** (v1) - A BERT trained with the same hyperparameters as first published by Google.
- **bert-base-swedish-cased-ner** *(experimental)* - a BERT fine-tuned for NER using SUC 3.0.
- **albert-base-swedish-cased-alpha** *(alpha)* - A first attempt at an ALBERT for Swedish.


## BERT Base Swedish

A standard BERT base for Swedish trained on a variety of sources. Vocabulary size is ~50k and can be downloaded here. Models can be downloaded from Huggingface.

```
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
```


## BERT base fine-tuned for Swedish NER

This model is fine-tuned on the SUC 3.0 dataset. Using the Huggingface pipeline the model can be easily instantiated. However, it seems the tokenizer must be loaded separately to disable lower-casing of input strings:

```
from transformers import BertTokenizer,pipeline

tok = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased-ner', do_lower_case=False)
nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer=tok)

nlp('Idag släpper KB tre språkmodeller.')
```

Running the Python code above should produce in something like the result below. Entity types used are `TME` for time, `PRS` for personal names, `LOC` for locations and `ORG` for organisations. These labels are subject to change.

```
[ { 'word': 'Idag', 'score': 0.9998126029968262, 'entity': 'TME' },
  { 'word': 'KB',   'score': 0.9814832210540771, 'entity': 'ORG' } ]
```

The BERT tokenizer often splits words into multiple tokens, with the subparts starting with `##`, for example the string `Engelbert kör Volvo` gets tokenized to `Engel ##bert kör Volvo`. To "glue" them back together one can use something the following:

```
tokens = nlp('Engelbert kör Volvo.')

l = []
for token in tokens:
    if token['word'].startswith('##'):
        l[-1]['word'] += token['word'][2:]
    else:
        l += [ token ]
```

Which should result in the following:

```
[ { 'word': 'Engelbert', 'score': 0.9997760057449341, 'entity': 'PRS'},
  { 'word': 'Volvo',     'score': 0.9970706105232239, 'entity': 'OBJ'}]
```


## ALBERT base

Vocab files (sentencepiece) are located here and here. Tensorflow checkpoint is here and Pytorch-model is here. The easisest way to do this is, again, using Huggingface Transformers:

```
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha')
model = AutoModel.from_pretrained('KB/albert-base-swedish-cased-alpha')
```

