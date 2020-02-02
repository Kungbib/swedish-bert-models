# Swedish BERT Models

The National Library of Sweden / KBLab releases three pretrained language models based on BERT and ALBERT. The models are trained on aproximately 15-20GB of text (200M sentences, 3000M tokens) from various sources (books, news, government publications, swedish wikipedia and internet forums) aiming to provide a representative BERT model for Swedish text. A more complete description will be published later on.

The following three models are currently available:

- **bert-base-swedish-cased** (v1) - A BERT trained with the same hyperparameters as first published by Google.
- **bert-base-swedish-cased-ner** *(experimental)* - a BERT fine-tuned for NER using SUC 3.0.
- **albert-base-swedish-cased-alpha** *(alpha)* - A first attempt at an ALBERT for Swedish.

## Files

| **name**                        | **files** |
|---------------------------------|-----------|
| bert-base-swedish-cased         | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/vocab.txt), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/pytorch_model.bin) |
| bert-base-swedish-cased-ner     | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/vocab.txt) [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/pytorch_model.bin) |
| albert-base-swedish-cased-alpha | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/vocab.txt), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/pytorch_model.bin) |

TensorFlow model weights will be released soon.

## Usage

### BERT Base Swedish

A standard BERT base for Swedish trained on a variety of sources. Vocabulary size is ~50k and can be downloaded [here. Models can be downloaded from Huggingface.

```
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased', do_lower_case=True)
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
```


### BERT base fine-tuned for Swedish NER

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

The BERT tokenizer often splits words into multiple tokens, with the subparts starting with `##`, for example the string `Engelbert kör Volvo till Herrängens fotbollsklubb` gets tokenized as `Engel ##bert kör Volvo till Herr ##ängens fotbolls ##klubb`. To "glue" them back together one can use something like this:

```
tokens = nlp('Engelbert kör Volvo till Herrängens fotbollsklubb')

l = []
for token in tokens:
    if token['word'].startswith('##'):
        l[-1]['word'] += token['word'][2:]
    else:
        l += [ token ]
```

Which should result in the following:

```
[ { 'word': 'Engelbert',     'score': 0.99..., 'entity': 'PRS'},
  { 'word': 'Volvo',         'score': 0.99..., 'entity': 'OBJ'},
  { 'word': 'Herrängens',    'score': 0.99..., 'entity': 'ORG'},
  { 'word': 'fotbollsklubb', 'score': 0.99..., 'entity': 'ORG'} ]
```


### ALBERT base

Vocab files (sentencepiece) are located here and here. Tensorflow checkpoint is here and Pytorch-model is here. The easisest way to do this is, again, using Huggingface Transformers:

```
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha', do_lower_case=False)
model = AutoModel.from_pretrained('KB/albert-base-swedish-cased-alpha')
```

## Acknowledgements ❤️

- Resources from Stockholms University, Umeå University and Swedish Language Bank at Gothenburg University was used when fine-tuning BERT for NER.
- Model pretraining was made partly in-house at the KBLab and partly (for material without active copyright) with the support of Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
- Models are hosted on S3 by Huggingface 🤗

