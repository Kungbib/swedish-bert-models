# Swedish BERT Models

The National Library of Sweden / KB lab releases three pretrained language models based on BERT and ALBERT. The models are trained on aproximately 15-20GB of text (200M sentences, 3000M tokens) from various sources (books, news, government publications, swedish wikipedia and internet forums) aiming to provide a representative BERT model for Swedish text. A more complete description will be published later on.

- *bert-base-swedish-cased* (v1) - A BERT trained with the same hyperparameters as first published by Google.
- *bert-base-swedish-cased-ner* (experimental) - a BERT fine-tuned for NER using SUC 3.0.
- *albert-base-swedish-cased-alpha* (alpha) - A first attempt at an ALBERT for Swedish.

## BERT Base Swedish



## BERT base fine-tuned for Swedish NER

This model is fine-tuned on the SUC 3.0 dataset. Using the Huggingface pipeline the model can be easily instantiated. However, it seems the tokenizer must be loaded separately to disable lower-casing of input strings:

```
from transformers import BertTokenizer,pipeline

tok = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased-ner', do_lower_case=False)
nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer=tok)

nlp('Idag släpper KB tre språkmodeller.')
```

Running the Python code above should results in something like

```
[{'word': 'Idag', 'score': 0.9998126029968262, 'entity': 'TME'}, {'word': 'KB', 'score': 0.9814832210540771, 'entity': 'ORG'}]
```

The BERT tokenizer sometimes splits strings into multiple tokens, to "glue" them back together one can use the following:

```


```

## ALBERT base

Vocab files (sentencepiece) are located here and here. Tensorflow checkpoint is here and Pytorch-model is here. The easisest way to do this is, again, using Huggingface Transformers:

```
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha')
model = AutoModel.from_pretrained('KB/albert-base-swedish-cased-alpha')
```

