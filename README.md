
# Mission Bhashini: ISHAAN and VIDYAAPATI Projects

## Overview

As part of **Mission Bhashini**, we are working on two major Machine Translation projects: **ISHAAN** and **VIDYAAPATI**. These projects focus on creating high-quality parallel corpora and robust bi-directional Neural Machine Translation (NMT) models for various Indian languages, enabling seamless translation between English, Hindi, and regional languages.

- **ISHAAN** focuses on developing translation models for translation between English, Hindi, and North-East Indian languages.
- **VIDYAAPATI** addresses translation needs for Hindi and several Indo-Aryan languages.

### Resources

- **Datasets**: Parallel corpora generated in these projects cover multiple languages and domains, offering a robust foundation for developing and fine-tuning Machine Translation (MT) models.
- **Pre-trained Machine Translation Models**: The models are designed to handle diverse translation tasks and can be used directly or fine-tuned for specific applications.

#### Resource Links
- [**Datasets**](https://www.cfilt.iitb.ac.in/bhashini_deployments/)
- [**Pre-trained Machine Translation Models**](https://www.cfilt.iitb.ac.in/bhashini_deployments/)


---

## ISHAAN: Language Pairs 

The ISHAAN project covers the following language pairs:

| Language Pair         |
|-----------------------|
| English-Assamese      |
| English-Bodo          |
| English-Manipuri      | 
| English-Nepali        | 
| Hindi-Manipuri        | 
| Assamese-Bodo         |


---

## VIDYAAPATI: Language Pairs

The VIDYAAPATI project covers the following language pairs:

| Language Pair         |
|-----------------------|
| Hindi-Bengali         |
| Hindi-Konkani         |
| Hindi-Maithili        |
| Hindi-Marathi         |

---

## Consortia Members

The following prestigious institutes are collaborating on these projects:

### ISHAAN Project:
- IIT Bombay
- Gauhati University
- IIIT Manipur
- NIT Silchar
- North Bengal University

### VIDYAAPATI Project:
- IIT Bombay
- IIT Patna
- Goa University
- Jadavpur University
- C-DAC Pune
- Jawaharlal Nehru University (JNU)
- Indian Statistical Institute (ISI), Kolkata


---
## Usage 
### Requirements
`pip3 install ctranslate2 mosestokenizer indic-nlp-library codecs subword-nmt`

### Inference

We provide two example inference scripts that use different tokenization schemes. 
- [**Hindi to Konkani**](scripts/translate-hi-ko.py): This script uses bpe tokenizer.
- [**English to Manipuri**](scripts/translate-en-mn.py): This script uses spm tokenizer.




---

## License
The resources in this repository are released under the Creative Commons Attribution 4.0 International License (CC-BY 4.0). This license allows for sharing and adaptation of the material, provided appropriate credit is given.



Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[![CC BY 4.0][cc-by-image]][cc-by]


---




## Contributions

We welcome contributions to this repository. Feel free to raise issues or submit pull requests.

---

## Contact

For any inquiries, reach out to our team at [nltmbhashini@gmail.com](mailto:nltmbhashini@gmail.com).
