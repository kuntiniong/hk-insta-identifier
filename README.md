# Hong Kong Instagram username identifier with Cantonese linguistics

NLP pipeline that identifies Hong Kong Instagram users solely based on Romanized Cantonese phonetic patterns in usernames.

> *disclaimer: i have no formal training in linguistics except i took a computational linguistcs course in my sophomore year; however, this project was created during my **freshman year** as an experimental project for my personal interest in linguistics and machine learning.*

## About

This binary classification project aims to identify Hong Kong Instagram users **without relying on real-time users' metadata from Meta**. I created this end-to-end pipeline that collects raw username data, and then separates the **syllable-based** phoentic pattern from usernames as features. The main objective here is to examine whether only usernames are sufficient enough for ML models to identify Hong Kong users.

An example use case for it could be creating a **lightweight ad bot** that exclusively engages with Hong Kong IG users for social media marketing purposes. (just an example, this violates Meta's policy so don't do this)

Since usernames are not complex and the goal is creating a lightweight bot, simpler traditional models like **Logistic Regression**, **Random Forest** & **SVM** are used for evaluation.

[Training data](datasets) is collected from [hypeauditor.com](https://hypeauditor.com/) using [hypeauditor_scraper.py](hypeauditor_scraper.py) with Selenium.

> *try HK-Insta-Identifier yourself on my [streamlit app](https://hk-insta-identifier.streamlit.app/)!*

## How Does It Work?

The core principle of this classification task revolves around **Romanized Cantonese linguistic features** and the behavior of **NLTK syllable tokenizer**.       

Notably, the NLTK syllable tokenizer is *not* Romanized Cantonese-specific, but it could still provide a workaround by capturing some distinctive patterns.

The following are the visualizations of the ***distribution of repeated and unique syllables***, and the ***top 10 most appeared syllables in Non-HK and HK usernames***:

<img src="images/repeat_pie.png">
<img src="images/freq_chart.png">

> Terminologies:
>
> * **Vowel**: a,e,i,o,u,y* and they *can* be a *standalone syllable*
> * **Consonant**: a character that is not a vowel and *cannot* be a *standalone syllable*
> * **Consonant-vowel (CV) syllable**: a syllable that contains *both* vowels and consonants, e.g. 'fi', 'ha', etc.
> * **Monosyllabic**: a word with single/ one syllable
> 
> *note: y sometimes can act as a vowel as well*

1. **Higher Appearance of Standalone Vowels in HK** -> *Unique Vowel Clusters*
    - Romanized Cantonese has a lot of unique adjacent vowels compared to English or other languages
    - For example: {"張": "ch-***eu***-ng", "楊": "  ***yeu***-ng", "趙": "ch-***iu***", "游": "  ***yau***", ...}
    - The tokenizer is not familiar with these clusters and might treat them as an individual syllable

2.  **Less CV syllables, More Unique Syllables in HK** -> *Complex Consonants Clusters*
    - Romanized Cantonese also has a lot of complex consonants combinations and some can even contain no vowels at all
    - For example : {"翠": "***ts***-ui", "芷": "  ***tsz***", "吳": "***ng***", "郭": "***kw***-ok", ...}
    - This confuses the tokenizer to group the consonants to other vowels, resulting in more unique syllables

3. **Lower Overall Syllable Counts in HK** -> *Monosyllabic Chinese Characters*
    - Hong Kong People's name are mostly made up of 3 Chinese characters, and chinese characters are monosyllabic
    - i.e. Hong Kong people's name at most have 3 syllables and leads to lower overall syllable counts in usernames
    - So the maximum syllable count is around 75 ("ha") in Non-HK while it is only roughly 50 ("i") in HK 

All these differences contributed as the **patterns** for the models to identify HK usernames from non-HK usernames.

## Why Syllables?

In NLP, conventional tokens might be **words, phrases, or subword units**. On the contrary, syllabic tokenization is regarded as a rather "**inconsistent**" tokenization technique since the phonetics in the English language is also inconsistent in some extent, such as the "k" in "knife" or "olo" in "colonel". 

However, I found that syllabic tokenization could still be the **most suitable existing solution** in tokenizing usernames with following reasons:

* **Usernames don't contain whitespaces** 
    - Lack of whitespaces in the IG usernames makes the traditional tokenizers that heavily rely on whitespaces can't work properly

* **Usernames are not sentences**
    - In other words, usernames are too short to extract a "word" as a unit for the features

* **Usernames are not proper English**
    - Any conventional tokenizers won't have the word embeddings for usernames, so a **subword tokenizer** that tokenizes a word based on the **prefixes** and **suffixes** would also *not* work

* **No existing Romanized Cantonese-specific tokenizer**
    - As demonstrated in the last part, syllabic tokenizer can somehow still be able to extract some unique patterns, albeit the lack of Romanized Cantonese word embeddings     

> *learn more in [Forbidden Spellings](https://www.youtube.com/shorts/3ipFdRfFvK4) & [NLP pipeline deep dive: Why doesn't anyone tokenize by syllables?](https://www.youtube.com/watch?v=4_KxnoMnVVs&t=2990s&ab_channel=RachaelTatman)*

## Results

<img src="images/confusion_matrix.png">

After running a GridSearchCV, it was found that both **Logistic Regression (LR)** and **Support Vector Machines (SVM)** yielded the best testing results with **0.742**. On the other hand, Random Forest (RF) with 0.691  showed the worst performance due to potential underfitting.

> *check out [hk_ig_clf.ipynb](hk_ig_clf.ipynb) or [hk_ig_clf.pdf](hk_ig_clf.pdf) for full results*

## Limitations & Conclusion 

Username analysis is a *super super* complicated topic because of the freedom and creativity users have when choosing their usernames, for example:  

- **Private account holders** may choose *not* to include their **government (Cantonese) names** in their usernames 
- **English names** are widely adopted by many Hong Kong users and greatly reduce the visibility of linguistic patterns tied to Cantonese
- There is often **overlap among Romanized Chinese dialects**, making it difficult to distinguish between users from different regions  

So while my approach might seem effective enough(? ), just keep in mind these factors should also be considered when interpreting the results.

