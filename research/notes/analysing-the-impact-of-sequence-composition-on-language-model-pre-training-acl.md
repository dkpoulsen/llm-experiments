---
title: Analysing The Impact of Sequence Composition on Language Model Pre-Training
  - ACL Anthology
id: analysing-the-impact-of-sequence-composition-on-language-model-pre-training-acl
tags:
- cmp170hx-training-feasibility-a0d4fc
- sequence-packing
- data-curation
- determinism
- long-context
- thin-landing-page
created: '2026-10-27T10:58:21.846365Z'
updated: '2026-10-27T19:00:31.379411Z'
source: https://doi.org/10.18653/v1/2024.acl-long.427
source_domain: doi.org
fetched_at: '2026-10-27T10:58:21.225125Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'ACL Anthology landing page for Zhao et al., ''Analysing The Impact of Sequence
  Composition on Language Model Pre-Training'' (ACL 2024 Long Papers, pp. 7897-7912,
  DOI 10.18653/v1/2024.acl-long.427). Abstract only — full text is in the arXiv primary
  (2402.13991) fetched separately. Core claims: standard fixed-length concatenation
  with causal masking lets tokens attend to previous documents, injecting distracting
  context; intra-document causal masking conditions each token only on prior tokens
  of the same document and significantly improves LM and downstream performance; and
  the retrieval-based packing method BM25Chunk improves in-context learning (+11.6%),
  knowledge memorisation (+9.8%) and context utilisation (+7.2%) without sacrificing
  efficiency. Directly relevant to the repo''s data-pipeline packing/determinism claims:
  packing strategy is a first-order quality variable, not a neutral implementation
  detail.'
doi: 10.18653/v1/2024.acl-long.427
utility_score: 14.0
citation_count: 1
venue: Annual Meeting of the Association for Computational Linguistics (ACL)
is_retracted: false
---

## [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427.pdf)
[Yu Zhao](https://aclanthology.org/people/yu-zhao/unverified/), [Yuanbin Qu](https://aclanthology.org/people/yuanbin-qu/unverified/), [Konrad Staniszewski](https://aclanthology.org/people/konrad-staniszewski/unverified/), [Szymon Tworkowski](https://aclanthology.org/people/szymon-tworkowski/unverified/), [Wei Liu](https://aclanthology.org/people/wei-liu/unverified/), [Piotr Miłoś](https://aclanthology.org/people/piotr-milos/unverified/), [Yuxiang Wu](https://aclanthology.org/people/yuxiang-wu/), [Pasquale Minervini](https://aclanthology.org/people/pasquale-minervini/)
##### Correct Metadata for 
ALL author names match the snapshot above—including middle initials, hyphens, and accents.
Create GitHub issue for staff review
##### Abstract
Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use _causal masking_ to compute the likelihood of each token given its context; this strategy is widely adopted due to its simplicity and efficiency. However, to this day, the influence of the pre-training sequence composition strategy on the generalisation properties of the model remains under-explored.In this work, we find that applying causal masking can lead to the inclusion of distracting information from previous documents during pre-training, which negatively impacts the performance of the models on language modelling and downstream tasks. In _intra-document causal masking_ , the likelihood of each token is only conditioned on the previous tokens in the same document, eliminating potential distracting information from previous documents and significantly improving performance. Furthermore, we find that concatenating related documents can reduce some potential distractions during pre-training, and our proposed efficient retrieval-based sequence construction method, Bm25Chunk, can improve in-context learning (+11.6%), knowledge memorisation (+9.8%), and context utilisation (+7.2%) abilities of language models without sacrificing efficiency. 

Anthology ID:
    2024.acl-long.427 

Volume:
    [Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)](https://aclanthology.org/volumes/2024.acl-long/) 

Month:
    August 

Year:
    2024 

Address:
    Bangkok, Thailand 

Editors:
     [Lun-Wei Ku](https://aclanthology.org/people/lun-wei-ku/), [Andre Martins](https://aclanthology.org/people/andre-f-t-martins/), [Vivek Srikumar](https://aclanthology.org/people/vivek-srikumar/) 

Venue:


SIG:


Publisher:
    Association for Computational Linguistics 

Note:


Pages:
    7897–7912 

Language:


URL:
    <https://aclanthology.org/2024.acl-long.427/> 

DOI:
    [10.18653/v1/2024.acl-long.427](https://doi.org/10.18653/v1/2024.acl-long.427 "To the current version of the paper by DOI") 

Bibkey:
    zhao-etal-2024-analysing 

Cite (ACL):
    Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłoś, Yuxiang Wu, and Pasquale Minervini. 2024. [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/). In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 7897–7912, Bangkok, Thailand. Association for Computational Linguistics. 

Cite (Informal):
    [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/) (Zhao et al., ACL 2024) 

Copy Citation:
     BibTeX Markdown MODS XML Endnote More options… 

PDF:
    <https://aclanthology.org/2024.acl-long.427.pdf>
[Cite ](https://aclanthology.org/2024.acl-long.427/ "Open dialog for exporting citations")[Search ](https://www.semanticscholar.org/search?+q=Analysing+The+Impact+of+Sequence+Composition+on+Language+Model+Pre-Training "Search for 'Analysing The Impact of Sequence Composition on Language Model Pre-Training' on Semantic Scholar")[ Fix data](https://aclanthology.org/2024.acl-long.427/ "Correct problems with title, author list, and abstract")
##### Export citation



```
@inproceedings{zhao-etal-2024-analysing,
    title = "Analysing The Impact of Sequence Composition on Language Model Pre-Training",
    author = "Zhao, Yu  and
      Qu, Yuanbin  and
      Staniszewski, Konrad  and
      Tworkowski, Szymon  and
      Liu, Wei  and
      Mi{\l}o{\'s}, Piotr  and
      Wu, Yuxiang  and
      Minervini, Pasquale",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.427/",
    doi = "10.18653/v1/2024.acl-long.427",
    pages = "7897--7912",
    abstract = "Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use \textit{causal masking} to compute the likelihood of each token given its context; this strategy is widely adopted due to its simplicity and efficiency. However, to this day, the influence of the pre-training sequence composition strategy on the generalisation properties of the model remains under-explored.In this work, we find that applying causal masking can lead to the inclusion of distracting information from previous documents during pre-training, which negatively impacts the performance of the models on language modelling and downstream tasks. In \textit{intra-document causal masking}, the likelihood of each token is only conditioned on the previous tokens in the same document, eliminating potential distracting information from previous documents and significantly improving performance. Furthermore, we find that concatenating related documents can reduce some potential distractions during pre-training, and our proposed efficient retrieval-based sequence construction method, Bm25Chunk, can improve in-context learning (+11.6{\%}), knowledge memorisation (+9.8{\%}), and context utilisation (+7.2{\%}) abilities of language models without sacrificing efficiency."
}
```

Download as File Copy to Clipboard

```
<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
<mods ID="zhao-etal-2024-analysing">
    <titleInfo>
        <title>Analysing The Impact of Sequence Composition on Language Model Pre-Training</title>
    </titleInfo>
    <name type="personal">
        <namePart type="given">Yu</namePart>
        <namePart type="family">Zhao</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Yuanbin</namePart>
        <namePart type="family">Qu</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Konrad</namePart>
        <namePart type="family">Staniszewski</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Szymon</namePart>
        <namePart type="family">Tworkowski</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Wei</namePart>
        <namePart type="family">Liu</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Piotr</namePart>
        <namePart type="family">Miłoś</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Yuxiang</namePart>
        <namePart type="family">Wu</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Pasquale</namePart>
        <namePart type="family">Minervini</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <originInfo>
        <dateIssued>2024-08</dateIssued>
    </originInfo>
    <typeOfResource>text</typeOfResource>
    <relatedItem type="host">
        <titleInfo>
            <title>Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)</title>
        </titleInfo>
        <name type="personal">
            <namePart type="given">Lun-Wei</namePart>
            <namePart type="family">Ku</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Andre</namePart>
            <namePart type="family">Martins</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Vivek</namePart>
            <namePart type="family">Srikumar</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <originInfo>
            <publisher>Association for Computational Linguistics</publisher>
            <place>
                <placeTerm type="text">Bangkok, Thailand</placeTerm>
            </place>
        </originInfo>
        <genre authority="marcgt">conference publication</genre>
    </relatedItem>
    <abstract>Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use causal masking to compute the likelihood of each token given its context; this strategy is widely adopted due to its simplicity and efficiency. However, to this day, the influence of the pre-training sequence composition strategy on the generalisation properties of the model remains under-explored.In this work, we find that applying causal masking can lead to the inclusion of distracting information from previous documents during pre-training, which negatively impacts the performance of the models on language modelling and downstream tasks. In intra-document causal masking, the likelihood of each token is only conditioned on the previous tokens in the same document, eliminating potential distracting information from previous documents and significantly improving performance. Furthermore, we find that concatenating related documents can reduce some potential distractions during pre-training, and our proposed efficient retrieval-based sequence construction method, Bm25Chunk, can improve in-context learning (+11.6%), knowledge memorisation (+9.8%), and context utilisation (+7.2%) abilities of language models without sacrificing efficiency.</abstract>
    <identifier type="citekey">zhao-etal-2024-analysing</identifier>
    <identifier type="doi">10.18653/v1/2024.acl-long.427</identifier>
    <location>
        <url>https://aclanthology.org/2024.acl-long.427/</url>
    </location>
    <part>
        <date>2024-08</date>
        <extent unit="page">
            <start>7897</start>
            <end>7912</end>
        </extent>
    </part>
</mods>
</modsCollection>

```

Download as File Copy to Clipboard

```
%0 Conference Proceedings
%T Analysing The Impact of Sequence Composition on Language Model Pre-Training
%A Zhao, Yu
%A Qu, Yuanbin
%A Staniszewski, Konrad
%A Tworkowski, Szymon
%A Liu, Wei
%A Miłoś, Piotr
%A Wu, Yuxiang
%A Minervini, Pasquale
%Y Ku, Lun-Wei
%Y Martins, Andre
%Y Srikumar, Vivek
%S Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
%D 2024
%8 August
%I Association for Computational Linguistics
%C Bangkok, Thailand
%F zhao-etal-2024-analysing
%X Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use causal masking to compute the likelihood of each token given its context; this strategy is widely adopted due to its simplicity and efficiency. However, to this day, the influence of the pre-training sequence composition strategy on the generalisation properties of the model remains under-explored.In this work, we find that applying causal masking can lead to the inclusion of distracting information from previous documents during pre-training, which negatively impacts the performance of the models on language modelling and downstream tasks. In intra-document causal masking, the likelihood of each token is only conditioned on the previous tokens in the same document, eliminating potential distracting information from previous documents and significantly improving performance. Furthermore, we find that concatenating related documents can reduce some potential distractions during pre-training, and our proposed efficient retrieval-based sequence construction method, Bm25Chunk, can improve in-context learning (+11.6%), knowledge memorisation (+9.8%), and context utilisation (+7.2%) abilities of language models without sacrificing efficiency.
%R 10.18653/v1/2024.acl-long.427
%U https://aclanthology.org/2024.acl-long.427/
%U https://doi.org/10.18653/v1/2024.acl-long.427
%P 7897-7912
```

Download as File Copy to Clipboard
##### Markdown (Informal)
[Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/) (Zhao et al., ACL 2024)
  * [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/) (Zhao et al., ACL 2024)


##### ACL
  * Yu Zhao, Yuanbin Qu, Konrad Staniszewski, Szymon Tworkowski, Wei Liu, Piotr Miłoś, Yuxiang Wu, and Pasquale Minervini. 2024. [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/). In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 7897–7912, Bangkok, Thailand. Association for Computational Linguistics.


Copy Markdown to Clipboard Copy ACL to Clipboard
