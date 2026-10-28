---
title: 'SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational
  Collapse - ACL Anthology'
id: simsmoe-toward-efficient-training-mixture-of-experts-via-solving-representationa
tags:
- cmp170hx-training-feasibility-a0d4fc
- moe-training
- expert-collapse
- routing
- thin-metadata-only
created: '2026-10-27T11:26:29.429481Z'
updated: '2026-10-27T19:00:31.498062Z'
source: https://doi.org/10.18653/v1/2025.findings-naacl.107
source_domain: doi.org
fetched_at: '2026-10-27T11:26:28.808324Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'SimSMoE (Do, Le & Tran, Findings of NAACL 2025, pp. 2012-2025; DOI 10.18653/v1/2025.findings-naacl.107)
  targets representation collapse in sparse mixture-of-experts training — the failure
  where experts converge to near-identical representations, wasting parameters and
  hurting quality. It proposes a similarity-based routing algorithm that provably
  drives experts apart under a fixed FLOPs budget, and evaluates on three LLMs for
  both pre-training and fine-tuning, reporting gains over existing SMoE routing policies.
  Code at github.com/giangdip2410/SimSMoE. NOTE: this note is the ACL Anthology landing
  page only (abstract + BibTeX, ~1.1k words) — no numbers, no training-cost data;
  the primary PDF is aclanthology.org/2025.findings-naacl.107.pdf. Value here is corroborating
  that expert collapse / load balancing is a first-order MoE training risk at the
  scale this box can reach, not providing quantitative evidence.'
doi: 10.18653/v1/2025.findings-naacl.107
utility_score: 12.0
citation_count: 6
venue: 'Findings of the Association for Computational Linguistics: NAACL'
is_retracted: false
---

## [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107.pdf)
[Giang Do](https://aclanthology.org/people/giang-do/), [Hung Le](https://aclanthology.org/people/hung-le/), [Truyen Tran](https://aclanthology.org/people/truyen-tran/)
##### Correct Metadata for 
ALL author names match the snapshot above—including middle initials, hyphens, and accents.
Create GitHub issue for staff review
##### Abstract
Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks. Our implementation is publicly available at <https://github.com/giangdip2410/SimSMoE>. 

Anthology ID:
    2025.findings-naacl.107 

Volume:
    [Findings of the Association for Computational Linguistics: NAACL 2025](https://aclanthology.org/volumes/2025.findings-naacl/) 

Month:
    April 

Year:
    2025 

Address:
    Albuquerque, New Mexico 

Editors:
     [Luis Chiruzzo](https://aclanthology.org/people/luis-chiruzzo/), [Alan Ritter](https://aclanthology.org/people/alan-ritter/), [Lu Wang](https://aclanthology.org/people/lu-wang/unverified/) 

Venue:


SIG:


Publisher:
    Association for Computational Linguistics 

Note:


Pages:
    2012–2025 

Language:


URL:
    <https://aclanthology.org/2025.findings-naacl.107/> 

DOI:
    [10.18653/v1/2025.findings-naacl.107](https://doi.org/10.18653/v1/2025.findings-naacl.107 "To the current version of the paper by DOI") 

Bibkey:
    do-etal-2025-simsmoe 

Cite (ACL):
    Giang Do, Hung Le, and Truyen Tran. 2025. [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107/). In _Findings of the Association for Computational Linguistics: NAACL 2025_ , pages 2012–2025, Albuquerque, New Mexico. Association for Computational Linguistics. 

Cite (Informal):
    [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107/) (Do et al., Findings 2025) 

Copy Citation:
     BibTeX Markdown MODS XML Endnote More options… 

PDF:
    <https://aclanthology.org/2025.findings-naacl.107.pdf>
[Cite ](https://aclanthology.org/2025.findings-naacl.107/ "Open dialog for exporting citations")[Search ](https://www.semanticscholar.org/search?+q=SimSMoE%3A+Toward+Efficient+Training+Mixture+of+Experts+via+Solving+Representational+Collapse "Search for 'SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse' on Semantic Scholar")[ Fix data](https://aclanthology.org/2025.findings-naacl.107/ "Correct problems with title, author list, and abstract")
##### Export citation



```
@inproceedings{do-etal-2025-simsmoe,
    title = "{S}im{SM}o{E}: Toward Efficient Training Mixture of Experts via Solving Representational Collapse",
    author = "Do, Giang  and
      Le, Hung  and
      Tran, Truyen",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {NAACL} 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.107/",
    doi = "10.18653/v1/2025.findings-naacl.107",
    pages = "2012--2025",
    ISBN = "979-8-89176-195-7",
    abstract = "Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks. Our implementation is publicly available at \url{https://github.com/giangdip2410/SimSMoE}."
}
```

Download as File Copy to Clipboard

```
<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
<mods ID="do-etal-2025-simsmoe">
    <titleInfo>
        <title>SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse</title>
    </titleInfo>
    <name type="personal">
        <namePart type="given">Giang</namePart>
        <namePart type="family">Do</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Hung</namePart>
        <namePart type="family">Le</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Truyen</namePart>
        <namePart type="family">Tran</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <originInfo>
        <dateIssued>2025-04</dateIssued>
    </originInfo>
    <typeOfResource>text</typeOfResource>
    <relatedItem type="host">
        <titleInfo>
            <title>Findings of the Association for Computational Linguistics: NAACL 2025</title>
        </titleInfo>
        <name type="personal">
            <namePart type="given">Luis</namePart>
            <namePart type="family">Chiruzzo</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Alan</namePart>
            <namePart type="family">Ritter</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Lu</namePart>
            <namePart type="family">Wang</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <originInfo>
            <publisher>Association for Computational Linguistics</publisher>
            <place>
                <placeTerm type="text">Albuquerque, New Mexico</placeTerm>
            </place>
        </originInfo>
        <genre authority="marcgt">conference publication</genre>
        <identifier type="isbn">979-8-89176-195-7</identifier>
    </relatedItem>
    <abstract>Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks. Our implementation is publicly available at https://github.com/giangdip2410/SimSMoE.</abstract>
    <identifier type="citekey">do-etal-2025-simsmoe</identifier>
    <identifier type="doi">10.18653/v1/2025.findings-naacl.107</identifier>
    <location>
        <url>https://aclanthology.org/2025.findings-naacl.107/</url>
    </location>
    <part>
        <date>2025-04</date>
        <extent unit="page">
            <start>2012</start>
            <end>2025</end>
        </extent>
    </part>
</mods>
</modsCollection>

```

Download as File Copy to Clipboard

```
%0 Conference Proceedings
%T SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse
%A Do, Giang
%A Le, Hung
%A Tran, Truyen
%Y Chiruzzo, Luis
%Y Ritter, Alan
%Y Wang, Lu
%S Findings of the Association for Computational Linguistics: NAACL 2025
%D 2025
%8 April
%I Association for Computational Linguistics
%C Albuquerque, New Mexico
%@ 979-8-89176-195-7
%F do-etal-2025-simsmoe
%X Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks. Our implementation is publicly available at https://github.com/giangdip2410/SimSMoE.
%R 10.18653/v1/2025.findings-naacl.107
%U https://aclanthology.org/2025.findings-naacl.107/
%U https://doi.org/10.18653/v1/2025.findings-naacl.107
%P 2012-2025
```

Download as File Copy to Clipboard
##### Markdown (Informal)
[SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107/) (Do et al., Findings 2025)
  * [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107/) (Do et al., Findings 2025)


##### ACL
  * Giang Do, Hung Le, and Truyen Tran. 2025. [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://aclanthology.org/2025.findings-naacl.107/). In _Findings of the Association for Computational Linguistics: NAACL 2025_ , pages 2012–2025, Albuquerque, New Mexico. Association for Computational Linguistics.


Copy Markdown to Clipboard Copy ACL to Clipboard
