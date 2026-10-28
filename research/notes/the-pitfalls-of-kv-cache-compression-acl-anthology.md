---
title: The Pitfalls of KV Cache Compression - ACL Anthology
id: the-pitfalls-of-kv-cache-compression-acl-anthology
tags:
- cmp170hx-training-feasibility-a0d4fc
- kv-cache-compression
- evaluation-pitfalls
- sparse-attention
- instruction-following
created: '2026-10-27T11:40:31.601812Z'
updated: '2026-10-27T19:00:31.557124Z'
source: https://aclanthology.org/2026.acl-long.1926/
source_domain: aclanthology.org
fetched_at: '2026-10-27T11:40:30.982423Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: paper
deprecated: false
summary: 'The Pitfalls of KV Cache Compression (Chen, Geh, Grover, Van den Broeck
  & Israel, ACL 2026 Long Papers, pp. 41530-41553; DOI 10.18653/v1/2026.acl-long.1926)
  — an evaluation-methodology paper, not a training study. It evaluates five inference-time
  KV compression methods (StreamingLLM, SnapKV, TOVA, H2O, K-Norm) on Llama3.1 8B
  and Qwen2.5 14B under multi-instruction prompting with IFEval, and shows that aggregate
  benchmark scores hide method-specific failures: certain instructions degrade much
  faster under compression and end up completely ignored by the model. It documents
  system-prompt leakage as a case study and attributes it to compression method, instruction
  order and KV eviction bias, then proposes simple eviction-policy changes that reduce
  the effect. Relevance to this run: it is the primary source behind the periodic-weak-spots
  claim that average scores conceal compression failures, and it supplies a concrete
  evaluation protocol (multi-instruction prompting, per-instruction breakdown, leakage
  probes) that a CSA2-at-32k experiment should adopt. Scope caveat: these are eviction-based
  inference KV compressors, not DeepSeek''s chunked training-time CSA/CSA2 — the failure
  mechanism (eviction bias) is different from phase sensitivity, so it corroborates
  the evaluation principle rather than the specific CSA2 claim. NOTE: this note is
  the ACL Anthology landing page (abstract + BibTeX, ~1.4k words); the PDF is at aclanthology.org/2026.acl-long.1926.pdf.'
---

*Suggested by [[periodic-weak-spots]] — Cited as the primary source for the claim that aggregate scores hide method-specific failures of KV-cache compression, which is the evaluation protocol a CSA2 experiment needs*

## [The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926.pdf)
[Alex Chen](https://aclanthology.org/people/alex-chen/), [Renato Geh](https://aclanthology.org/people/renato-geh/), [Aditya Grover](https://aclanthology.org/people/aditya-grover/unverified/), [Guy Van Den Broeck](https://aclanthology.org/people/guy-van-den-broeck/), [Daniel Mingyi Israel](https://aclanthology.org/people/daniel-mingyi-israel/)
##### Correct Metadata for 
ALL author names match the snapshot above—including middle initials, hyphens, and accents.
Create GitHub issue for staff review
##### Abstract
KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls that practitioners should be aware of when deploying KV cache compressed LLMs. We evaluate five KV cache compression methods (StreamingLLM, SnapKV, TOVA, H2O, and K-Norm) on Llama3.1 8B and Qwen2.5 14B under multi-instruction prompting with IFEval. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example, we highlight system prompt leakage as a case study, empirically demonstrating the impact of compression on leakage and general instruction-following. We identify several factors that contribute to system prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks. 

Anthology ID:
    2026.acl-long.1926 

Volume:
    [Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)](https://aclanthology.org/volumes/2026.acl-long/) 

Month:
    July 

Year:
    2026 

Address:
    San Diego, California, United States 

Editors:
     [Maria Liakata](https://aclanthology.org/people/maria-liakata/), [Viviane P. Moreira](https://aclanthology.org/people/viviane-moreira/), [Jiajun Zhang](https://aclanthology.org/people/jiajun-zhang/unverified/), [David Jurgens](https://aclanthology.org/people/david-jurgens/) 

Venue:


SIG:


Publisher:
    Association for Computational Linguistics 

Note:


Pages:
    41530–41553 

Language:


URL:
    <https://aclanthology.org/2026.acl-long.1926/> 

DOI:
    [10.18653/v1/2026.acl-long.1926](https://doi.org/10.18653/v1/2026.acl-long.1926 "To the current version of the paper by DOI") 

Bibkey:
    chen-etal-2026-pitfalls 

Cite (ACL):
    Alex Chen, Renato Geh, Aditya Grover, Guy Van Den Broeck, and Daniel Mingyi Israel. 2026. [The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926/). In _Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 41530–41553, San Diego, California, United States. Association for Computational Linguistics. 

Cite (Informal):
    [The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926/) (Chen et al., ACL 2026) 

Copy Citation:
     BibTeX Markdown MODS XML Endnote More options… 

PDF:
    <https://aclanthology.org/2026.acl-long.1926.pdf> 

Checklist:
    [ 2026.acl-long.1926.checklist.pdf](https://aclanthology.org/attachments/2026.acl-long.1926.checklist.pdf)
[Cite ](https://aclanthology.org/2026.acl-long.1926/ "Open dialog for exporting citations")[Search ](https://www.semanticscholar.org/search?+q=The+Pitfalls+of+KV+Cache+Compression "Search for 'The Pitfalls of KV Cache Compression' on Semantic Scholar")[ Checklist ](https://aclanthology.org/attachments/2026.acl-long.1926.checklist.pdf "Open checklist for 'The Pitfalls of KV Cache Compression'")[ Fix data](https://aclanthology.org/2026.acl-long.1926/ "Correct problems with title, author list, and abstract")
##### Export citation



```
@inproceedings{chen-etal-2026-pitfalls,
    title = "The Pitfalls of {KV} Cache Compression",
    author = "Chen, Alex  and
      Geh, Renato  and
      Grover, Aditya  and
      Van Den Broeck, Guy  and
      Israel, Daniel Mingyi",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1926/",
    doi = "10.18653/v1/2026.acl-long.1926",
    pages = "41530--41553",
    ISBN = "979-8-89176-390-6",
    abstract = "KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls that practitioners should be aware of when deploying KV cache compressed LLMs. We evaluate five KV cache compression methods (StreamingLLM, SnapKV, TOVA, H2O, and K-Norm) on Llama3.1 8B and Qwen2.5 14B under multi-instruction prompting with IFEval. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example, we highlight system prompt leakage as a case study, empirically demonstrating the impact of compression on leakage and general instruction-following. We identify several factors that contribute to system prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks."
}
```

Download as File Copy to Clipboard

```
<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
<mods ID="chen-etal-2026-pitfalls">
    <titleInfo>
        <title>The Pitfalls of KV Cache Compression</title>
    </titleInfo>
    <name type="personal">
        <namePart type="given">Alex</namePart>
        <namePart type="family">Chen</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Renato</namePart>
        <namePart type="family">Geh</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Aditya</namePart>
        <namePart type="family">Grover</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Guy</namePart>
        <namePart type="family">Van Den Broeck</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <name type="personal">
        <namePart type="given">Daniel</namePart>
        <namePart type="given">Mingyi</namePart>
        <namePart type="family">Israel</namePart>
        <role>
            <roleTerm authority="marcrelator" type="text">author</roleTerm>
        </role>
    </name>
    <originInfo>
        <dateIssued>2026-07</dateIssued>
    </originInfo>
    <typeOfResource>text</typeOfResource>
    <relatedItem type="host">
        <titleInfo>
            <title>Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)</title>
        </titleInfo>
        <name type="personal">
            <namePart type="given">Maria</namePart>
            <namePart type="family">Liakata</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Viviane</namePart>
            <namePart type="given">P</namePart>
            <namePart type="family">Moreira</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">Jiajun</namePart>
            <namePart type="family">Zhang</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <name type="personal">
            <namePart type="given">David</namePart>
            <namePart type="family">Jurgens</namePart>
            <role>
                <roleTerm authority="marcrelator" type="text">editor</roleTerm>
            </role>
        </name>
        <originInfo>
            <publisher>Association for Computational Linguistics</publisher>
            <place>
                <placeTerm type="text">San Diego, California, United States</placeTerm>
            </place>
        </originInfo>
        <genre authority="marcgt">conference publication</genre>
        <identifier type="isbn">979-8-89176-390-6</identifier>
    </relatedItem>
    <abstract>KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls that practitioners should be aware of when deploying KV cache compressed LLMs. We evaluate five KV cache compression methods (StreamingLLM, SnapKV, TOVA, H2O, and K-Norm) on Llama3.1 8B and Qwen2.5 14B under multi-instruction prompting with IFEval. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example, we highlight system prompt leakage as a case study, empirically demonstrating the impact of compression on leakage and general instruction-following. We identify several factors that contribute to system prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks.</abstract>
    <identifier type="citekey">chen-etal-2026-pitfalls</identifier>
    <identifier type="doi">10.18653/v1/2026.acl-long.1926</identifier>
    <location>
        <url>https://aclanthology.org/2026.acl-long.1926/</url>
    </location>
    <part>
        <date>2026-07</date>
        <extent unit="page">
            <start>41530</start>
            <end>41553</end>
        </extent>
    </part>
</mods>
</modsCollection>

```

Download as File Copy to Clipboard

```
%0 Conference Proceedings
%T The Pitfalls of KV Cache Compression
%A Chen, Alex
%A Geh, Renato
%A Grover, Aditya
%A Van Den Broeck, Guy
%A Israel, Daniel Mingyi
%Y Liakata, Maria
%Y Moreira, Viviane P.
%Y Zhang, Jiajun
%Y Jurgens, David
%S Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
%D 2026
%8 July
%I Association for Computational Linguistics
%C San Diego, California, United States
%@ 979-8-89176-390-6
%F chen-etal-2026-pitfalls
%X KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls that practitioners should be aware of when deploying KV cache compressed LLMs. We evaluate five KV cache compression methods (StreamingLLM, SnapKV, TOVA, H2O, and K-Norm) on Llama3.1 8B and Qwen2.5 14B under multi-instruction prompting with IFEval. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example, we highlight system prompt leakage as a case study, empirically demonstrating the impact of compression on leakage and general instruction-following. We identify several factors that contribute to system prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks.
%R 10.18653/v1/2026.acl-long.1926
%U https://aclanthology.org/2026.acl-long.1926/
%U https://doi.org/10.18653/v1/2026.acl-long.1926
%P 41530-41553
```

Download as File Copy to Clipboard
##### Markdown (Informal)
[The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926/) (Chen et al., ACL 2026)
  * [The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926/) (Chen et al., ACL 2026)


##### ACL
  * Alex Chen, Renato Geh, Aditya Grover, Guy Van Den Broeck, and Daniel Mingyi Israel. 2026. [The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926/). In _Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 41530–41553, San Diego, California, United States. Association for Computational Linguistics.


Copy Markdown to Clipboard Copy ACL to Clipboard
