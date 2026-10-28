---
title: HuggingFaceFW/fineweb-edu · Datasets at Hugging Face
id: huggingfacefwfineweb-edu-datasets-at-hugging-face
tags:
- cmp170hx-training-feasibility-a0d4fc
- dataset
- data-curation
- data-pipeline
- training-cost
- measured
created: '2026-10-27T10:58:50.947621Z'
updated: '2026-10-27T19:00:31.384435Z'
source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
source_domain: huggingface.co
fetched_at: '2026-10-27T10:58:50.321261Z'
fetch_provider: crawl4ai
status: review
type: note
tier: institutional
content_type: dataset
deprecated: false
summary: 'Dataset card for HuggingFaceFW/fineweb-edu: 1.3T GPT-2 tokens of educational
  web text filtered from the 15T-token FineWeb corpus, plus a sibling FineWeb-Edu-score-2
  at 5.4T tokens (threshold 2 instead of 3). Filtering: a Snowflake-arctic-embed-based
  regression classifier fine-tuned on 500k Llama3-70B-Instruct educational-quality
  annotations (0-5 additive scale); threshold 3 removed 92% of FineWeb; classifier
  F1 82% at threshold 3. Classifying the full 15T tokens took 6,000 H100 GPU hours
  — an anchor for how much accelerator time data curation alone consumes.


  Practical shape for a small run: 114 configs (per-CC-MAIN dump plus sample-10BT
  9.67M rows, sample-100BT 97.3M rows, sample-350BT 339M rows, nested 10BT within
  100BT within 350BT); default has 1.53B rows in the viewer, 5.84 TB on disk. Fields:
  text, id, dump, url, date, file_path, language, language_score, token_count (35-159k),
  score (2.52-5.06), int_score. License ODC-By v1.0 plus CommonCrawl ToU. The deduplicated
  version (220B tokens) ships as the fineweb-edu-dedup subset of SmolLM-Corpus, and
  the authors report dedup had no measurable effect on a 1.8B model trained on 350B
  tokens.


  Relevance to the repo''s 9B-token data pipeline: sample-10BT alone is roughly the
  right order of magnitude for a 9B-token mix, and the token_count field makes exact
  token budgeting possible without a tokenizer pass.'
utility_score: 16.0
---

This dataset has 1 file scanned as unsafe.
Show files
Dataset Viewer
[ Auto-converted to Parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/refs%2Fconvert%2Fparquet/default) API Embed [ Duplicate](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu?duplicate=true) Data Studio
Subset (114)
default · 1.53B rows
default (1.53B rows) CC-MAIN-2013-20 (11M rows) CC-MAIN-2013-48 (10.8M rows) CC-MAIN-2014-10 (11M rows) CC-MAIN-2014-15 (10.4M rows) CC-MAIN-2014-23 (11.7M rows) CC-MAIN-2014-35 (11.1M rows) CC-MAIN-2014-41 (11.4M rows) CC-MAIN-2014-42 (10.7M rows) CC-MAIN-2014-49 (9.39M rows) CC-MAIN-2014-52 (11.2M rows) CC-MAIN-2015-06 (10.3M rows) CC-MAIN-2015-11 (10.4M rows) CC-MAIN-2015-14 (9.79M rows) CC-MAIN-2015-18 (11.1M rows) CC-MAIN-2015-22 (10.9M rows) CC-MAIN-2015-27 (9.74M rows) CC-MAIN-2015-32 (10.2M rows) CC-MAIN-2015-35 (10.3M rows) CC-MAIN-2015-40 (8.25M rows) CC-MAIN-2015-48 (10.1M rows) CC-MAIN-2016-07 (9.88M rows) CC-MAIN-2016-18 (8.6M rows) CC-MAIN-2016-22 (8.92M rows) CC-MAIN-2016-26 (7.41M rows) CC-MAIN-2016-30 (9.86M rows) CC-MAIN-2016-36 (9.57M rows) CC-MAIN-2016-40 (11.1M rows) CC-MAIN-2016-44 (14M rows) CC-MAIN-2016-50 (13.8M rows) CC-MAIN-2017-04 (14.8M rows) CC-MAIN-2017-09 (15.1M rows) CC-MAIN-2017-13 (17.7M rows) CC-MAIN-2017-17 (17.5M rows) CC-MAIN-2017-22 (13M rows) CC-MAIN-2017-26 (13.9M rows) CC-MAIN-2017-30 (12.9M rows) CC-MAIN-2017-34 (13.7M rows) CC-MAIN-2017-39 (12.8M rows) CC-MAIN-2017-43 (14.9M rows) CC-MAIN-2017-47 (13.2M rows) CC-MAIN-2017-51 (12.1M rows) CC-MAIN-2018-05 (14.2M rows) CC-MAIN-2018-09 (14.3M rows) CC-MAIN-2018-13 (13.5M rows) CC-MAIN-2018-17 (12.9M rows) CC-MAIN-2018-22 (11.5M rows) CC-MAIN-2018-26 (14.1M rows) CC-MAIN-2018-30 (14.2M rows) CC-MAIN-2018-34 (12.7M rows) CC-MAIN-2018-39 (12.9M rows) CC-MAIN-2018-43 (14.7M rows) CC-MAIN-2018-47 (12.6M rows) CC-MAIN-2018-51 (14.6M rows) CC-MAIN-2019-04 (12.8M rows) CC-MAIN-2019-09 (14.1M rows) CC-MAIN-2019-13 (12.8M rows) CC-MAIN-2019-18 (13M rows) CC-MAIN-2019-22 (13M rows) CC-MAIN-2019-26 (12.5M rows) CC-MAIN-2019-30 (12.5M rows) CC-MAIN-2019-35 (13.7M rows) CC-MAIN-2019-39 (12M rows) CC-MAIN-2019-43 (13M rows) CC-MAIN-2019-47 (12.1M rows) CC-MAIN-2019-51 (11.2M rows) CC-MAIN-2020-05 (15.3M rows) CC-MAIN-2020-10 (11.3M rows) CC-MAIN-2020-16 (14.4M rows) CC-MAIN-2020-24 (12.2M rows) CC-MAIN-2020-29 (15.5M rows) CC-MAIN-2020-34 (12M rows) CC-MAIN-2020-40 (16.9M rows) CC-MAIN-2020-45 (13M rows) CC-MAIN-2020-50 (12.7M rows) CC-MAIN-2021-04 (17.2M rows) CC-MAIN-2021-10 (13.9M rows) CC-MAIN-2021-17 (17.1M rows) CC-MAIN-2021-21 (13.8M rows) CC-MAIN-2021-25 (12.4M rows) CC-MAIN-2021-31 (18.4M rows) CC-MAIN-2021-39 (15.9M rows) CC-MAIN-2021-43 (18.7M rows) CC-MAIN-2021-49 (12.3M rows) CC-MAIN-2022-05 (16M rows) CC-MAIN-2022-21 (20.3M rows) CC-MAIN-2022-27 (18.2M rows) CC-MAIN-2022-33 (13.1M rows) CC-MAIN-2022-40 (19.1M rows) CC-MAIN-2022-49 (20M rows) CC-MAIN-2023-06 (20.1M rows) CC-MAIN-2023-14 (18.9M rows) CC-MAIN-2023-23 (20.3M rows) CC-MAIN-2023-40 (22.9M rows) CC-MAIN-2023-50 (22.1M rows) CC-MAIN-2024-10 (20M rows) CC-MAIN-2024-18 (18M rows) CC-MAIN-2024-22 (17.1M rows) CC-MAIN-2024-26 (17.1M rows) CC-MAIN-2024-30 (15.3M rows) CC-MAIN-2024-33 (13.5M rows) CC-MAIN-2024-38 (17.4M rows) CC-MAIN-2024-42 (14.8M rows) CC-MAIN-2024-46 (16.3M rows) CC-MAIN-2024-51 (13M rows) CC-MAIN-2025-05 (19.3M rows) CC-MAIN-2025-08 (16.1M rows) CC-MAIN-2025-13 (16.8M rows) CC-MAIN-2025-18 (17.3M rows) CC-MAIN-2025-21 (15.6M rows) CC-MAIN-2025-26 (14M rows) sample-100BT (97.3M rows) sample-10BT (9.67M rows) sample-350BT (339M rows)
Split (1)
train · 1.53B rows
train (1.53B rows)
SQL
Console  
|  text stringlengths 150 592k  |  stringlengths  |  dump stringclasses 1 value  |  stringlengths 14 846  |  date stringclasses 0 values  |  file_path stringlengths 138 138  |  language stringclasses 1 value  |  language_score float64 0.65 1  |  token_count int64 35 159k  |  score float64 2.52 5.06  |  int_score int64  |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  
|  No. 24; Updated March 2011 Click here to download and print a PDF version of this document. Parents are usually the first to recognize that their child has a problem with emotions or behavior. Still, the decision to seek professional help can be difficult and painful for a parent. The first step is to gently try to tal...  |  <urn:uuid:673b1bf6-2c30-40ae-992b-c387d00a836a>  |  CC-MAIN-2013-20  |  http://aacap.org/page.ww?name=When+to+Seek+Help+for+Your+Child&section=Facts+for+Families  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.927742   |  3.375   |  
|  Previous abstract Next abstract Session 40 - The Interstellar Medium. Display session, Tuesday, June 09 Gamma Ray Burst (GRB) explosions can make kpc-size shells and holes in the interstellar media (ISM) of spiral galaxies if much of the energy heats the local gas to above 10^7 K. Disk blowout is probably the major cau...  |  <urn:uuid:e2300ad5-01dd-4e80-92b3-7ec88785cc9d>  |  CC-MAIN-2013-20  |  http://aas.org/archives/BAAS/v30n2/aas192/abs/S040015.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.912641   |  2.765625   |  
|  Question: How is bipolar disorder different from unipolar depression or 'regular' depression? Answer: Both bipolar disorder and major depression are typically associated with depressive episodes. So both illnesses are accompanied by depressions. The difference is that in bipolar disorder people also have periods of ele...  |  <urn:uuid:e6ba92ad-ed0a-4cac-8e5d-204b78cdd250>  |  CC-MAIN-2013-20  |  http://abcnews.go.com/Health/BipolarOverview/story?id=4359993  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.943297   |  2.609375   |  
|  Making the Case for Action This fact sheet(pdf) and slide deck provide essential state-specific information that addresses the economic imperative, the equity imperative, and the expectations imperative of the college- and career-ready agenda. These resources can be used on their own or serve as the foundation for a pe...  |  <urn:uuid:3b2c1a91-4f52-464d-ad69-49c1cbadaba8>  |  CC-MAIN-2013-20  |  http://achieve.org/Idaho  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.927479   |  2.6875   |  
|  A land whose rich cultural heritage is discovered not only from within the walls of numerous museums, galleries and churches, many of which today, as zero category monuments are included in a part of the UNESCO World Heritage List, but also in that magical place on the Mediterranean, where even the shortest stroll beco...  |  <urn:uuid:a69aabbc-f529-4d67-843a-a5c3cb4e8fe0>  |  CC-MAIN-2013-20  |  http://adriatictraveller.com/ru/croatia-essential/heritage.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.951876   |  1,292   |  2.53125   |  
|  adopt many methods to determine whether the unborn baby is a boy or a girl. The Chinese pregnancy calendar is an often used method to know about the gender of the new life in the mothers womb. is an ancient way for predicting the gender of the unborn baby It is also known as a Chinese conception chart, or the Chinese C...  |  <urn:uuid:f7082439-68e9-45b6-a427-4600dceaf5e3>  |  CC-MAIN-2013-20  |  http://ainads.com/Pregnancy/Chinese%20Pregnancy%20Calendar%20.php  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.890836   |  3.125   |  
|  Average life span in the wild: 12 years Size: 21 in (50 cm) Weight: 14.4 oz (408 g) Did you know? Chameleons don't change colors to match their surroundings. Each species displays distinct color patterns to indicate specific reactions or emotions. The Meller's chameleon is the largest of the chameleons not native to Ma...  |  <urn:uuid:9c71b6db-6728-48b5-96b5-05fbc0b5bb4f>  |  CC-MAIN-2013-20  |  http://amazingpicturesoftheanimals.blogspot.com/2012/05/mellers-chameleon-facts-pictures.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.939643   |  3.40625   |  
|  Nuclear Energy in France Nuclear energy is the cornerstone of french energy policy. In the ‘70s France chose to develop nuclear as its base load electricity source as a response to the oil crisis and assure its energy independence. Nuclear Electricity Production: France currently counts 58 commercial nuclear reactors i...  |  <urn:uuid:f5c220a7-7276-4cf2-9208-33679d478b1f>  |  CC-MAIN-2013-20  |  http://ambafrance-us.org/spip.php?article949&xtor=AL-13  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.912335   |  1,305   |  3.125   |  
|  Mexican America - Introduction "Mexican America" is a sampling of objects from the collections of the National Museum of American History. The stories behind these objects reflect the history of the Mexican presence in the United States. They illustrate a fundamentally American story about the centuries-old encounter b...  |  <urn:uuid:ff577d1a-83b8-467c-af1c-4c0aa2ead4fb>  |  CC-MAIN-2013-20  |  http://americanhistory.si.edu/collections/object-groups/mexican-america?edan_start=0&edan_fq=date%3A%221840s%22  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.776227   |  1,938   |  4.0625   |  
|  - Action research (6 posts) - Artist CPD (11 posts) - Barriers to participation (2 posts) - Change management (8 posts) - Co-construction (3 posts) - Community cohesion (12 posts) - Creative curriculum development (13 posts) - Creative teaching and learning (28 posts) - Cross-curricular working (21 posts) - Developing ...  |  <urn:uuid:69d15397-ebe5-4147-830d-84d945741e63>  |  CC-MAIN-2013-20  |  http://anewdirection.org.uk/knowledge/resources?category=370  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.888793   |  2.640625   |  
|  White-throated Magpie-Jays (Calocitta formosa) are beautiful big jays that travel the North Pacific slopes in small flocks. Their songs and calls are quite varied - this is one of the typical calls, recorded on the road to Monteverde (Costa Rica). Douglas Von Gausig (recordist; copyright holder), Naturesongs.com This w...  |  <urn:uuid:e5ac96fc-a147-40c0-a0ba-bc1fa8515745>  |  CC-MAIN-2013-20  |  http://animaldiversity.ummz.umich.edu/accounts/Bilateria/sounds/collections/contributors/naturesongs/wtmj1/?start=90  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.852656   |  2.921875   |  
|  Octodon degus is generally considered endemic to west central Chile, where it inhabits the lower slopes of the Andes. Although some have argued that its range may extend north into Peru, this is not well supported. It is common in the international pet trade, however, and is often used in laboratory studies outside of ...  |  <urn:uuid:2653877c-a97a-4524-a9e3-91af93f1f619>  |  CC-MAIN-2013-20  |  http://animaldiversity.ummz.umich.edu/site/accounts/information/Octodon_degus.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.928159   |  3,982   |  3.0625   |  
|  Details of Glycemic Index (GI) The GI Scale The glycemic index uses a scale from 1 to 100, which indicates the rate at which 50 grams of carbohydrate in a particular food is absorbed into the bloodstream as blood-sugar. The main reference food (rated 100) is glucose. GI Rating Categories The glycemic index divides carb...  |  <urn:uuid:17b26358-fba0-4434-86b5-ce1458abe71f>  |  CC-MAIN-2013-20  |  http://annecollins.com/gi-food-guide.htm  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.909349   |  3.75   |  
|  Discover the cosmos! Each day a different image or photograph of our fascinating universe is featured, along with a brief explanation written by a professional astronomer. 2010 August 12 Explanation: Each August, as planet Earth swings through dust trailing along the orbit of periodic comet Swift-Tuttle, skygazers can ...  |  <urn:uuid:37377bba-5d22-4f0b-91a9-cd45df72de7c>  |  CC-MAIN-2013-20  |  http://apod.nasa.gov/apod/ap100812.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.883452   |  2.875   |  
|  1. Why is extremism an issue in prisons? Extremist groups often pose special security risks in prisons. They may encourage the overthrow of the government, and prison officials can be targeted as agents of "illegal" government authority. Further, their literature often encourages ethnic hatred, promoting a violent and ...  |  <urn:uuid:4b29297a-7223-4dae-a9fc-a294677b62b3>  |  CC-MAIN-2013-20  |  http://archive.adl.org/civil_rights/prison_ex.asp  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.950167   |  2,202   |  2.796875   |  
|  Wikipedia sobre física de partículas Rapidinho. Me falaram que a definição de física de partículas da Wikipedia era muito ruim. E de fato, era assim: Particle physics is a branch of physics that studies the elementary particle|elementary subatomic constituents of matter and radiation, and their interactions. The field ...  |  <urn:uuid:e7f0a003-07f1-4148-a77c-6e0cb215fc0e>  |  CC-MAIN-2013-20  |  http://arsphysica.wordpress.com/2011/08/14/wikipedia-sobre-fisica-de-particulas/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.65767   |  
|  - published: 19 Mar 2013 - views: 42 - author: T.A. B possibly testing on weans, that worries me http://www.bbc.co.uk/news/world-us-canada-21849808. A vaccine is a biological preparation that improves immunity to a particular disease. A vaccine typically contains an agent that resembles a disease-causing microorganism,...  |  <urn:uuid:049ed48d-f01e-4fc9-846b-2e2c5e6c254d>  |  CC-MAIN-2013-20  |  http://article.wn.com/view/2013/01/16/Vaccine_timetable_for_children_is_safe_US_experts_say_t/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.945185   |  4,705   |  3.78125   |  
|  Wars have given us the Jeep, the computer and even the microwave. Will the war in Iraq give us the Tiger? Military scientists at Edgewood Chemical Biological Center at Aberdeen Proving Ground hope so. The machine - its full name is the Tactical Garbage to Energy Refinery - combines a chute, an engine, chemical tanks an...  |  <urn:uuid:749f4f2e-01bf-42ab-ac03-1dfa84af34dc>  |  CC-MAIN-2013-20  |  http://articles.baltimoresun.com/2008-07-21/news/0807200131_1_garbage-aberdeen-proving-ground-war-in-iraq  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.959887   |  1,205   |  2.828125   |  
|  Then and now, there is ample proof that Americans do take Supreme Court nominations seriously. With good reason. Sooner or later, the nation's most vexing disagreements over our most vital issues wind up before the Supreme Court. None quite penetrates to the core of our democratic being more than those involving First ...  |  <urn:uuid:9b2b1d71-d1b0-4968-9420-c92bddd3e0ae>  |  CC-MAIN-2013-20  |  http://articles.centralkynews.com/2005-10-13/news/24908337_1_decisive-fifth-vote-chief-justice-john-roberts-first-amendment  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.959613   |  2.703125   |  
|  The vaunted protection that intellectually active adults get from Alzheimer’s disease has a dark downside, a study released Wednesday has found. Once dementia symptoms become evident and Alzheimer’s disease is diagnosed in such patients, their mental decline can come with frightening speed. That finding, published in t...  |  <urn:uuid:5d156165-181a-4195-a926-d51850c7b599>  |  CC-MAIN-2013-20  |  http://articles.latimes.com/2010/sep/01/news/la-heb-alzheimers-20100901  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.958416   |  2.71875   |  
|  Belgian physicist Francois Englert, left, speaks with British physicist… (Fabrice Coffrini / AFP/Getty…) For physicists, it was a moment like landing on the moon or the discovery of DNA. The focus was the Higgs boson, a subatomic particle that exists for a mere fraction of a second. Long theorized but never glimpsed, t...  |  <urn:uuid:fb237ffb-9cc0-4077-99d5-56c6fce1ca5f>  |  CC-MAIN-2013-20  |  http://articles.latimes.com/2012/jul/05/science/la-sci-higgs-boson-new-particle-20120705  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.963451   |  1,134   |  2.59375   |  
|  Discrimination against citizens because of their sexual orientation or their youth is just as unfair as discrimination based on race, religion or gender. Access to housing, hotels, theaters and restaurants should be equal to all, as a matter of plain fairness. The Palm Beach County Commission recognized this truth in p...  |  <urn:uuid:04a92c65-e194-4b3c-ae6e-d4a7b6b1631d>  |  CC-MAIN-2013-20  |  http://articles.sun-sentinel.com/1990-01-19/news/9001190945_1_ordinance-public-accommodations-housing  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.959891   |  2.6875   |  
|  "Helplessness" and "confusion" are words that easily come to mind when the issue of sick building syndrome is mentioned. It is a problem that does not have a regulatory solution, and is bound with engineering, medicine and emotions that will challenge the best of school administrators. A careful management style and kn...  |  <urn:uuid:f673350f-8b06-4580-b54b-a3217aacfede>  |  CC-MAIN-2013-20  |  http://asumag.com/mag/university_prescribing_cure  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.938972   |  1,048   |  3.40625   |  
|  Ethiopian kids hack Zoom tablets in five months. The One Laptop Per Child (OLPC) project recently tried a new approach to distributing computers to developing countries, by leaving boxes of tablets in remote Ethiopian villages with no instructions. The Motorola Zoom tablets had a custom English language OS, a solar cha...  |  <urn:uuid:9dc0e204-f56c-43cf-b426-74e04dd58130>  |  CC-MAIN-2013-20  |  http://augustobre.tumblr.com/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.969853   |  2.796875   |  
|  The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose. Elizabeth’s refusal of Mr. Collins offer of marriage showed an independence seldom seen in heroines of the day. Her refusal of Mr....  |  <urn:uuid:0d8a309d-25c5-405d-a08a-c11239f0d717>  |  CC-MAIN-2013-20  |  http://austenauthors.net/the-independent-jane  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.97432   |  2.75   |  
|  "GOT NOTHING BUT BLUE SKIES" It is September 19,1783. The place, Lyons, France. Preparations are being made for a journey. A journey that will eventually take man from his secure environment of terra firma, and place him in a hostile environment called the atmosphere. The vehicle to be used is a hot air balloon. The br...  |  <urn:uuid:ac6a19cf-dd31-4352-bb69-1c00f45050a7>  |  CC-MAIN-2013-20  |  http://avstop.com/stories/blueskies.htm  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.952665   |  1,401   |  3.46875   |  
|  Here to There: A History of Mapping From the 16th to 18th centuries, many European mapmakers were convinced that California was an island — an Edenic paradise populated by black Amazons. The error persisted for over a hundred years after expeditions had proven that California was, in fact, firmly attached to the mainla...  |  <urn:uuid:ee4268b8-d3ab-4049-a3ac-f695c938a3ca>  |  CC-MAIN-2013-20  |  http://backstoryradio.org/here-to-there-a-history-of-mapping/quote-comment-21383/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.949669   |  3.390625   |  
|  Nursing a critically ill state back to health |Indranill Basu Ray highlights the core problems that afflict Bengal's health sector and suggests a few ways to improve the situation| Despite many technological and other achievements that have propelled India from being a developing nation to one of the top economies of t...  |  <urn:uuid:a51737a0-6a1a-4721-a739-791f50bfecba>  |  CC-MAIN-2013-20  |  http://basantipurtimes.blogspot.com/2011/11/nursing-critically-ill-state-back-to.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.959302   |  2,399   |  2.703125   |  
|  A “magic” herb, Carissa Edulis, that drew thousands of people to a remote Loliondo village in Tanzania was identified by Kenyan scientists a few years ago as a cure for a drug-resistant strain of a sexually transmitted disease, gonorrhoea. This herb also is believed to cure many other diseases besides gonorrhoea. The K...  |  <urn:uuid:f03bdf02-ee92-4cf8-bf3e-b8881d16ad7a>  |  CC-MAIN-2013-20  |  http://beamsdoorway.bizland.com/wordpress/?tag=tolo  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.945648   |  2.875   |  
|  The Convention adjourned from July 26th to August 6th to allow the Committee of Detail – composed of John Rutledge of South Carolina, Edmund Randolph of Virginia, Nathaniel Gorham of Massachusetts, Oliver Ellsworth of Connecticut, and James Wilson of Pennsylvania – to prepare a rough draft of a constitution, based on t...  |  <urn:uuid:ea2a4a62-5765-4ec7-ba52-1a99029763a0>  |  CC-MAIN-2013-20  |  http://billofrightsinstitute.org/blog/2011/08/15/countdown-to-the-constitution-a-rough-draft-of-the-constitution/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.963835   |  1,246   |  3.09375   |  
|  The concept of a numbering machine is quite simple. It consists of various wheels inside and all wheels are stamped with numbers. But, what could be done with this? Well, the wheels of a numbering machine can be set and arranged to form a date or various other numbers. Basically numbers machines are used to imprint ite...  |  <urn:uuid:34a18b86-5717-4716-8eed-650c970bbd9c>  |  CC-MAIN-2013-20  |  http://blog.acornsales.com/tag/stationery/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.956298   |  2.828125   |  
|  Did You Know? Facts, Figures & Folklore About Fall / Autumn Sept 18 : 04 days to Fall Did you know that the word autumn comes from the French word, automne? The word came into common usage around the 16th century, before which “harvest” was used to refer to this time of year. The use of harvest fell out of popularity a...  |  <urn:uuid:a787fa6b-c56e-4652-88cc-84e18746f986>  |  CC-MAIN-2013-20  |  http://blog.holidays.net/index.php/2010/09/18/did-you-know-countdown-to-fall-0918/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.965644   |  2.890625   |  
|  In 2006, two scientists announced that they had cooked an egg by placing it in between two cell phones. It has been thrououghly disproven and analyzed since the surface of the claim, but it is still an excellent opportunity to use the Scientific Thinking Principles on! #1: Ruling Out Rival Hypotheses This principle isn...  |  <urn:uuid:87230adf-bb1f-4b4b-9c4e-ebaed1589758>  |  CC-MAIN-2013-20  |  http://blog.lib.umn.edu/hamdi002/blog/2011/09/cooking-an-egg-between-two-cell-phones-not-the-best-way-to-get-out-of-dorm-food.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.974436   |  2.875   |  
|  How To Win At Science Fairs (Dec, 1960) How To Win At Science Fairs by Ronald Benrey YOU CAN WIN at a Science Fair as long as one thing interests you more than winning does. This is your project itself. It is going to be judged on scientific thought, creative ability, and presentation. You will really have to know the ...  |  <urn:uuid:6ad66c1d-7e72-4ed4-889b-c4b7f1b988a7>  |  CC-MAIN-2013-20  |  http://blog.modernmechanix.com/how-to-win-at-science-fairs/1/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.945775   |  1,357   |  2.890625   |  
|  With the development of science and technology, computer has become more and more popular in our daily life, which is intended to be a part of our life. But at the same time it also brings the safety problem, because increasing number of bad people would like to break into computer systems to steal the secret informati...  |  <urn:uuid:3d310cea-07c0-49a6-a372-5da020bfac96>  |  CC-MAIN-2013-20  |  http://blog.resetwindowspassword.com/tag/antivirus  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.952135   |  3.1875   |  
|  Ever wondered what white tea is? Why we call it white anyway? Here are a few interesting facts about white tea, made from the plant Camellia sinensis: - White tea is similar to green tea in that it has undergone little processing and no fermentation whatsoever. - The leaves are harvested before they open fully, when th...  |  <urn:uuid:6d64f04d-9635-43ad-b483-874653a0280b>  |  CC-MAIN-2013-20  |  http://blog.savorique.com/tag/white-tea-health-benefits/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.922025   |  2.59375   |  
|  .8grams X kilogram of body weight/day How to convert pounds into kilograms: #pounds/2.2 = kg For example someone who weighs say 110 would be 50kg. So we take 50kg and multilpy that by .8gm and they should be getting about 40 grams of protein a day. Low protein diets can result in weak bones, muscle wasting, and decreas...  |  <urn:uuid:549e680b-1e0f-40fd-b7f5-2a96abd000ee>  |  CC-MAIN-2013-20  |  http://blog.themaddiet.com/2012/02/protein-and-why-it-is-important.html  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.962502   |  2.609375   |  
|  Our opinion: Fostering better nutrition is a key to reducing public health costs. That makes the scarcity of large grocery stores in some urban neighborhoods, especially poor ones, a matter of public policy. It’s easy, and comforting, to assume that in an modern, industrialized nation, everyone eats well. Surely with p...  |  <urn:uuid:fc2b2237-bb58-4d64-ab60-655ff3250f48>  |  CC-MAIN-2013-20  |  http://blog.timesunion.com/opinion/green-up-%E2%80%A8urban-deserts/16365/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.962732   |  2.65625   |  
|  Yesterday’s House passage of cap-and-trade legislation designed to confront climate change is a landmark achievement, the first tangible step taken by the country that emits more greenhouse gas per capita than anyone in the world. The bill itself still faces a tough test in the Senate. Passage is far from assured, and ...  |  <urn:uuid:dbaccb35-1074-476c-9b46-8f3c424bfecd>  |  CC-MAIN-2013-20  |  http://blogs.ajc.com/jay-bookman-blog/2009/06/27/passage-of-climate-change-bill-a-landmark-achievement/?cp=all  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.950843   |  2.5625   |  
|  Just as there are many variants and forms of electronic malware and Internet-based threats around the globe, so there are many forms of protection against these threats. Signature-based detection is one of the multifarious forms of defense that have been developed in order to keep us safe from malicious content. Althou...  |  <urn:uuid:1dc7b057-7b64-4e02-98fa-2e3e2cfcf30c>  |  CC-MAIN-2013-20  |  http://blogs.avg.com/business/signature-based-detection/  |  s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz  |  0.965388   |  3.015625   |  
End of preview. [Expand in Data Studio](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/default/train)
#  📚 FineWeb-Edu 
> 1.3 trillion tokens of the finest educational data the 🌐 web has to offer
**Paper:** <https://arxiv.org/abs/2406.17557>
##  What is it? 
📚 FineWeb-Edu dataset consists of **1.3T tokens** and **5.4T tokens** ([FineWeb-Edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)) of educational web pages filtered from 🍷 FineWeb dataset. This is the 1.3 trillion version.
To enhance FineWeb's quality, we developed an [educational quality classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) using annotations generated by LLama3-70B-Instruct. We then used this classifier to retain only the most educational web pages. FineWeb-Edu outperforms FineWeb on popular benchmarks and shows the power of classifiers trained on synthetic data. 
The [Dataset Curation](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#dataset-curation) section details the process for creating the dataset.
You can find a deduplicated version of FineWeb-edu in [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). We find that the deduplication of this dataset doesn't have any impact on model performance in our ablation setup (1.8B trained on 350B tokens).
##  What is being released? 
Along with the dataset, which includes all filtered CommonCrawl dumps since 2013, we also release the educational classifier used for the filtering as well as the code for training it and running inference at: <https://github.com/huggingface/cosmopedia/tree/main/classification>
##  Changelog 
_Previous versions remain available in the branch`version name`._
  * **v1.4.0 (11-07-2025):** Added 6 new snapshots: `CC-MAIN-2025-05`, `CC-MAIN-2025-08`, `CC-MAIN-2025-13`, `CC-MAIN-2025-18`, `CC-MAIN-2025-21`, and `CC-MAIN-2025-26` (January to June 2025)
  * **v1.3.0 (31-01-2025):** Fixed an issue with some dumps where some documents hadn't been processed: `CC-MAIN-2024-10`, `CC-MAIN-2024-18`, `CC-MAIN-2024-22`, `CC-MAIN-2024-26`, `CC-MAIN-2024-30`, `CC-MAIN-2024-33`, `CC-MAIN-2024-38`, `CC-MAIN-2024-42`, `CC-MAIN-2024-46` -- they now contain more data (~35B additional tokens).
  * **v1.2.0 (03-01-2025):** Added 9 new snapshots: `CC-MAIN-2024-18`, `CC-MAIN-2024-22`, `CC-MAIN-2024-26`, `CC-MAIN-2024-30`, `CC-MAIN-2024-33`, `CC-MAIN-2024-38`, `CC-MAIN-2024-42`, `CC-MAIN-2024-46`, `CC-MAIN-2024-51`, covering April to December 2024.
  * **v1.0.0 (02-06-2024):** Initial version


##  How to load the dataset 
Similarily to FineWeb, You can load the full dataset or a specific crawl/dump. Dumps have the format `CC-MAIN-(year)-(week number)`.
###  (Smaller) sample versions 
Along with config `default` (all the data), and the configs for each individual dump, you can also download the following configs:
  * `sample-350BT`: a subset randomly sampled from the whole dataset of around 350B gpt2 tokens 
  * `sample-100BT`: a subset randomly sampled from the whole dataset of around 100B gpt2 tokens 
  * `sample-10BT`: a subset randomly sampled from the whole dataset of around 10B gpt2 tokens


`sample-10BT` was sampled from `sample-100BT` which in turn was sampled from `sample-350BT`.
###  Using 🏭 [`datatrove`](https://github.com/huggingface/datatrove/)

```
from datatrove.pipeline.readers import ParquetReader

# limit determines how many documents will be streamed (remove for all)
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet", limit=1000)
# or to fetch a specific dump CC-MAIN-2024-10,  eplace "CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/CC-MAIN-2024-10", limit=1000) 
for document in data_reader():
    # do something with document
    print(document)

###############################    
# OR for a processing pipeline:
###############################

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline_exec = LocalPipelineExecutor(
    pipeline=[
        # replace "CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/CC-MAIN-2024-10", limit=1000),
        LambdaFilter(lambda doc: "hugging" in doc.text),
        JsonlWriter("some-output-path")
    ],
    tasks=10
)
pipeline_exec.run()

```

###  Using `datasets`

```
from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)

```

##  Dataset curation 
A new approach has recently emerged for filtering LLM training datasets: using synthetic data to develop classifiers for identifying educational content. This technique was used in the trainings of [LLama3](https://ai.meta.com/blog/meta-llama-3-meta-ai-responsibility/) and [Phi3](https://arxiv.org/abs/2404.14219), but its large-scale impact on web data filtering hasn't been fully explored or published.
The highly popular Phi3 models were trained on 3.3 and 4.8 trillion tokens, with the paper stating: “Our training data consists of heavily filtered publicly available web data (according to the 'educational level') from various open internet sources, as well as synthetic LLM-generated data". Similarly, the LLama3 blog post notes: “We found that previous generations of Llama are good at identifying high-quality data, so we used Llama 2 to help build the text-quality classifiers that are powering Llama 3.” However these classifiers and filtered datasets are not publicly available. To enhance FineWeb's quality, we developed an educational quality classifier using annotations generated by [LLama3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to create FineWeb-Edu.
###  Annotation 
We used [Llama3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to score 500k FineWeb samples for their educational quality on a scale from 0 to 5.
We explored various prompts and found that the additive scale by [Yuan et al.](https://arxiv.org/pdf/2401.10020) worked best. To avoid the LLM favoring highly technical pages like arXiv abstracts and submissions, we focused on grade-school and middle-school level knowledge. By setting a threshold of 3 (on a scale of 0 to 5) during the filtering process, we were able to also retain some high-level educational pages. The final prompt can be found [here](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/utils/prompt.txt).
We also experimented with different LLMs: Llama3-70B-Instruct, Mixtral-8x-7B-Instruct, and Mixtral-8x22B-Instruct. Llama 3 and Mixtral-8x22B produced similar scores, while Mixtral-8x7B tended to be more generous, not fully adhering to the score scale. Verga et al. suggest using multiple LLMs as juries. We tried averaging the scores from the three models, but this shifted the distribution to the right due to the higher scores from Mixtral-8x7B. Training on a dataset filtered with a classifier using jury annotations performed worse than using a classifier based on Llama3 annotations. We hypothesize that the jury-based approach retains more low-quality samples.
###  Classifier training 
We fine-tuned a Bert-like regression model using these annotations, based on [Snowflake-arctic-embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m). When converted to a binary classification using a score of 3 as a threshold for keeping and removing files, the model achieved an F1 score of 82%. The classification of FineWeb 15T tokens took 6k H100 GPU hours.
The classifier is available at: [HuggingFaceFW/fineweb-edu-classifier/](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/)
###  Filtering and results 
**Note** : You can find more details about the ablations and results in the FineWeb [blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
We investigated the impact of using different thresholds for the filtering and found that threshold 3 gave the best overall results. Although using a threshold higher than 3 improves performance on knowledge and reasoning intensive benchmarks, it significantly degrades performance on HellaSwag and PIQA.
We then built 📚 FineWeb-Edu by filtering out samples with scores lower than 3. This removed 92% of the dataset, leaving us with 1.3T educational tokens. Our ablation demonstrated that this refined dataset surpasses 🍷 FineWeb and all other open web datasets, with remarkable improvements on educational benchmarks such as MMLU, ARC, and OpenBookQA. The plot below compares FineWeb-Edu to other web datasets:
To retain more tokens, we also experimented with a less strict threshold of 2 instead of 3. While being less performant than using threshold 3, it still outperformed FineWeb and it preserved 5.4T tokens. We release these two dataset as [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [FineWeb-Edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) along with the [classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier).
You will find all the ablation models in [this collection](https://huggingface.co/collections/HuggingFaceFW/ablation-models-662457b0d213e8c14fe47f32). The FineWeb-Edu ablation model (trained on 350B tokens) is available at <https://huggingface.co/HuggingFaceFW/ablation-model-fineweb-edu>.
##  Considerations for Using the Data 
This section is copied from the parent dataset: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
###  Social Impact of Dataset 
With the release of this dataset we aim to make model training more accessible to the machine learning community at large. 
While multiple open-weights models with strong performance have been publicly released in the past, more often than not these releases are not accompanied by the corresponding training dataset. This is unfortunate as the dataset specificities and characteristics have been demonstrated to have a very large impact and role in the performances of the models. As the creation of a high quality training dataset is a fundamental requirement to training an LLM capable of excelling at downstream tasks, with 🍷 FineWeb we (a) not only make the dataset creation process more transparent, by sharing our entire processing setup including the codebase used, we also (b) help alleviate the costs of dataset curation, both in time and in compute, for model creators by publicly releasing our dataset with the community.
###  Discussion of Biases 
Efforts were made to minimize the amount of NSFW and toxic content present in the dataset by employing filtering on the URL level. However, there are still a significant number of documents present in the final dataset that could be considered toxic or contain harmful content. As 🍷 FineWeb was sourced from the web as a whole, any harmful biases typically present in it may be reproduced on our dataset.
We deliberately avoided using machine learning filtering methods that define text quality based on the similarity to a “gold” source such as wikipedia or toxicity classifiers as these methods have been known to [disproportionately remove content in specific dialects](https://aclanthology.org/D16-1120/) and [overclassify as toxic text related to specific social identities](https://arxiv.org/pdf/2109.07445.pdf), respectively.
###  Other Known Limitations 
As a consequence of some of the filtering steps applied, it is likely that code content is not prevalent in our dataset. If you are training a model that should also perform code tasks, we recommend you use 🍷 FineWeb with a code dataset, such as [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2). You should also probably consider complementing 🍷 FineWeb with specialized curated sources (such as Wikipedia, for example) as they will likely have better formatting than the wikipedia content included in 🍷 FineWeb (we did not tailor the processing to individual websites).
##  Additional Information 
###  Licensing Information 
The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0** [license](https://opendatacommons.org/licenses/by/1-0/). The use of this dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use).
###  Future work 
We plan to work on better educational classifier to improve the quality of FineWeb-Edu.
###  Citation Information 
You can cite our paper <https://arxiv.org/abs/2406.17557> or this dataset:

```
@misc{lozhkov2024fineweb-edu,
    author       = { Lozhkov, Anton and Ben Allal, Loubna and von Werra, Leandro and Wolf, Thomas },  
    title        = { FineWeb-Edu: the Finest Collection of Educational Content }, 
    year         = 2024,  
    url          = { https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu },  
    doi          = { 10.57967/hf/2497 },
    publisher    = { Hugging Face }
}

```

Copy to bucket new
Use this dataset  

Downloads last month
    435,452
Number of rows: 3,496,736,741  Total file size: 5.84 TB 
##  Models trained or fine-tuned on HuggingFaceFW/fineweb-edu
[ Text Generation • 0.1B • Updated Aug 5 • 102k • 63  ](https://huggingface.co/SupraLabs/Supra2-100M-Instruct)
[ Image-Text-to-Text • 7B • Updated Aug 24 • 31.2k • 7  ](https://huggingface.co/nicosuter/Qwen3.8-27B-AWQ)
[ Text Generation • 8B • Updated Oct 22, 2025 • 30.4k • 85  ](https://huggingface.co/BSC-LT/salamandra-7b-instruct)
[ Text Generation • Updated 19 days ago • 21.7k • 239  ](https://huggingface.co/BlinkDL/rwkv7-g1)
[ Text Generation • 7B • Updated Dec 17, 2024 • 11.7k • 248  ](https://huggingface.co/tiiuae/falcon-mamba-7b)
[ Fill-Mask • 0.4B • Updated 11 days ago • 10.8k • 74  ](https://huggingface.co/dmis-lab/UltraBERT-380M)
[Browse 1504 models trained on this dataset](https://huggingface.co/models?dataset=dataset:HuggingFaceFW/fineweb-edu)
##  Spaces using HuggingFaceFW/fineweb-edu 24
##  Collections including HuggingFaceFW/fineweb-edu
[ FineWeb-Edu datasets, classifier and ablation model • 5 items • Updated Jun 12, 2024 • 21 ](https://huggingface.co/collections/HuggingFaceFW/fineweb-edu)
[ 7 items • Updated Jun 20, 2025 • 36 ](https://huggingface.co/collections/HuggingFaceFW/fineweb)
##  Papers for HuggingFaceFW/fineweb-edu
[ Paper • 2406.17557 • Published Jun 25, 2024 • 108 ](https://huggingface.co/papers/2406.17557)[ Paper • 2404.14219 • Published Apr 22, 2024 • 262 ](https://huggingface.co/papers/2404.14219)[ Paper • 2401.10020 • Published Jan 18, 2024 • 156 ](https://huggingface.co/papers/2401.10020)[ Paper • 2109.07445 • Published Sep 15, 2021 • 3 ](https://huggingface.co/papers/2109.07445)
