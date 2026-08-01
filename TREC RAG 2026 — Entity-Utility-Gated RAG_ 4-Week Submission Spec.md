# Tab 1

# **TREC RAG 2026: Entity-Utility-Gated RAG**

**Deadline: Aug 8, 2026** (\~4 weeks). Test topics released Jul 6\. Corpus: **ClimbMix-400b** via **Pyserini REST API**. Task: system-owned retrieval → sentence-cited grounded answer. Eval: **pairwise system battles** on usefulness, coverage, grounding, citation quality, and *absence of misleading claims* (nuggets used diagnostically). Tools: **RAGDoll** \+ dev data (RAG25 nuggets, UMBRELA qrels, ResearchRubrics).

> **What this is:** a *system* submission \+ notebook paper that deploys the paper's **CEU^sel** idea (utility of an entity as a retrieval/selection signal) as a live pipeline, and validates the framework "in the wild." **Not** part of the v4 evaluation resource (that stays on TREC 2025 / MS MARCO; ClimbMix is incompatible and 2026 runs don't exist yet). Designed as a **bolt-on module** to whatever RAG pipeline you're already building.

## **Why the framework fits this track**

The battle rubric rewards exactly what entity-utility gating produces: **coverage** (surface vital entities), **citation quality** (entity-specific attribution — your ENC-citation), and **penalizes misleading claims** (your high-observable / low-utility "misleading-signal" entities). You get a principled reason your run should win battles that most participants lack.

## **Architecture (module you add to a standard RAG pipeline)**

narrative q  
  │  
  ├─(1) decompose q → sub-questions/aspects  A(q)               \[LLM, 1 call/topic\]  
  ├─(2) candidate entities: link(q) ∪ generate(q)               \[REL/WAT \+ LLM\]  
  │  
  ├─(3) ENTITY-UTILITY SCORING  (the contribution) ───────────────┐  
  │      for each candidate entity e:                              │  
  │        observable(e): link conf, corpus freq/discriminativeness│  
  │        utility(e)   : CEU^sel proxy (below)                    │  
  │        flag MISLEADING if observable high ∧ utility low        │  
  │                                                                │  
  ├─(4) utility-gated retrieval:                                   │  
  │        base run: BM25(q) via Pyserini REST                     │  
  │        entity-conditioned runs: BM25(q ⊕ high-utility e)       │  
  │        RRF-fuse base \+ high-utility-entity runs                │  
  │        EXCLUDE evidence brought in only by misleading entities │  
  │                                                                │  
  ├─(5) grounded generation \+ entity-aware citation:              │  
  │        generate sentence-level answer over fused evidence      │  
  │        each surfaced entity must be supported by its citation  │  
  │        drop/repair sentences citing unsupported entities       │  
  │                                                                │  
  └─(6) format → rag\_output\_trec\_rag\_2026.jsonl → validate (RAGDoll)

## **The CEU^sel proxy under time pressure (pick by budget)**

Full controlled CEU is out (no time). Use a **fast proxy**, and say so plainly in the notebook — it's a judged/retrieval approximation, not the causal estimand.

* **Retrieval-delta (cheap, no per-entity generation):** utility(e) ≈ how much entity-conditioning improves aspect coverage of the evidence pool. Concretely: does BM25(q ⊕ e) retrieve evidence covering sub-questions in A(q) that BM25(q) misses? Score \= new-aspect coverage gain (LLM checks each top passage against A(q)).  
* **LLM utility gate (fast, 1 call/entity):** given A(q) and e, ask "is e vital / helpful / misleading for answering these sub-questions?" A judged-vitality proxy. Use to assign the vital/okay/misleading flag.  
* **Answer-grounding LOO (expensive — top-k entities only):** for the few highest-observable entities, draft an answer, drop e's evidence bundle, regenerate the affected sentences, check quality drop — a targeted misleading-entity catch. Reserve for the final gate, not all entities.

**Recommended combo:** decompose → LLM utility gate \+ retrieval-delta for the utility score; run the answer-grounding LOO only on the top-k high-observable entities to catch misleading signal. Exclude high-observable/low-utility entities from steering and from citation.

## **Runs to submit (multiple runs allowed → built-in ablation)**

1. **`baseline`** — BM25(q) \+ vanilla grounded generation. (Control.)  
2. **`entity-expand`** — entity-conditioned retrieval \+ RRF, *no* utility gating. (Isolates expansion.)  
3. **`entity-utility-gated`** — full: utility-scored entities steer retrieval, misleading entities excluded, entity-aware citation. (The contribution.)

This ablation *is* your notebook-paper experiment and a preliminary signal for the entity-nugget thesis: does utility gating beat naive entity expansion and baseline in battles / RAGDoll dev scoring?

## **4-week timeline**

* **Week 1 — infra \+ valid baseline early.** Register/confirm **ClimbMix Pyserini REST access** (critical path — do day 1). Ingest `trec_rag_2026_queries.jsonl`. Stand up RAGDoll on the dev topics. Ship an end-to-end **baseline** run that validates — so you have *a* submission in hand. Build entity generation+linking.  
* **Week 2 — utility scoring \+ gated retrieval.** Sub-question decomposition; LLM utility gate \+ retrieval-delta; misleading flag; entity-conditioned BM25 \+ RRF fusion; misleading-evidence exclusion. Produce `entity-expand` and `entity-utility-gated` evidence.  
* **Week 3 — entity-aware generation \+ internal eval.** Sentence-level cited generation with entity-support checking; targeted answer-grounding LOO on top entities. Score all three runs on RAGDoll dev nuggets / ResearchRubrics; iterate.  
* **Week 4 — finalize.** Tune λ/thresholds on dev topics, validate all three runs' JSONL, submit (set priority ordering), draft the notebook-paper skeleton.

## **Critical-path blockers (do these first)**

1. **ClimbMix / Pyserini REST access** requires TREC registration in Evalbase — the whole run is blocked without it. Confirm the endpoint works day 1\.  
2. **RAGDoll setup on dev data** — your only internal signal (battle scores come post-submission); get it scoring before Week 3\.  
3. **Entity linking on ClimbMix** — REL/WAT link to Wikipedia/Wikidata; confirm your `common/ids.py` normalization works on ClimbMix doc text.

## **Discipline / honesty notes**

* The proxy is a **proxy**: don't claim causal CEU from the fast gate; the notebook says "utility-guided" not "causally validated."  
* Keep the two-object rule even here: this run is a *system*; ENC-style attribution checking inside generation is a design choice, not the evaluation resource.  
* **Bonus for later:** unlike participant runs, *you own this pipeline* — so your 2026 run is legitimately ablatable, making it valid controlled-CEU material after the deadline. And once 2026 nuggets/judgments release (post-conference), the entity-nugget resource can extend to ClimbMix. Position now, harvest later.

## **Verify before building**

* Confirm the **exact submission mechanics/priority rules** and whether a subset submission is allowed, from the 2026 guidelines skill.  
* Confirm **baselines release** (soon after Jul 6\) — a released baseline may shortcut Week 1\.  
* Confirm **RAGDoll** covers the metrics you want for the dev ablation (nugget coverage \+ citation support).

# Tab 2

# **Entity-Utility-Gating Module**

## **Implementation Spec** 

**Supersedes v1.** Rebuilt per review: aspect-first pipeline, **branch-level** gating (not document exclusion), greedy budgeted selection, identical-across-runs harness, and **Û^sel** naming. Codeable now on RAG25 **dev topics** via local Pyserini; ClimbMix only needed for the final test runs.

> **Naming discipline.** The live gate is **Û^sel** \= *estimated entity-selection utility*, inspired by CEU but **not** causal. The notebook claims marginal-utility-gated expansion \+ entity-role citation verification \+ controlled comparison — never "deployed CEU." Controlled CEU^sel is a *post-submission* ablation (§11).

> **VERIFY AGAINST LIVE REPO before relying (deadline-critical, secondhand snapshot):**

> * **119 topics** (`rag2026-0`…`rag2026-118`), highly heterogeneous (medical, policy, ethics, creative writing) — many **not** entity-centric.  
> * Test file currently **TSV** (`trec_rag_2026_queries.tsv`), not JSONL — support both, preserve IDs exactly.  
> * Deadline **Aug 7 vs Aug 8** conflict — treat **Aug 7 as hard**; confirm on Evalbase \+ mailing list.  
> * **Run limits / priority rules unconfirmed** — three runs is an *internal* experiment; submit whatever's allowed.  
> * **RAGDoll Arena doesn't render citations to the judge** — approximates usefulness, not citation quality; run support assessment separately.  
> * Corpus index `climbmix-400b` via **Pyserini REST**, needs an **access token** (request via Pyserini access group).  
> * Run type: fully system-generated \= `automatic`; per-topic human editing \= `manual`.  
  ---

  ## **Pipeline (aspect-first)**

* topic (title, narrative)  
*  (1) PLAN: aspects A(q)={a\_i}, weights w\_a, requested entities, per-aspect queries  
*  (2) ASPECT RETRIEVAL → R\_0  (title \+ per-aspect queries)   ← the TRUE baseline  
*  (3) CANDIDATE ENTITIES E(q) \= E\_query ∪ E\_first-pass-evidence ∪ E\_LLM  
*  (4) Û^sel(e): marginal aspect-coverage gain − redundancy − drift − source-risk  
*  (5) GREEDY branch selection under budget B; action ∈ {promote, neutral, suppress}  
*  (6) UTILITY-WEIGHTED fusion of base+aspect+selected-entity branches  
*      → dedup → aspect quotas → global rerank vs full narrative  
*  (7) GENERATE sentence-cited answer → verify (general support \+ entity-role support) → repair


Abstract/non-entity topics simply stay at R\_0 (aspect-only) — the fallback is built in.

## **Data objects**

* Aspect \= {"id","text","weight","query"}          \# per-aspect concise query  
*   
* @dataclass  
* class CandidateEntity:  
*     entity\_id: str; surface\_forms: list\[str\]; short\_desc: str  
*     provenance: list\[str\]                 \# \["query","first\_pass","llm"\]; linking optional  
*     requested: bool \= False               \# explicitly named in narrative → never suppress  
*     u\_sel: float \= 0.0                    \# estimated entity-selection utility  
*     d\_cov: dict \= None                    \# aspect\_id \-\> new-coverage gain  
*     redundancy: float \= 0.0; drift: float \= 0.0; source\_risk: float \= 0.0  
*     action: str \= ""                      \# promote | neutral | suppress  
*   
* @dataclass  
* class Branch:                              \# a retrieval branch to fuse  
*     kind: str                             \# base | aspect | entity  
*     query: str; weight: float; run: list  \# \[(docid,text,score,rank)\]


  ## **Stage 1 — Plan the information need (one structured call)**

* def plan(title, narrative) \-\> {aspects:\[Aspect\], requested\_entities:\[str\], constraints:{...}}


Prompt (JSON out): decompose into 4–8 **aspects** with importance weights (0–1), extract **explicitly requested entities**, note required comparisons/dates/jurisdictions/populations, and emit **one concise retrieval query per aspect** (short keyword queries, not the full narrative). Cache per topic. *Rationale:* the full long narrative dilutes BM25; the title \+ per-aspect queries retrieve better (matches official baseline guidance).

## **Stage 2 — Aspect retrieval → R\_0 (the baseline)**

* R0 \= fuse(\[ retrieve(title,100) \] \+ \[ retrieve(a.query,50) for a in aspects \])   \# weighted RRF, aspect weights \= w\_a


R\_0 is Run A's evidence and the reference pool for marginal-gain scoring. Record per-aspect coverage of R\_0 (Stage 4 uses it).

## **Stage 3 — Candidate entities**

* E \= dedup( extract\_entities(title+narrative)         \# E\_query (mark requested=True)  
*          ∪ extract\_entities(top\_passages(R0))         \# E\_first-pass (corpus-grounded, long-tail)  
*          ∪ llm\_hypothesize\_entities(title,narrative) ) \# E\_LLM (missing actors/orgs/policies/cases)


REL/WAT linking is **optional** — canonical name \+ aliases suffice to issue ClimbMix queries. Keep linking artifacts if cheap (they let this run feed the paper's entity resource later), but never make linking critical-path.

## **Stage 4 — Û^sel: marginal selection utility (batched, not per-pair)**

For each candidate e, issue **aspect-conditioned** queries `a ⊕ e` (not one global narrative⊕e), get R\_e per aspect:

* Û^sel(e) \= Σ\_a w\_a · ΔCov(a,e)  −  α·Redundancy(e)  −  β·Drift(e)  −  γ·SourceRisk(e)  
    
* **ΔCov(a,e):** aspects newly supported by R\_e beyond R\_0 (batched LLM yes/no: passage answers aspect?). *Marginal* — no credit for re-covering what R\_0 already has.  
* **Redundancy(e):** overlap of R\_e evidence with R\_0.  
* **Drift(e):** fraction of R\_e passages about e but **not** the requested aspect.  
* **SourceRisk(e):** weak/duplicated/promotional/unreliable fraction. Defaults `α=0.3, β=0.4, γ=0.3` (tune on dev). **Batch passage–aspect judgments per topic; never one LLM call per entity–passage pair.**

  ## **Stage 5 — Greedy budgeted branch selection \+ three actions**

Do **not** independently threshold all positive entities (creates redundant branches). Greedy set-cover:

* selected=\[\]; covered=coverage(R0)  
* while len(selected) \< B and marginal\_gain \> ε:  
*     e\* \= argmax\_e  Σ\_a w\_a · (newly-covered aspects by e given \`covered\`)   \# coverage-conditional marginal gain  
*     selected.append(e\*); covered ∪= aspects\_covered\_by(e\*)


Then assign actions:

| Status | Action |
| ----- | ----- |
| Positive marginal utility (selected) | **promote** — retrieval branch, weight ∝ Û^sel |
| Low/uncertain (not selected) | **neutral** — no branch; **do NOT exclude its documents** |
| Demonstrable **drift** or attribution risk (high Drift/SourceRisk) | **suppress** — zero branch weight |
| **Never suppress a `requested=True` entity** (query-named entities stay, even if peripheral). Note: gating acts on **branches**, then documents are reranked globally in Stage 6 — an inert entity's document can still enter via base/aspect/other branches. |  |

## **Stage 6 — Utility-weighted fusion → quotas → global rerank**

* S(d) \= Σ\_branch  ω\_branch / (k\_rrf \+ rank\_branch(d))          \# k\_rrf=60  
*        ω\_base=1 ; ω\_aspect ∝ w\_a ; ω\_entity ∝ Û^sel(e) ; suppressed=0


Then: dedup by docid; **aspect quotas** (guarantee each aspect ≥ q\_min passages so one easy aspect can't dominate); **global rerank** of the pooled top-N against the full narrative (cross-encoder / the shared reranker). Output `N_fused` (≈100) candidate docs.

## **Stage 7 — Generate \+ two-layer verification**

Generate a sentence-level cited answer over the fused pool. Then verify:

1. **General claim support:** does each cited doc fully/partially support the whole sentence?  
2. **Entity-role support:** for each named entity in the sentence, does the citation support (a) its identity, (b) the attributed action/property/role, (c) the claimed relationship to other entities? **Repair** unsupported sentences by narrowing, re-citing, or deleting. Entity-role check *supplements* the general check. Enforce format: `references` \= only docs cited ≥1×; citations \= zero-indexed positions into `references`, sorted by support strength.

   ## **The three-run harness (identical except the gate)**

Hold **aspect decomposition, reranker, generator, context budget, answer prompt, and citation repair identical** across runs; vary only entity-branch logic:

* **Run A — aspect-only:** R\_0, no entity branches.  
* **Run B — ungated expansion:** all eligible entity branches, equal/fixed weight.  
* **Run C — Û^sel-gated:** greedy-selected branches, utility-weighted, suppress on drift. This isolates the one question: *does estimated marginal entity utility beat both aspect-only and naïve expansion?* If official limits allow only 1–2 RAG runs: prioritize **C → A → B**; keep all three internally. If Retrieval-task submission is allowed, also submit the three evidence pipelines' rankings (separates "better evidence" from "better generation").

  ## **Loader**

* def load\_topics(path) \-\> list\[{id,title,narrative}\]   \# accept .tsv AND .jsonl; preserve IDs exactly; expect 119


  ## **Offline evaluation (separate axes — do not aggregate)**

* **Retrieval (RAG25 dev, one fixed UMBRELA qrel set):** nDCG, recall; aspect coverage; **unique gain over R\_0**; drift rate; redundancy.  
* **Answer coverage:** RAGDoll nugget coverage; vital-nugget recall; **aspect-level minimum coverage**.  
* **Grounding:** full/partial/no support; unsupported-sentence rate; **unsupported-entity-attribution rate**; citation precision/recall (RAGDoll support).  
* **Pairwise:** RAGDoll Arena (usefulness) **\+ a local citation-aware pairwise prompt** (shows sentences \+ cited evidence, since Arena omits citations); per-topic paired A/B/C. Keep **RAG25 dev** (TREC-like coverage) and **ResearchRubrics** (complex instruction-following robustness) **separate**, not pooled.

  ## **Config (tune on dev)**

* plan: {min\_aspects: 4, max\_aspects: 8}  
* retrieval: {base\_k: 100, aspect\_k: 50, entity\_k: 50}  
* u\_sel: {alpha: 0.3, beta: 0.4, gamma: 0.3}      \# redundancy, drift, source-risk penalties  
* select: {branch\_budget\_B: 8, greedy\_eps: 0.02}  
* fuse: {k\_rrf: 60, N\_fused: 100, aspect\_quota\_qmin: 5}  
* suppress: {drift\_hi: 0.5, sourcerisk\_hi: 0.5}


  ## **Fast path (if time compresses)**

Aspect-first pipeline stays (it's the baseline and the fallback). Simplify Û^sel to `Σ w_a·ΔCov − β·Drift` (drop redundancy/source-risk), and replace greedy with top-B by Û^sel. Keep branch-level gating \+ never-suppress-requested. Still yields all three runs. Restore full Û^sel only if time allows.

## **Function checklist**

* load\_topics(path) ; plan(title,narrative) ; aspect\_retrieve(...) \-\> R0  
* candidate\_entities(title,narrative,R0) ; u\_sel(e, aspects, R0) \-\> (score, dcov, drift,...)  
* greedy\_select(entities, aspects, B, eps) \-\> selected ; assign\_actions(...)  
* weighted\_fuse(branches) \-\> pool ; aspect\_quota\_and\_rerank(pool, aspects, narrative)  
* generate\_cited(pool) ; verify\_general(...) ; verify\_entity\_role(...) ; repair(...)  
* run\_topic(topic, cfg, mode∈{A,B,C}) \-\> submission\_line


  ## **Sanity checks before submit**

* All 119 topics have exactly one output; every reference cited ≥1×; citation indices valid \+ sorted by support.  
* Runs A/B/C differ on most topics (else gate is a no-op → retune).  
* **No document excluded solely because a low-utility entity retrieved it** (branch-gating invariant).  
* No `requested` entity suppressed.  
* Dev: **C ≥ A** and **C ≥ B** on nugget coverage / citation support (the hypothesis; retune if not).  
* All branches, Û^sel scores, prompts, contexts, model versions, seeds **saved** (for post-submission CEU^sel).  
* LLM calls cached, temperature 0, fixed seed.

  ## **§11 Post-submission (turns the run into paper evidence)**

Because you own the pipeline, after submission run the controlled ablation you dropped from the critical path:

* CEU^sel(e) \= Q(Y\_with entity-branch) − Q(Y\_without entity-branch)     \# your pipeline, both conditions


over saved contexts, ≥2 generators, replicates — legitimate controlled entity-selection utility, and material for Paper 1 / a follow-on.

* 

# Tab 3

# **TREC RAG 2026: Claim-Gated Retrieval-Augmented Generation**

**System name:** Claim-Gated Retrieval-Augmented Generation 

**Deadline:** Submissions due **August 8, 2026** (test topics released July 6). \~4 weeks. 

**Tasks we enter:** **both** — the **R (Retrieval)** task and the **RAG (Generation)** task. They share a retrieval front-end. R is the home of our CGPC reranker (it has a ranking metric); RAG is the home of our claim-gate (coverage/grounding). See §1.5 for the split. 

---

## **0\. The one-paragraph pitch (read this first)**

Most RAG systems retrieve documents and let the generator sort out what's relevant. Our system moves the relevance decision **before** generation: we decompose retrieved passages into claim-bearing evidence units, classify each by claim profile (claim-empty / wrong-aspect / directly-supporting), plan the answer's facets, select only claim-grounded evidence that covers those facets, generate a sentence-cited answer from **only** that evidence, and then verify every citation. The motivation comes from our own reranking experiments (E1): documents that share a query's topic but answer the *wrong aspect* are a real, measurable failure mode, and claim-profile-aware selection repairs them. For TREC RAG we apply the same idea at **inference time** as an evidence gate, targeting the track's actual evaluation — **coverage of important information** and **grounding** — rather than a ranking metric.

**Why this fits the track:** TREC RAG 2026 evaluates answers by pairwise *battles* judged on usefulness, correctness, **coverage**, **grounding**, **citation quality**, and **absence of unsupported/misleading claims**. There is *no* ranking metric to win. The diagnostic is **nugget-based**: did the answer cover the atomic facts a good answer should contain? Our claim-gate \+ facet-coverage pipeline targets exactly this. The 2025 track overview found citation accuracy is nearly solved but **nugget coverage remains the open problem** — so **coverage is where we compete.**

---

## **1\. What the track actually is (facts, from the 2026 guidelines)**

* **Corpus:** NVIDIA **ClimbMix-400b**, accessed through the **Pyserini REST API** (not a local index). Requires a `PYSERINI_API_TOKEN`. Index name is `climbmix-400b`. Document IDs are shard-based, e.g. `shard_00459_61697`.  
* **Topics:** `trec_rag_2026_queries.jsonl`, each with `id`, `title`, and a 2–3 sentence `narrative` (long-form, multi-faceted information need). Preserve these exactly in output.

**Output file:** `rag_output_trec_rag_2026.jsonl`, one JSON object per topic:  
 {  "metadata": {    "team\_id": "iris-mst", "run\_id": "cg-rag-v1", "type": "automatic",    "narrative\_id": "\<topic id\>", "title": "\<topic title\>", "narrative": "\<topic narrative\>",    "prompt": "\<optional\>"  },  "references": \["shard\_00459\_61697", "shard\_01012\_88420"\],  "answer": \[    {"text": "First answer sentence.", "citations": \[0\]},    {"text": "Second sentence.", "citations": \[0, 1\]}  \]}

*   
* **Hard answer rules (these are graded):**  
  * Break the answer into sentences; **each sentence must be grounded** in cited evidence.  
  * `citations` are **0-indexed positions into `references`**.  
  * `references` must contain **only** documents cited by at least one sentence; **every** reference must be cited by at least one sentence.  
  * **Do NOT dump top-k retrieved docs into references.** Only genuinely-supporting docs.  
  * **No unsupported claims**, even if the model "knows" them from pretraining.  
  * Total answer length \< 400 words.  
* **Evaluation:** hidden pairwise **battles** between systems per topic (judge picks better), plus **nugget** diagnostics (strict vital score \+ sub-narrative coverage). We never produce battle output ourselves — we just produce good answers.  
* **Agent-first track.** Read the official guidelines via the SKILL, not blog summaries: `https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/trec-rag-2026-track-guidelines`

## **1.5 The two tasks — and why we do both**

TREC RAG 2026 has **two tasks; you may enter one or both.** They map cleanly onto our two assets, and they **share the retrieval front-end**, so doing both is far less than double the work.

| task | what you submit | metric | our asset |
| ----- | ----- | ----- | ----- |
| **R (Retrieval)** | ranked top-100 segments per topic, TREC run file `r_output_trec_rag_2026.tsv` | **ranking (nDCG etc.)** against UMBRELA/NIST qrels | **CGPC reranker** — R *is* a reranking task, so this is where our trained reranker is measured on its actual metric |
| **RAG (Generation)** | sentence-cited answer per topic, `rag_output_trec_rag_2026.jsonl` | pairwise **battles** on coverage \+ grounding (no ranking metric) | **claim-gate pipeline** — evidence discipline for answers |

**Why both, and how they share work:**

* Both start with the **same retrieval** (Pyserini REST over ClimbMix). The R run file *is* the ranked pool; the RAG pipeline *consumes* that same pool.  
* Our **CGPC reranker**, if trained, reranks the pool for R (submitted directly as the run) AND can optionally clean the pool that feeds RAG. One model, two uses.  
* So the pipeline is: **retrieve → \[CGPC rerank\] → {R: submit run} and {RAG: claim-gate → answer}.**

**The strategic split (risk vs. research value):**

* **RAG \= the safe, guaranteed deliverable.** The claim-gate is training-free, targets the track's open problem (coverage), and does not depend on standing up ClimbMix training infra. **This must ship no matter what.**  
* **R \= the CGPC showcase and our method paper's natural experiment.** Because R has a *ranking metric*, it is the correct place to show CGPC beats a baseline reranker — on a large, in-distribution corpus (train on ClimbMix dev UMBRELA qrels, deploy on ClimbMix test). This is higher research value (it feeds the WSDM paper with a big-corpus result) but depends on the ClimbMix grounding \+ training infra landing in weeks 1–2.

**Decision rule (revisit end of week 2):**

* Week-1 foundation \+ RAG pipeline on track → **also train CGPC and enter R.**  
* Foundation slipping → **drop R, ship RAG only.** R is the stretch goal; RAG is the commitment.

**Feasibility of CGPC on ClimbMix (why R is even possible in 4 weeks):** the dev data ships **UMBRELA qrels graded 0–3**, which is exactly what CGPC positive-selection needs. Our existing CGPC pipeline (audit → split → build typed negatives → train → score) runs unchanged with `--dataset-id` pointed at ClimbMix dev \+ `--pos-min-grade` set for the UMBRELA scale. The only new work is grounding the ClimbMix dev pools (same judge step as our other datasets). The training machinery already exists and is validated.

## **2\. What we have that others don't**

* A **validated claim-grounding methodology** (our GPT-4.1 answerability/evidence judge, κ≈0.77 on prior work) and a **claim-profile taxonomy** (C0 claim-empty / C1 wrong-aspect / C2 entity-aspect-confusable) that is exactly an evidence-quality filter.  
* **E1 evidence** that wrong-aspect (C1) confusables are a real reranker failure mode and that claim-profile selection repairs them (net 3× a count-matched baseline, verified by inspection). This is our **motivation** — we cite it as the reason claim-gating helps.  
* A working grounding \+ construction codebase we can repurpose for the gate.

## **3\. The system architecture (what we build)**

Shared front-end, then the two tasks branch:

ClimbMix topic narrative  
        │  
        ▼  
\[S1\] Retrieve broadly     ── Pyserini REST (BM25; \+ dense/hybrid if available), merge by RRF, top \~100  
        │  
        ▼  
\[S1b\] CGPC rerank (optional / R-task)  ── trained CGPC reranker reorders the pool  
        │  
        ├────────────────────────────► \[R TASK\] submit reranked top-100 as r\_output\_trec\_rag\_2026.tsv  
        │  
        ▼  
\[S2\] Claim-gate candidates ── per passage: extract answer-relevant claim(s), classify C0/C1/C2  
        │                      C0 \= drop; C1 \= keep only for caveats/background; C2 \= primary evidence  
        ▼  
\[S3\] Facet / sub-narrative plan ── decompose narrative into 4–8 facets (mirrors nugget coverage)  
        │  
        ▼  
\[S4\] Evidence selection    ── assign claims→facets; select by utility; \~8–15 claims, 1–3 per facet  
        │  
        ▼  
\[S5\] Cited generation      ── generate sentence answer from ONLY selected claims; each sentence cites  
        │  
        ▼  
\[S6\] Citation verifier     ── for each sentence, check cited evidence supports it; drop/weaken/fix  
        │  
        ▼  
\[RAG TASK\] rag\_output\_trec\_rag\_2026.jsonl (references \= only cited docs)

**S1 feeds both tasks. S1b (CGPC rerank) is the R-task deliverable and an optional RAG pool-cleaner. S2–S6 are the RAG-task deliverable.**

**Core principle:** the generator only ever sees the **selected, claim-grounded evidence** — never the raw retrieved passages. This is what prevents wrong-aspect and unsupported material from entering the answer.

### **Stage-by-stage spec**

**S1 — Retrieve broadly (`retrieve_topics.py`)**

* Call the Pyserini REST API for each topic narrative. Get BM25 top-100. If a dense/hybrid baseline endpoint is available, get that too and merge by **Reciprocal Rank Fusion (RRF)**.  
* Goal is **recall**, not precision — do not over-optimize retrieval. The gate does precision.  
* Cache raw results to disk (`data/retrieval/<topic_id>.json`) so we never re-hit the API.  
* Output: candidate pool of \~100 (docid, text, rank) per topic.

**S2 — Claim-gate (`claim_gate_evidence.py`)** — the heart of the system.

For each candidate passage `P` and topic narrative `T`, one LLM-judge call:  
 Given the information need T and passage P:1. List the answer-relevant factual claim(s) P makes about T (0 if none). Quote/point to text.2. For each claim, which facet/sub-question of T does it address?3. Classify P:   C0 \= topical but no useful claim (claim-empty)   C1 \= partially useful / wrong-aspect / insufficient   C2 \= directly supporting answer evidence4. Return: {claims:\[{text, facet\_hint, support\_span}\], profile: C0|C1|C2, citation\_docid}Return concise JSON only.

*   
* **C0 → drop. C1 → keep only if a facet has no C2 (caveat/background). C2 → primary evidence.**  
* This is the same claim-profile decomposition from our reranking work, applied at selection time.  
* **Cost control:** this is one judge call per (topic, passage). Budget it (see §6). Gate only the top \~30–40 candidates per topic, not all 100, unless cost allows more.  
* Use an **open model** for the judge if possible (Qwen2.5-32B / Gemma-3-27B local, or GPT-4.1 via API if budget allows). Few-shot the prompt — do not zero-shot.

**S3 — Facet plan (`plan_facets.py`)**

* From the narrative alone (before or alongside gating), generate **4–8 facets** \= the sub-questions a complete answer must address. This mirrors the track's **sub-narrative coverage** metric and its **nugget** structure.  
* **Validate against dev nuggets:** the RAG25 dev data ships gold nuggets. Check whether our generated facets align with the gold nuggets on dev topics — this is our closed-loop signal that facet-planning works BEFORE we touch test topics.

**S4 — Evidence selection (`select_claims_by_facet.py`)**

Assign each gated claim to a facet. Select evidence to maximize **facet coverage** while gating out wrong-aspect. Simple utility score per claim:  
 U(claim) \= grounding\_score         \+ facet\_coverage\_gain      (does it cover an as-yet-uncovered facet?)         \+ novelty                  (not redundant with already-selected claims)         \- redundancy         \- wrong\_aspect\_penalty     (C1 penalized vs C2)         \- citation\_risk            (low support-span confidence)

*   
* Keep **\~8–15 claims total, 1–3 per major facet.** Every facet should have ≥1 claim if evidence exists; note uncovered facets (the answer can say evidence is insufficient).  
* Start with a **greedy** selector (cover each facet with best-utility claim, then fill). Do not build a learned utility model in v1.

**S5 — Cited generation (`generate_cited_answer.py`)**

Prompt the generator with **only the selected claims** (with their docids), the narrative, and the facet plan. Hard instructions:  
 Write a concise (\<400 word) answer to the narrative using ONLY the provided evidence claims.- Output one sentence per line.- Every factual sentence MUST cite the evidence id(s) it is grounded in.- Do NOT introduce facts not present in the evidence.- If a facet has no evidence, state that the evidence is insufficient for it.

*   
* Map generated citations to `references` (0-indexed). Ensure every reference is cited.

**S6 — Citation verifier (`verify_citations.py`)** — highest-value safety net.

* For each answer sentence, an LLM/NLI check: does the cited evidence actually support it?  
  * Supported → keep.  
  * Unsupported → delete the sentence, or weaken it, or attach the correct citation.  
  * Missing citation on a factual sentence → add it or remove the sentence.  
* Rebuild `references` to include only docs still cited after verification.  
* This directly targets the battle criterion "unsupported or misleading claims" and the nugget "not\_support" penalty.

## **4\. The measurement loop (build this FIRST — week 1\)**

The single most important de-risking fact: **we can score ourselves on dev topics before submitting.** The dev data ships gold nuggets and UMBRELA qrels, and **RAGDoll** runs the full eval. So:

* Get **ClimbMix retrieval** working via Pyserini REST on dev topics.  
* Get **RAGDoll** (`https://github.com/castorini/RAGDoll`) running end-to-end on ONE dev topic with a trivial baseline answer (e.g. retrieve top-5, concatenate as the "answer").  
* Once RAGDoll produces a nugget-coverage score for a baseline, we have a **closed-loop dev signal**: every change to the pipeline can be measured on dev nugget coverage before we ever run test topics.

**Do not build the full system before the measurement loop works.** Without it we're flying blind.

## **5\. Four-week schedule**

**Week 1 (foundation \+ measurement loop)** — *the make-or-break week.*

* Register for TREC; obtain `PYSERINI_API_TOKEN`.  
* Pull dev data: RAG25 topics, UMBRELA qrels, **RAG25 nuggets**, ResearchRubrics. (`github.com/TREC-RAG/trec-rag-data/tree/main/trec-rag-2026/development-data`)  
* `retrieve_topics.py`: ClimbMix retrieval via Pyserini REST, cache results.   
* Stand up **RAGDoll** on dev; score a trivial baseline answer to confirm the eval loop.   
* **Exit criterion:** we can retrieve from ClimbMix AND get a RAGDoll nugget-coverage number on a dev topic. If this isn't done by end of week 1, everything slips — escalate early.

**Week 2 (the gate \+ facets || CGPC data prep for R)**

* *RAG track:* `claim_gate_evidence.py` (S2), C0/C1/C2 judge (open model, few-shot). (Debayan)  
* *RAG track:* `plan_facets.py` (S3) \+ validate facets against dev gold nuggets.   
* *RAG track:* `select_claims_by_facet.py` (S4), greedy utility selector.   
* *R track (if foundation solid):* ground ClimbMix dev BM25 pools; run the CGPC audit → split → build\_training\_pairs on ClimbMix dev with UMBRELA qrels (`--pos-min-grade` set to the UMBRELA positive threshold). Reuses the existing CGPC codebase.   
* First RAG end-to-end run on \~10 dev topics; measure dev nugget coverage vs the trivial baseline.  
* **Exit criterion (RAG):** claim-gated selection beats the naive top-k baseline on dev nugget coverage.  
* **DECISION POINT:** is the R track viable in the remaining time? If ClimbMix grounding \+ CGPC data prep is working, commit to R. If not, **drop R, focus everything on RAG.**

**Week 3 (generation \+ verification \+ hardening || CGPC train+score for R)**

* *RAG track:* `generate_cited_answer.py` (S5) with strict grounding prompt.   
* *RAG track:* `verify_citations.py` (S6).  
* *RAG track:* full pipeline on the full dev set; iterate facet count, claims-per-facet, utility weights, using dev nugget coverage \+ citation-support as the signal.  
* *R track (if committed):* train the CGPC reranker on ClimbMix dev; score held-out dev; confirm it beats a baseline reranker on nDCG (this doubles as a WSDM-paper big-corpus result).  
* **Output validators (BOTH):** RAG — enforce every hard answer rule from §1 (0-indexed citations, refs all cited, \<400 words, valid JSONL). R — valid 6-column TREC run, top-100, scores non-increasing. A malformed submission scores zero — these validators are not optional.  
* **Exit criterion:** a valid verified RAG run on full dev above baseline; (if R committed) a valid CGPC-reranked R run scored on dev.

**Week 4 (test runs \+ submission)**

* Run the frozen RAG pipeline on the **test topics** (released July 6\) → `rag_output_trec_rag_2026.jsonl`.  
* *If R committed:* run retrieval \+ CGPC rerank on test → `r_output_trec_rag_2026.tsv`.  
* Validate output formats (RAGDoll for RAG; TREC-run check for R). Produce 1–2 runs per task plus an ablation (e.g. `no-gate` / `no-verify` for RAG; `bm25-baseline` vs `cgpc-rerank` for R) for the notebook paper.  
* Submit by **Aug 8** (both task files if R committed, RAG file otherwise).  
* Start the notebook paper (draft from this plan \+ dev results).  
* **Buffer:** finish test runs by **Aug 5**, leaving 3 days for format/submission issues.

## **6\. Cost budgeting (do this in week 1, before committing)**

The gate (S2) and verifier (S6) are LLM calls at scale. Estimate before building:

* gate calls ≈ (topics) × (candidates gated per topic, \~30–40)  
* verify calls ≈ (topics) × (answer sentences, \~8–12)  
* Multiply by dev-set size for iteration, plus one test run.  
* **If using an API judge (GPT-4.1):** compute $ and rate limits; may force a smaller gated pool.  
* **If using an open model (Qwen2.5-32B / Gemma-3-27B / MiniCheck local):** free at scale, needs a GPU and a quick validation that its C0/C1/C2 judgments are sane on a few dev topics.  
* **Recommendation:** open model for the gate (cheap, scalable, reproducible — a plus for the notebook paper), API only if the open judge underperforms on a dev spot-check.

## **7\. Scope discipline — what NOT to do (v1)**

Under a 4-week deadline, do **not**:

* Treat the CGPC reranker as a **RAG-task requirement**. For RAG, it's optional (RAG has no ranking metric). The reranker's home is the **R task**, which *is* a ranking task — train it only if you commit to R at the week-2 decision point. **Never let R-track work block the RAG submission**, which is the guaranteed deliverable.  
* Build a learned utility model, entity/aspect C2 operationalization, or a complex agent loop in v1.  
* Optimize retrieval heavily. Recall-oriented BM25 \+ RRF is enough; the gate does the precision.  
* Skip the output validator or the measurement loop to "save time" — those are the two things that actually prevent a zero-score submission.

## **8\. The notebook paper angle (write alongside)**

The notebook paper can cover both tasks: claim-gated generation (RAG) and, if entered, CGPC reranking (R). The unifying thesis is the same — claim-profile structure improves evidence quality, whether for ranking (R) or answer construction (RAG).

> We adapt claim-grounded supervision ideas to a training-free RAG pipeline. Rather than treating retrieved passages as atomic evidence, we decompose them into answer-bearing claims, classify each by claim profile (claim-empty / wrong-aspect / directly-supporting), plan answer facets, and generate only from claim-grounded evidence that covers those facets — explicitly gating out wrong-aspect and claim-empty passages before generation. Our reranking experiments show wrong-aspect claim-confusable documents are a real failure mode and that claim-profile selection repairs them; for TREC RAG we apply the same diagnosis as an inference-time evidence gate. We target coverage and grounding — the metrics the track evaluates — rather than ranking.

Include ablations on dev: **RAG** — no-gate vs full, no-verify vs full (show the gate and verifier each help coverage/grounding); **R** (if entered) — baseline reranker vs CGPC reranker on nDCG. The R result also feeds the WSDM method paper as a large-corpus (ClimbMix) data point beyond TREC-COVID/SciFact. That's the paper's empirical spine.

## **9\. First actions this week (assign now)**

1. **Shubham:** register for TREC, get `PYSERINI_API_TOKEN`, pull dev data, read the official RAG task \+ development-data guidelines in the SKILL repo (not blog summaries).  
2. **Debayan:** `retrieve_topics.py` — ClimbMix retrieval via Pyserini REST, cache to disk. Confirm you can retrieve top-100 for a dev topic and read the doc text.  
3. **Utshab:** stand up **RAGDoll**; score a trivial baseline answer on one dev topic to get a nugget-coverage number. This is the measurement loop.

When 1–3 are done, we have retrieval \+ a scoring loop, and week 2 (the gate) can start against a real dev signal. Nothing downstream begins until the week-1 exit criterion is met.

---

*Key references:*

* Track guidelines (READ THESE): https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/trec-rag-2026-track-guidelines  
* RAG task spec: `.../references/rag-task.md`  
* Dev data: https://github.com/TREC-RAG/trec-rag-data/tree/main/trec-rag-2026/development-data  
* RAGDoll eval toolkit: https://github.com/castorini/RAGDoll  
* Pyserini REST API: https://github.com/castorini/pyserini/blob/master/docs/usage-rest.md  
* ClimbMix API usage example: https://github.com/castorini/NanoKnow

