# Release implementation progress

Plan: docs/human-html/2026-11-07-plan-reddit-stash-insights-next-release.html
Base: 78e00b8705218aefc72b7f700f7eef47d5eb0c6a
Branch: feat/release-0.1.0
Authorization: user explicitly authorizes implementation and publication.

## Scope and interfaces

| Tasks | Shared boundary | Resolution |
|---|---|---|
| Safety / chat / UI | provider creation | Safety first; all callers use explicit cloud opt-in. |
| Eval / chunks | document identities | Stable source-thread IDs and evidence IDs; compare identical grouping. |
| Chunks / CLI | indexing records | Preserve legacy result fields; version chunk schema. |
| Chat / providers | token limits | Add provider counters and output reserve; pack only cited evidence. |
| UI / packaging | installed module | Test wheel outside checkout. |
| Release / CI | publication | Full checks and review before PR, merge, tag and public install. |

All six task descriptions were checked for internal conflicts.
Ruling: author and independently review labels against source evidence; record agent review honestly, never claim human review.
Ruling: use cached local models or keyword evaluation; never transfer private archive for hosted embedding by implication.
Ruling: no new model default; compare development results before opening held-out results.
Ruling: implementation tasks run sequentially; root prepares read-only evidence while each worker runs.

## Status

Task 1: complete, bc3a927. Independent spec and quality review PASS. Core/search/service checks pass; cached local model tests pass.
Historical checkpoint: Task2 started; superseded by the final validation checkpoint below.
Private corpus frozen at /private/tmp/rsi-private-eval/corpus; 2353 items; baseline keyword index built from 78e00b8 with 32000-char cap.
GitHub Discussions enabled and public description corrected.
Project .venv now includes search/build/publication tools. Historical missing-attribute compatibility shim was simplified to keep typing valid with installed transformers.
Public PyPI release pending; Discussions and GitHub access verified.

Private labels: 50 answerable +10 subject-scoped absent controls, independently source-reviewed by a separate agent. Fourteen held-out cases retain valid after-32k evidence. No human label review claimed.
Public synthetic fixture independently reviewed: all50answers,2500qrels,10absent controls and60sourcefiles pass. Functional fixture does not prove general search quality.
Development keyword baseline independently calculated: recall@10=0.9,MRR@10=0.8625,nDCG@10=0.8715338279036696. Candidate not yet built.
PyPI token credentials absent from environment/config. Brave sign-in is now verified; see final checkpoint.

Ruling: the full dense baseline runs concurrently from a frozen 78e00b8 source archive. Current branch edits cannot change its pipeline. Do not open held-out results until candidate settings freeze.
Public synthetic fixture is independently agent-reviewed; retain explicit functional-only scope, not a broad quality claim.
PyPI Brave session verified signed in as rhnfzl; pending publisher form prepared for repo/workflow pypi-publish.yml and pypi environment. Form has NOT been submitted.
GitHub pypi environment created with custom deployment policy allowing only v* tags.

Task 2: code committed070b90f, independent review PASS, installed wheel fixture outside checkout PASS. Three private development baselines completed, all zero errors: keyword(0.9,.8625,.8715338279036697),semantic(1,.8625,.897719778617961),hybrid(1,.9,.9246141434859123) forrecall/MRR/nDCG@10.
Task 3 initial code committed e66196e with reviewed UI and release scaffolding.
Ruling from measured full archive: per-comment chunking creates279128passages. Initial candidate run47037/PID81835 was deliberately interrupted after confirming target; partial test index retained, no live change. Pack adjacent short comments before restarting a NEW candidate index. Preserve every evidence ID/permalink/parent and exact source offsets.
Task4 code and review complete. Task 5: UI3tests and liveBraverender PASS; GIF needs finalinstall/chat/source frames. Task6: publication/scaffold prepared but not uploaded.
Private labels/runtime/reports at /private/tmp/rsi-private-eval. Baseline CPU modelbuild890.725s/RSS4956913664bytes; candidate uses local GPU outside sandbox (MPS unavailable inside). No indexing speed claims across devices.
Active helper: local UI execsession45471 listens8517. No hosted archive calls. PyPI Brave form is prepared, not submitted.

## Final validation checkpoint

Tasks 1-4 implemented and independently reviewed. Chat uses exact local GGUF templates and token budgets; packed citations list every included evidence reference. Real local Qwen3B answer/source smoke passes. No hosted archive calls.
Final eval adapter uses raw candidate rows, avoiding duplicate overfetch and thread grouping. Previous development reports are superseded; final reports use matching MPS/FP16 baseline and candidate settings.
Chunk algorithm3 fixes hidden embedder512-token truncation, packs adjacent short comments, bounds tokenizer probes, and preserves source offsets. Two final span-order regressions fixed and independently reviewed: whitespace paragraph overlap and nonmonotonic tokenizer packing.57 scoped tests and348 seeded tokenizer cases pass.
Full current suite259 tests passes with10 expected skips; Ruff and mypy source+changed tests pass. Real cached BGE tests passed earlier; final artifact validation still pending.
Full-corpus final counter/hash run exec77723 active. Final baseline development scoring exec84488 active. Final candidate fresh directory candidate-final-bge-index is not yet built. Final private config uses cached BGE-M3, FP16, batch512,512-token passages,64-token overlap.
GIF worker owns only docs/assets/rsi-demo.gif, preparing real installed-wheel install/search/local-chat/source frames using synthetic data.
PyPI sign-in remains verified in Brave; pending publisher form has not been submitted. Browser policy requires action-time confirmation for the new publishing authority. Ask only after release artifacts and checks are concrete.
No push, merge, tag or PyPI upload yet. Main merge triggers existing GHCR image publication; disclose before merge. No manual homelab deployment is authorized by implication.
Final distribution validation: /private/tmp/rsi-final-dist wheel and sdist pass strict Twine checks. Wheel100members/99611bytes, sdist134members/207114bytes. All60publicsourcefiles and installed UI entrypoint included; private corpus/labels, AGENTS, agent notes and human HTML excluded. Audit initially used the wrong expected UI filename(app.py); corrected to actual launcher targetchat.py after archive inspection.
Fresh core-only and service-extra environments install the final wheel outside checkout. Core version/scan passes without LanceDB/Torch; service fixture50answerable+10absent completes0errors, UI/MCPhelp works, Torch remains absent. Final wheel hash9a15475d7058494079427b463e8620aa5fd1e9d19793e9272fa0693007cfcc6c matches the independently tested local-chat demo wheel exactly.
Final baseline development runs all3modes complete0errors with the raw-candidate adapter. Scores remain keyword(.9,.8625,.8715338279036697),semantic(1,.8625,.897719778617961),hybrid(1,.9,.9246141434859123).
Candidate GPU build exec81824 active at fresh candidate-final-bge-index. Fullcounterexec77723 active. Final build marker requires subsequent fullpayloadhash verification before evaluation. GIF finalized4frames/27seconds/83621bytes, independent final auditPASS; case-study test-count wording corrected to249passed+10skipped.
Full algorithm3 tokenizer/hash check complete:84571passages,2353items,0parseerrors,max512tokens,705.9009seconds. Every nonempty source body has full coverage and strictly advancing ends. Exact expected passage hashes saved privately for stored-index verification.
GitHub PR4 opened draft:https://github.com/rhnfzl/reddit-stash-insights/pull/4 atb8e62ba. Packagebuildpasses. Core/service CI found a test-only colored-output assertion failure: ANSI styling splits--allow-remote. Reproduced withTERM=xterm-256color FORCE_COLOR=1; fix usesClickunstyle while retaining exact option/destination assertions, explicitly requestscoloredoutput.259tests(249pass10skip),Ruff,mypypassagain. Productionwheelunchanged; rebuildsdistaftertestfix.
Installed services-wheel UI renders inBrave atlocalhost8518; serverexec80557active. HTMLpreviewserverexec26950at8519active; earlierUIexec45471at8517alsoactive. Root must stop these self-owned previews when done.
