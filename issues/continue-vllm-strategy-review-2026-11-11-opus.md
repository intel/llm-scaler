# Should the vLLM-on-Lunar-Lake work continue? A whole-picture review (2026-11-11)

*A few terms used below. **vLLM** and **llama.cpp** are the two programs that run a language model on the console. **tok/s** (tokens per second) is roughly words per second of reply. **Upstream** means the official vLLM project; a **PR** (pull request) is a fix offered to it. The **Sparks** are the three NVIDIA DGX Spark machines at home. **Long-prompt processing** is how quickly the engine reads a large input, such as a coding agent sending thousands of lines, before the first word of the reply appears. "est." marks an estimate. No hours were ever logged.*

---

## 1. The short answer

No: not as your main project, and not the way it has run since May. Freeze the vLLM stack properly so nothing is lost. Put the coding agent and both bots on llama.cpp now, and send heavy or long-prompt jobs to the Sparks. Replace the "should I upgrade?" question with a short written list of trigger events and a review date in February 2027. **The one condition that flips this:** if you would still want to open the vLLM fork next weekend even with working assistants already running, the engineering is the reward in itself. In that case keep vLLM as a declared hobby with a fixed budget, and judge each session by a finished public result (option C below).

---

## 2. What the last eight months show

| Phase | What the time went into | What it bought |
|---|---|---|
| Mid-March (days) | llama.cpp: one install script, one launcher | A working server at ~13 tok/s, essentially no upkeep since, and a public setup guide |
| March–April (~60 active days, est. 180–250 h) | The vLLM port: ~190 fork commits plus ~57 handheld-repo commits, kernels, patches, benchmarks, PR #40295 | The 20 tok/s target was met (18–30 tok/s), plus real know-how. No service. |
| May–August (est. 50–150 h) | Keeping the stack alive through five releases. Fixes stayed inside a Python environment, outside git. | The stack still runs. No commits, and little that can be reused. |
| August–November (est. 40–100 h of your time, plus agent time) | About ten "should I upgrade?" checks, three status reports, studies, a 70-idea brainstorm, plans v1 and v2 | Better understanding, but no decision and no service. Plan v2 now lists corrections owed to ten earlier documents. |

**Building versus using.** Roughly est. 270–500 hours went into building, maintaining and analysing the vLLM stack. Close to zero hours went into using it for the four jobs. None of them runs on vLLM as a finished service, and the bots run on a cloud model through a gateway, with local llama.cpp as the fallback.

**The patterns that matter:**

1. **The goal was reached in April, and the finish line kept moving after that.** vLLM releases about every two weeks. Plan v2 needs four weeks of evenings before it expects any new gain, and another cutover (torch 2.15) has already started. A plan longer than the release cycle cannot catch up, so the measurements go stale before they can be used.
2. **The output shifted from code to documents, and the last mile was skipped.** PR #40295 was never rebased. The Gemma 4 PR and the upstream issue were never filed, and several drafts are marked "not posted yet". This is a pattern of how the work was organised, not a personal failing. A coding agent makes documents cheap, and a document feels like progress.
3. **The analysis only ever looked at vLLM.** Re-measuring today's llama.cpp on this console would take about one evening. It has never been done, and plan v2 explicitly rejected it. One warning cuts the other way too: llama.cpp has worked since March, yet it never became your daily default either. The engine may not be the only thing in the way.

---

## 3. vLLM versus llama.cpp as a way of working

| What shapes your weeks | vLLM on this console | llama.cpp on this console |
|---|---|---|
| Time to a working setup | ~2 months to reach the target, and still not a service after 8 | Days |
| Upkeep per month | Recurring: re-checks at every upstream generation (est. 10–20 h each, every 1–3 months), plus the habit of upgrade checks | Est. 1–3 h, going by eight months of near-zero maintenance |
| What breaks, and how often | "Backends mismatch" errors, a contaminated environment, memory leaks after crashes that can need a reboot, slow 36–40 s loads. It recurs with releases. | Nothing reported since March |
| How results arrive | Big early gains, then measurements that expire with each version | Once, then they stay |
| Speed where it matters | Bots: faster, though the bots don't need it. Coding agent: several times faster on long prompts, its one clear lead. | Bots: 9.5–17 tok/s measured here in March, ~25 tok/s on other Lunar Lake machines (not measured here). Long prompts are slower. |
| Heavy jobs | Not needed: the Sparks run much larger models at 26–30 tok/s | Same |
| Dependence on upstream | High: local patches, pinned versions, 1.5–2 h builds | Low: one program, no local patches |
| How alone you are | Completely. Nobody else is known to run vLLM on this chip, and upstream tests only on discrete Arc cards. | Widely used, though Intel support on this chip has had reported problems (unverified) |
| What is learned | Deep, rare knowledge of the stack. It fades as upstream moves. | Little new. It is a tool, not a project. |
| Memory left for other things | 1.5–3 GiB less, and memory is held even when idle | More, and it unloads when idle |

---

## 4. The options

**A. Continue plan v2 as written.**
- *Week to week:* most evenings go to measurement (noise floors, memory checks, speculative-decoding trials), then the torch-2.15 cutover. Services wait.
- *The bet:* you enjoy this measurement work, or the coding agent truly needs vLLM's long-prompt speed.
- *Hours:* est. 50–80 h in month 1, then 10–20 h per upstream generation.
- *Best case:* by early 2027, a near-stock vLLM coding agent in daily use.
- *Likely failure:* the plan becomes the product again.
- *Warning sign:* asking "should I upgrade?" in week 1 or 2.
- *Suits:* someone who loves the measuring itself.

**B. Settle it in one sprint, then a capped coding lane on frozen vLLM.**
- *Week to week:* bots move to llama.cpp now. Run 2–3 evenings of fair side-by-side tests. If vLLM wins clearly, the coding agent alone stays on a frozen v0.33, at about two evenings a month.
- *The bet:* you often code on the console with long prompts, away from the Sparks.
- *Hours:* est. 10–15 h for the sprint, then about 10–13 h/month.
- *Best case:* a fast local coding agent with little upkeep.
- *Likely failure:* the sprint swells into plan v2, or the lane gets little use.
- *Warning sign:* "one more baseline" on day 3.
- *Suits:* a heavy offline coding-agent user.

**C. Engineering by choice: a declared, finite hobby.**
- *Week to week:* all services on llama.cpp. vLLM gets a stated budget and one 90-day goal (rebase or close #40295, file the kernel PR, publish a guide pinned to one version). Every session ends in a commit, PR or post.
- *The bet:* the engineering is the reward, and you will do the finishing work.
- *Hours:* a number you state up front (est. 8–30 h/month; zero months allowed).
- *Best case:* four to six finished public results by May 2027.
- *Likely failure:* it slides back into version-chasing.
- *Warning sign:* the first session ends in a report.
- *Suits:* the engineering-first owner.

**D. Pause (recommended).**
- *Week to week:* one evening to freeze vLLM properly (patches into git, container exported, a one-page "how to bring it back" note). Then services on llama.cpp, heavy jobs to the Sparks, and a quarterly trigger checklist.
- *The bet:* daily use matters more than extra speed, and llama.cpp is good enough.
- *Hours:* est. 15–25 h in month 1, then 2–6 h/month.
- *Best case:* working assistants within weeks, with vLLM resumable.
- *Likely failure:* the freed time goes nowhere, and the bots stay cloud-first.
- *Warning sign:* an empty usage log at week 4.
- *Suits:* the assistant-first owner, or anyone unsure.

**E. Sparks host the services; the console is a client.**
- *Week to week:* bots and the coding model run on a Spark (stock vLLM on NVIDIA). The console keeps llama.cpp for offline use.
- *The bet:* the Sparks are reliably reachable, and running them all day is acceptable.
- *Hours:* est. 10–15 h of setup, then 2–5 h/month.
- *Best case:* stronger models and a free handheld.
- *Likely failure:* the same treadmill, now on the Sparks.
- *Warning sign:* new documents but no routing in use by week 2.
- *Suits:* someone usually online.

**F. Stop and close out with a public write-up.**
- *Week to week:* commit the patches, export the container, publish "vLLM on a Lunar Lake iGPU: what works, what it costs", close #40295.
- *The bet:* the "never finish" feeling is lasting.
- *Hours:* est. 8–20 h once, then 1–4 h/month.
- *Best case:* a clean end plus a finished public record.
- *Likely failure:* relapse into upgrade checks.
- *Warning sign:* starting the old container "just to compare".
- *Suits:* the assistant-first owner who wants closure.

**Judges' scores (0–10)**

| Option | Wants a working assistant | Engineering-first | Record only |
|---|---|---|---|
| A. Plan v2 as written | 1.5 | 6 | 2 |
| B. Sprint + capped lane | 4.5 | 4 | 5 |
| C. Declared hobby | 3 | **8.5** | 3.5 |
| D. Pause | **8.5** | 3 | **8** |
| E. Sparks host services | 7 | 2 | 6 |
| F. Stop + write-up | 7.5 | 2.5 | 7.5 |

---

## 5. The recommendation in detail

**Next 30 days (option D, with E's routing for heavy jobs):**
- **By 18 Nov (week 1), first evening, nothing else:** commit every live patch to git, including the ones that exist only in the Python environment. Export the vLLM container to a Spark disk and write down its launch commands. Write a one-page "how to bring it back" note with the old numbers clearly dated. Post a short status comment on #40295.
- **Week 1–2:** rebuild current llama.cpp. Measure it once here: reply speed, and the wait for the first word on one real long coding-agent turn. Switch one bot to local-first and keep a two-line daily log (did I use it, did it feel slow, did I switch back to the cloud?).
- **Week 3:** move the second bot. Point the coding agent at llama.cpp over SSH, with large tasks going to a Spark.
- **Week 4:** a week of real use, logging what felt slow and every urge to check vLLM.

**Stop doing:** "should I upgrade?" checks, new status reports and plans, plan v2's measurement programme, and the torch-2.15 cutover.

**Keep, so nothing is lost:** the git history, the committed patches, the exported container, the resume note, the replay tooling, and the recorded agent turns. All of these also work against llama.cpp.

**Stop and resume rules:**
- **If the freeze is not done by 18 Nov:** go to F. A "resumable" state that was never saved is not real.
- **9 Dec (week 4) check:** if neither bot has moved to local-first and the local coding agent was not used on most days, the engine was not the bottleneck. Close vLLM out (F) instead of keeping a paused stack alive.
- **Resume vLLM (run B's sprint once) only if one of these fires:** an integrated-GPU branch appears in vLLM's Intel code; Lunar Lake is added to upstream's build or test lists; a stable torch-2.15 release ships with the needed fix; a second Lunar Lake vLLM user reports upstream; or your log shows long-prompt waits blocking real coding at least weekly with no Spark in reach.
- **Review on 15 Feb 2027** (suggested date): a checklist of under an hour. If no trigger has fired, move to F.
- **If the week-4 log shows you missed the engine work rather than felt relief,** switch to C.

---

## 6. Five questions only you can answer

1. **If a coding agent and both bots worked well on llama.cpp tomorrow with no effort from you, would you still want to open the vLLM fork next weekend?** *If yes,* the engineering is the reward: choose C and give it a budget. *If no,* choose D, and F at the review date.
2. **In a normal week, do you code on the console with long prompts, away from the Sparks and the cloud, and have you actually felt the wait?** *If yes,* run B's sprint once the freeze is done. *If no,* vLLM's one remaining lead serves no job you do. Stay with D or F.
3. **Are the Sparks reliably reachable when you are away, and is running them all day acceptable for power, noise and heat?** *If yes,* make E the main design, with the console as client and offline fallback. *If no,* keep llama.cpp on the console as the main host (D).
4. **Would you keep a version freeze and a stop rule written in advance, given that the plans since August kept growing?** *If yes,* B or C are workable. *If no,* choose D with the quarterly checklist as the only outlet, or F.
5. **Do you care about getting work merged upstream enough to do the rebases and the replies to maintainers?** *If yes,* make #40295 the first goal under C: rebase it or close it within 8 weeks. *If no,* close it with a short note and choose D or F.

---

## 7. Where the strategists and judges disagreed

- **Pause or continue with limits.** Five of the nine strategists leaned toward a pause, and four toward continuing within hard limits. None backed plan v2 as written. The "with limits" camp still put the services on llama.cpp first, so the real split is only about what happens to the vLLM evenings.
- **Measure first, or switch first.** The devil's-advocate and real-options readings say one fair evening of comparison settles the matter. The record-only judge warns that "quick comparisons" have turned into multi-week plans before. Hence D: a single stopwatch measurement, not a sprint.
- **Where the services should live.** The portfolio reading argues that the handheld is the weakest server you own and that the Sparks should host the bots (E). The others keep llama.cpp on the console as the host. The answer depends on question 3.
- **The engineering-first judge said yes**, scored C at 8.5 and told you to keep going as a hobby. The record-only judge scored the same option 3.5, because C's success rule demands exactly the finishing work the record shows gets skipped.
- **D or F.** F gives a cleaner ending. D keeps a cheap way back. Everyone agreed that deleting the containers without exporting them is the one step that cannot be undone.
- **What plan v2 costs.** Estimates for month 1 ranged from 40–60 h to 60–100 h.
- **A shared caveat.** Several strategists noted that llama.cpp has worked since March without becoming a daily habit. Local model quality, or what you actually enjoy, may be the real bottleneck rather than the engine.

---

## 8. What this review could not know

- **Your hours.** None were logged, so every phase total is an estimate.
- **Current llama.cpp speed on this console.** The ~25 tok/s figure comes from other machines.
- **Whether a local 20–35B model is good enough** to replace the cloud model the bots use now.
- **How often you code offline with long prompts**, and how long a wait you will tolerate.
- **Whether the Sparks are reachable from away**, and what running them all day costs.
- **Whether a frozen vLLM container survives** operating-system and driver updates.
- **The OpenVINO Model Server figure** (38 tok/s, Intel's own server) is unverified here.
- **Which way upstream will move** on integrated GPUs.
- **Above all, what you value most:** the engineering itself, or the assistant. This review can lay out the choices for each answer, but only you can give that answer.
