# Should the vLLM-on-Lunar-Lake work continue? A whole-picture review (2026-11-11)

Terms: llama.cpp and vLLM are two programs ("engines") that run AI models on your own machine. Tokens/s is how fast the answer is typed. "Upstream" is the main vLLM project, which ships a release about every two weeks. A pull request (PR) is a change offered to that project. Hours were never logged; all are estimates, marked "est."

## 1. The short answer

No, not as the main effort. Pause the vLLM work, save it properly so it can be resumed, and spend the next month putting the coding agent and both bots into daily use on llama.cpp, starting with one two-hour measurement of a current llama.cpp build on your console. Your speed target (about 20 tokens/s) was met in April; what has been missing since then is a job in daily use, and work plan v2 does not contain a step that produces one. Nothing is deleted, and dated rules below say when to come back. This flips on one condition: if the engineering itself is what you want (the building, being first on this chip, contributing), keep vLLM as a time-capped project with public results (option E), with the assistant no longer depending on it.

## 2. What the last eight months show

| Phase | What the time went into | What it bought |
|---|---|---|
| Mid-March | Setting up llama.cpp (a few days) | A working local server at 9.5-17 tokens/s, with essentially no upkeep since |
| March-April | The vLLM port: about 250 commits across two repositories, long builds, patches | 18-30 tokens/s, so the target was met; one upstream PR opened |
| May-August | Keeping the stack alive across five upstream releases; nothing saved in git | The stack still runs, but the fixes live only inside one software environment |
| August-November | About ten "should I upgrade?" checks; about ten reports, studies and plans; one script | Knowledge of where upstream is going; no job in daily service |

**Building versus using.** The whole effort is roughly 200-400 hours (est.; the May-August evenings are unrecorded), with under a tenth on llama.cpp (est.). Time spent using the vLLM stack for the four jobs is close to none: none of them is in daily service on it. Since September the fork has gained about ten lines of documents for each line of code.

**Three patterns.**

1. *The goal was reached, then the finish line moved.* Since April the work has been keeping a result alive and re-proving it. That has no end date, which is what "I will never finish" describes.
2. *Documents and checks replaced running things.* Coding agents make reports nearly free; your evenings at the console are the scarce thing. Asking the upgrade question ten times points to a missing rule, not missing information. Meanwhile upstream removed most of your patches without your help (14 down to two or three), so waiting has paid about as well as chasing.
3. *The jobs did not happen on either engine.* llama.cpp has worked since March and the agent and bots were not finished on it either, so the engine was not the only obstacle. The cheapest decisive measurement, a current llama.cpp build on this console, has never been taken; plan v2 leaves it out.

## 3. vLLM versus llama.cpp as a way of working

| What shapes your weeks | llama.cpp | vLLM on this console |
|---|---|---|
| Time to a working setup | Days | Several weeks of intense work (March-April) to the first results; not "finished" eight months on |
| Upkeep per month | Essentially none since March | No measured figure; 10-25 hours per month or per upstream generation (est.) |
| What breaks, how often | Nothing recorded here since March | The environment breaks if the wrong library is found; memory can leak after a crash and need a reboot; each fortnightly release raises the upgrade question |
| How results arrive | By themselves: a rebuild picks up other people's work (about 25 tokens/s reported on similar machines, not measured here) | Through your evenings: patches, 1.5-2 hour builds, re-measurement |
| Speed where it matters | Bots: fine, and it loads in seconds and unloads when idle. Coding agent: slow to read a long prompt for the first time (70-95 seconds est. for a long one) | Coding agent: reads long prompts several times faster (about 20 seconds, measured once on an older version). Bots: the extra speed is barely felt; loads take 36-40 seconds |
| Dependence on upstream | Low: one program, no local patches | High: releases, the graphics driver and other parts move separately; a new generation has just started and needs another changeover |
| How alone you are | Many people run the same chip family and report the same faults | You are the only known user on this kind of chip; upstream tests only on separate graphics cards |
| What is learned | Little | A great deal, and it is unique. It also transfers to the Sparks, where vLLM is fully supported |

vLLM's remaining lead matters for one job: the coding agent on long prompts, when you cannot reach the Sparks. For the bots, voice and "low upkeep", the simpler engine fits better.

## 4. The options

**A. Carry out plan v2 as written.**
Week to week: four weeks of evenings measuring, including a first trial of speculative decoding (letting the model guess several words ahead), then re-measuring when the next upstream generation settles. Bet: the measurements give a clearly faster stack and upstream holds still. Hours: 40-60 in month one (est.; estimates ranged 25-100), then 10-25 a month (est.). Best case: a simpler, faster stack and a first-of-its-kind result. Likely failure: day 30 brings fresh numbers, no job in use and a plan v3. Warning sign: day one overruns two sittings. Suits: an owner for whom measuring is the point.

**B. Freeze vLLM and put the jobs on it now.**
Week to week: pin one version, stop measuring, run the coding agent on it daily; upgrade at most once per upstream generation. Bet: fast long-prompt reading is felt every day and you can hold a freeze. Hours: 20-30 in month one, then 4-10 a month (est.). Best case: April's work finally pays rent. Likely failure: the freeze quietly breaks, or the memory leak turns every engine switch into a reboot. Warning sign: memory does not come back after a normal stop. Suits: someone who codes on the console away from home several times a week.

**C. Three-week trial, then decide.**
Week to week: measure current llama.cpp, an Intel-made server and the newest vLLM here, use one job for two weeks, decide on one page. Bet: the unknowns are cheap to find and you will obey thresholds written beforehand. Hours: 10-15 stated, 18-28 more likely (est.). Best case: a decision made on your own console's numbers. Likely failure: it becomes one more measurement round. Warning sign: measuring starts before the thresholds are written as numbers. Suits: someone truly undecided who keeps written rules.

**D. Pause and hold: jobs on llama.cpp, vLLM saved as an option to resume.**
Week to week: two evenings saving the vLLM stack and measuring llama.cpp; then the bots and coding agent in daily use, heavy work to the Sparks; no release checks. Bet: you mainly want working assistants and current llama.cpp is good enough here. Hours: 25-35 in month one (mostly wiring up the jobs), then 4-8 a month (est.). Best case: the console becomes an appliance, and "do I need vLLM?" is answered from your own use. Likely failure: the pause leaks, through release-reading or tuning llama.cpp instead. Warning sign: a version check or new document before any job is in daily use. Suits: someone who wants the assistant and is tired of the release treadmill.

**E. Separate the hobby from the appliance.**
Week to week: jobs on llama.cpp as in D; vLLM gets a fixed slot (one full day a month) and each session ends with something public: patches in git, a dated write-up, the PR settled, small upstream items. Bet: you value the engineering, and what was missing was a boundary and visible results. Hours: 6-16 a month capped, on top of D; a three-week closing sprint is 30-45 (est.). Best case: the spring work becomes a finished public result while it is still unique. Likely failure: publishing queues behind measuring again. Warning sign: the first session produces a measurement, not a commit and a PR comment. Suits: someone who would do the port even if it never made the assistant faster.

**F. Stop vLLM on the handheld; the Sparks become the server.**
Week to week: archive the work, move the bots and the main coding model to a Spark, keep llama.cpp on the handheld for when there is no network. Bet: the Sparks are reachable from where you work and you will not miss the port. Hours: 15-25 in month one, or 30-50 if the remote link must be built; then 2-6 a month (est.). Best case: the handheld is a handheld again and the upgrade question disappears. Likely failure: the tinkering moves to the Sparks, or they are unreachable. Warning sign: requests landing on the cloud fallback in the first week away. Suits: someone who codes mostly within reach of home.

**Judges' scores (0-10).** Two judges each assumed a different owner; the third took no view of what you want and judged only from the record.

| Option | Assistant-first | Engineering-first | Record-only |
|---|---|---|---|
| A. Plan v2 as written | 1 | 6 | 1.5 |
| B. Freeze and serve on vLLM | 4.5 | 4 | 4 |
| C. Three-week trial | 6 | 5 | 6 |
| D. Pause and hold | 8.5 | 4.5 | 7.5 |
| E. Capped hobby, public results | 4 | 8.5 | 4 |
| F. Stop; Sparks serve | 7 | 2 | 5 |

## 5. The recommendation in detail

Option D, entered through a one-week check (the useful core of C).

**Next 30 days (to 11 December 2026)**

By 18 November (8-10 hours est.):
1. Before touching the console, write the pass marks on half a page and date it: about 20 tokens/s, and no more than 60 seconds before the first word on a long coding prompt read for the first time (a limit already used in your own comparison report).
2. Save the vLLM stack: export the working container (the packaged copy, v0.32.0) to a file, put the patches into git, write a one-page restart note with the system versions that worked, and restore from the file once to prove it works. Every option needs this; it is the one thing that can be lost for good.
3. PR #40295: post a short honest note and close it (15 minutes). Do not rebase it; that is vLLM work in disguise. If question 1 below is a yes, post the note and leave it open.
4. One sitting, two hours, hard stop: build current llama.cpp and record, once each, the typing speed and the wait on one real long coding request, first time and on a follow-up.

By 2 December: llama.cpp starts by itself and unloads when idle; the coding agent and both bots are in daily use; heavy work goes to the Sparks when reachable. Keep a paper tally: days used, minutes spent fixing, each wait that annoyed you, each urge to check a vLLM release.

By 11 December: a first voice or OCR endpoint, only if the first three jobs are in daily use. Read the tally.

**Stop doing.** Release checks, until 28 February 2027. Plan v2 (file it; do not run it). New plans, studies and comparisons. Benchmarking llama.cpp beyond the one sitting. Testing a third engine.

**Keep so nothing is lost.** The saved container, patches and restart note; the April numbers with dates and versions; the reports as a record; the vLLM knowledge, which is usable on the Sparks.

**Stop and resume rules**

- *25 November:* pass marks met and the agent used on four of its first five working days: carry on.
- *11 December:* if the first-read wait is well over 60 seconds, the tally is mostly long-prompt waits, and the Sparks are not a usable route for those sessions, start the saved vLLM by hand for long coding sessions only, provided a normal stop gives the memory back without a reboot.
- *Any time:* if the "urge" tally is longer than the annoyance tally, or your answer to question 1 below is yes, switch to option E. Begin with one four-hour session that publishes; it must not measure.
- *Any week:* if the agent is used on fewer than three of five days for reasons unrelated to speed, the engine was never the obstacle. Stop stack work and ask whether you want a local coding agent on the handheld.
- *28 February 2027:* one 30-minute review, five yes/no lines: long-prompt waits are my main annoyance; a stock vLLM release runs my main model with no patch; upstream now tests chips like mine; a model I want runs well only on vLLM; I have missed it. Any yes: one boxed weekend on vLLM, then decide again. All no: stop holding and move to option F.

## 6. Five questions only you can answer

1. **If the agent and bots were handed to you finished tomorrow on llama.cpp, would you still want to spend evenings on the vLLM port?** If yes, then option E. If no, then option D now and F later.
2. **Do you spend several hours a month coding with long prompts and no route to the Sparks at all?** If yes, then the narrow form of B (vLLM for the coding agent only), once the two-hour measurement confirms the wait. If no, then D or F; vLLM's remaining lead does not reach your weeks.
3. **Will you name a number of hours a month and stop when a written rule fires?** If yes, then a trial (C) or a capped hobby (E) can work for you. If no, then choose the plain pause (D) with no hobby slot.
4. **Does being first on this chip, or having a change accepted upstream, matter to you personally?** If yes, then do E's closing sprint: one dated public write-up while the result is still unique. If no, then close the PR with a note and stay with D or F.
5. **Are the Sparks on and reachable from where you usually work, and does "offline" mean "no cloud company" more than "no network"?** If yes, then F is where this ends up, with the handheld as fallback. If no, then keep everything local on llama.cpp as in D.

## 7. Where the strategists and judges disagreed

- **Pause or continue.** Eight of nine strategists leaned to pausing. The motivation-and-identity strategist leaned to continuing with limits, reading the spring burst as craft that deserves a finish. The engineering-first judge agreed (E 8.5, D 4.5); the other two put D first.
- **Measure first or commit first.** Two strategists wanted a trial before deciding. The record-only judge answered that measuring followed by a one-page decision is what August to November looked like. Hence one week here, not three.
- **Where the assistant lives.** The portfolio strategist would move it to a Spark now. The judges treated that as the likely destination, not the first step, because the remote route is still listed as "Future" in your own repository.
- **The pull request.** Some want it refreshed with real readings so the spring work counts; the assistant-first judge calls that a way back into vLLM work.
- **The Intel-made server.** Several strategists wanted a one-hour test early; the stress tests called it a trapdoor to a third porting project.
- **What plan v2 costs.** First-month estimates ranged from 25 to 100 hours; the plan does not total itself.

## 8. What this review could not know

- How much you value the engineering itself. Every recommendation hinges on it, and no measurement settles it.
- What a current llama.cpp build does on this console. The figures here are from March; the 25 tokens/s figure is from other machines.
- How many hours the project has really taken, especially May to August.
- How often you are out of reach of the Sparks, and whether they are reachable from outside today.
- Whether the bots' local model does much real work today, since they use a cloud model first.
- Whether current vLLM still leaks memory after a normal stop (last measured in April), and whether a system update would break a saved copy.
- Whether upstream would accept changes for a chip it cannot test, and whether the Intel-made server's reported 38 tokens/s holds here.
- Some web sources could not be reached, so the view of upstream activity rests partly on search summaries.
