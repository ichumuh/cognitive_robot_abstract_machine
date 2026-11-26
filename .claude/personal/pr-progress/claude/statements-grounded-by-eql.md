Branch claude/statements-grounded-by-eql (stacked on claude/milk-scenario-on-main): let EQL ground the statements of a plan together.

Plan
- EQL rank-sum order + evaluate_under/Answer + nested where; cramph joint statement under values fixed by other statements; coraplex trial as SucceedsItsTrial predicate; grounding defaults to RANK_SUM

Done
- all implemented, uncommitted; pre-commit clean; ORM regenerated
- tests green: krrood test_eql 1637, cramph_test 932, coraplex test_designator 580, test_plan 276

Next
- user to review; commit/push and PR only when asked (labels: api-break, new-feature)
