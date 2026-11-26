Branch claude/statements-grounded-by-eql (stacked on claude/milk-scenario-on-main): let EQL ground the statements of a plan together.

Plan
- EQL rank-sum order + evaluate_under/Answer + nested where; cramph joint statement under values fixed by other statements; coraplex trial as SucceedsItsTrial predicate; grounding defaults to RANK_SUM

Done
- committed 5fa876b264, pushed to origin; pre-commit clean; ORM regenerated
- tests green: krrood test_eql 1637, cramph_test 932, coraplex test_designator 580, test_plan 276

Next
- draft PR cram2#1074 open (api-break, new-feature, bug-fix), stacked on #1071; discuss the api-break with the code owners before requesting review
