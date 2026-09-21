Predicting Credit Card Default
Predicting which credit card customers will miss their next payment, using six months of billing and repayment history from 30,000 cardholders.

Key findings
Final model: XGBoost on raw features plus engineered payment ratios. Held-out test ROC-AUC 0.760, against a cross-validated estimate of 0.761. The CV estimate held up.
Operational value: contacting the model's top-ranked 10% of customers reaches ~30% of all defaulters, about three times what random selection achieves with the same number of calls. Two-thirds of those contacts reach a customer who does go on to default.
Feature engineering's value for the "currently up to date" subgroup is small and run-dependent. In the current notebook run, engineered features moved XGBoost AUC on this subgroup from 0.638 to 0.647 — a gain smaller than the fold-to-fold standard deviation, i.e. not clearly distinguishable from noise. An earlier run of the same notebook found a much larger gain (0.557 → 0.632). See Section 6 for both and why they differ.
Removing protected attributes cost little (AUC 0.761 → 0.757), but did not remove their influence entirely. The model still flags defaulters with graduate degrees about 7 points less often than university-educated defaulters, even when education is excluded from the inputs.
1. The problem
A credit card issuer wants to identify, at the end of September, which customers are likely to miss their October payment, so that a collections team can intervene early.

This is a binary classification problem. The target is default.payment.next.month: 1 if the customer failed to make their minimum payment in October 2005, 0 otherwise. "Default" here means a single missed payment, not a charged-off account, which is a much narrower definition than the word usually carries in credit risk.

The model outputs a probability of default for each customer. Turning that into an action is treated as a separate business decision (section 7).

2. Data
Default of Credit Card Clients, UCI Machine Learning Repository. 30,000 credit card clients in Taiwan, April to September 2005. Amounts are in New Taiwan dollars.

Group	Columns
Credit and demographics	LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
Repayment status	SEP_REPAY_STATUS … APR_REPAY_STATUS (months behind, per month)
Bill amount	SEP_BILL_AMT … APR_BILL_AMT (statement balance, per month)
Payment amount	SEP_PAY_AMT … APR_PAY_AMT (amount paid, per month)
The original column names (PAY_0, PAY_2, BILL_AMT1, etc.) were renamed to the month-labelled versions above. Overall default rate is 22.1%.

Cleaning decisions
Undocumented category codes. EDUCATION contains values 0, 5, and 6, and MARRIAGE contains 0, none of which appear in the documentation. These were merged into each variable's existing "other" category (4 for education, 3 for marriage) rather than dropped.

Repayment status encoding. The documentation defines -1 as paid duly and positive integers as months of delay, but the data also contains -2 and 0. Following the common interpretation, -2 is treated as no balance and 0 as revolving credit with the minimum paid. In practice this ambiguity turns out not to matter much: all three non-positive codes have similar default rates (13–17%), so they behave as a single "not late" group.

Negative and zero bills. Around 10% of monthly bills are zero or negative. A negative bill means the customer is in credit (overpayment, a refund after payment, a statement credit). These are real customers, not data errors, and were kept.

3. Exploratory analysis
Repayment status is by far the strongest signal. Default rate by September repayment status:

Status	Meaning	Default rate
-2, -1, 0	Not late	13–17%
1	1 month behind	34%
2	2 months behind	69%
3	3 months behind	76%
4+	4+ months behind	50–78%, small groups
One month behind roughly doubles default risk; two months quintuples it. Statuses above 3 contain very few customers each, and their rates should not be over-read. The same pattern holds for earlier months, flattening somewhat the further back you go.

Credit limit shows a clear gradient. Default rate falls from 32% in the lowest limit quintile to 14% in the highest. Credit limit is set by the bank, so it already encodes the bank's own prior assessment of the customer.

Utilization matters at the extreme. 2,115 customers had a September balance above their credit limit. They defaulted at 30.1%, against the 22.1% base rate.

Demographics barely move. Default rates span roughly 19% to 27% across age, education, and marital groups, compared with 13% to 76% across repayment statuses.

Variable	Group	Default rate
Age	Youth	27%
Adults	22%
Seniors	25%
Education	Graduate school	19%
University	24%
High school	25%
Other	7% (small group)
Marriage	Married	23%
Single	21%
Other	24%
4. Baseline
All modelling uses an 80/20 stratified train/test split (24,000 / 6,000). The test set was set aside and used exactly once, in section 8. Everything before that uses 5-fold stratified cross-validation on the training set.

The baseline uses the 23 raw columns with no engineering and no tuning.

Model	ROC-AUC	Fold σ
Logistic regression	0.726	0.007
XGBoost	0.759	0.004
Accuracy is not used as a headline metric. A model that predicts "nobody defaults" scores 0.779 accuracy on this data, and the two models above score 0.810 and 0.812, which makes them look nearly identical when AUC shows a clear gap. ROC-AUC and PR-AUC are used throughout instead.

The XGBoost advantage indicates non-linear structure that a linear model can't reach. Logistic regression is also handicapped by treating SEX, EDUCATION, and MARRIAGE as numbers.

5. Feature engineering
Raw amounts are hard to interpret without a denominator: a 3,000 balance means something very different on a 10,000 limit than on a 200,000 one. Every engineered feature expresses a relationship between raw columns.

Payment ratios (5 columns). Amount paid ÷ prior month's bill. The September payment settles the August bill, so each payment is paired with the previous month's balance; April has no prior month. A ratio near 1 means paying in full, near 0.05 means paying the minimum, 0 means paying nothing. Where the bill is zero or negative, the ratio is undefined and set to missing rather than zero, since "no bill to pay" and "paid nothing" are different situations. Ratios are clipped at 2.

The distribution is strongly bimodal: the median customer pays about 7.5% of their bill, while the top quarter pay essentially all of it.

Utilization (6 columns). Bill ÷ credit limit, per month. Negative values (customer in credit) and values above 1 (over limit) were both kept, since both are meaningful.

Aggregates and trends. Per-customer summaries across the six months: average, max, min, and standard deviation of utilization; utilization trend (September minus April); payment ratio trend; bill growth ratio (September ÷ April bill); and count of months with no outstanding balance (zero utilization).

6. Results
Full population
Feature groups were added cumulatively, with the same model, folds, and data each time.

Feature set	XGBoost AUC	Fold σ
Raw	0.7585	0.0038
+ payment ratios	0.7610	0.0041
+ utilization	0.7609	0.0056
+ aggregates	0.7576	0.0044
All four configurations fall within a range of 0.0034, smaller than the fold-to-fold standard deviation. No feature group produced a meaningful improvement. Raw + payment ratios was carried forward as the simplest of the tied-best configurations.

The likely explanation is redundancy. XGBoost already has the raw bill amounts, payment amounts, and credit limit, and can approximate ratios between them through successive splits. And the repayment status columns already carry most of the "this customer is struggling" signal that the engineered features also capture.

Customers who currently look fine
The full-population result leaves an open question. A customer who is already two months behind is obviously high-risk, and a bank doesn't need a model to see that. The more useful question is: among customers who are currently up to date, who goes bad next month?

This section repeats the analysis on customers with a September repayment status of 0 or below, and also drops the repayment-status columns from the feature set entirely (demographics, bill amounts, and payment amounts remain). The problem gets substantially harder, which is the point.

Feature set	Logistic regression AUC	XGBoost AUC
Raw (no repayment status)	0.615	0.638
+ payment ratios	0.616	0.636
+ utilization	0.617	0.646
+ aggregates	0.617	0.647
On this run, the engineered features barely move either model: logistic regression gains 0.002 AUC and XGBoost gains 0.009, both comparable to or smaller than the fold-to-fold standard deviation (roughly 0.005–0.014 across these rows). Read at face value, this run doesn't support the "engineered features become valuable once repayment status is removed" hypothesis — the full-population conclusion (features are redundant) seems to extend here too.

This is not the whole story, though. An earlier run of this exact same notebook cell found a much larger effect: XGBoost AUC moving from 0.557 (raw) to 0.632 (+ aggregates), a 0.075 AUC gain — about seven fold-standard-deviations, a genuinely large and reproducible-looking effect at the time. The two runs used the same code, the same random seeds, and the same train/test split; the difference is almost certainly the xgboost/scikit-learn versions installed in the environment each run used. The conclusion for this subgroup should be treated as unsettled, not as an established result — re-running this notebook in a pinned environment (see Limitations) is needed before trusting either number.

7. From probabilities to decisions
The model outputs probabilities, and acting on them requires a cutoff. At the default of 0.5, the model catches only 36.5% of defaulters, which is too conservative to be useful.

Cost-based threshold
Assuming a missed default costs 1,000 and an unnecessary intervention costs 50 (a 20:1 ratio), total cost was computed across thresholds from 0.01 to 0.95. These costs are invented; the dataset contains no information about intervention or recovery costs.

Cost ratio (miss : false alarm)	Cost-minimizing threshold
5 : 1	0.15
20 : 1	≤ 0.01
At 20:1, the optimum hits the floor of the sweep, meaning the cost-minimizing policy is to flag nearly every customer. That's correct under the assumption and useless in practice: when misses are expensive enough and 22% of customers default, blanket intervention beats any selective strategy. This says more about the assumed costs than about the data.

Capacity-based threshold
Capacity is a better constraint, because it's one a collections team actually knows. Rather than asking which threshold minimizes cost, the question becomes: given that we can contact a fixed number of customers, which ones should they be?

This treats the model as a ranking. Sort customers by predicted probability and work down the list until capacity runs out. The threshold is whatever probability sits at the cutoff position.

Capacity	Threshold	Contacted	Share of defaulters caught	Precision
5%	0.77	1,204	16.5%	72.6%
10%	0.59	2,394	30.1%	66.7%
20%	0.35	4,715	47.8%	53.8%
What this is worth. The training set contains about 24,000 customers, of whom roughly 5,300 default. At 10% capacity, the team makes 2,394 calls regardless; the model only changes which customers receive them.

Chosen at random, those calls would reach about 10% of all defaulters, since defaulters are spread evenly through a random sample: roughly 530 people. Ranked by the model, the same calls reach 30.1%, about 1,600. Three times as many at-risk customers, with no additional headcount. Two-thirds of model-selected contacts reach someone who does default, against 22% under random selection.

Doubling capacity from 5% to 10% nearly doubles recall while precision falls six points. Doubling again to 20% adds 18 points of recall but drops precision to roughly a coin flip. On this evidence, 10% is a reasonable operating point.

This framing is also why ROC-AUC is the headline metric: it measures how well the model orders customers by risk, independent of any cutoff, which matches how the output would be used.

8. Final evaluation
The final model (XGBoost, raw + payment ratios, full population) was trained on the full training set and evaluated once on the 6,000-customer held-out test set, using the 10%-capacity threshold of 0.59.

Metric	Cross-validated	Test
ROC-AUC	0.761	0.760
PR-AUC	—	0.528 (base rate 0.221)
Predicted pay	Predicted default
Actually paid	4,472	201
Actually defaulted	943	384
The model flagged 585 customers (9.8% of the test set), of whom 384 (65.6%) defaulted, capturing 28.9% of all defaulters. These closely match the cross-validated capacity estimates of 30.1% recall and 66.7% precision.

The gap between cross-validated and test AUC is 0.001, well inside the fold standard deviation. Despite the many feature and model comparisons made on the training set, the cross-validation estimate did not become optimistic.

9. Fairness check
The dataset includes sex, education, and marital status. In real lending these are restricted or prohibited as model inputs, so two questions were tested.

Does the model need them? Removing all three reduced cross-validated AUC from 0.761 to 0.757, a difference comparable to the fold standard deviation. There is little performance argument for using these attributes.

Does removing them remove their influence? Recall (the share of actual defaulters the model flags, at threshold 0.59) was compared across groups, with and without the attributes as inputs.

Attribute	Group	Recall, with attributes	Recall, without
Sex	1 (male)	31.6%	31.0%
2 (female)	28.9%	30.1%
Education	Graduate school	26.1%	25.8%
University	32.6%	32.8%
High school	30.5%	32.5%
Other	small group	small group
Marriage	Married	31.5%	30.9%
Single	28.8%	30.2%
Other	27.8%	26.4%
For sex and marital status, removing the attributes narrowed the gap between groups from about 3 points to under 1.

Education behaved differently. Defaulters with graduate degrees were flagged about 7 points less often than university-educated defaulters, and the gap persisted after education was removed from the inputs. The model recovers education-linked differences through correlated features, most plausibly credit limit, since more highly educated customers tend to receive higher limits and therefore look safer on paper.

Removing a protected attribute is not sufficient on its own to remove its influence. The "other" education group contains too few defaulters to evaluate.

10. Limitations
Old, narrow data. Taiwan, 2005, a single six-month window. Consumer credit behaviour, lending rules, and economic conditions have all changed since.
One-month horizon. The target is a single missed payment next month, not longer-term loss.
Undocumented codes. The meaning of repayment statuses -2 and 0, and several education and marriage codes, had to be inferred.
Invented costs. The cost-based threshold analysis rests on assumed costs, and only two cost ratios (5:1 and 20:1) have actually been swept in the notebook. The capacity analysis avoids the cost assumption but assumes a fixed contact budget.
No hyperparameter tuning. All models used default settings, so that differences between feature sets could be attributed to the features. A tuned model would likely score somewhat higher.
Single split. Results come from one train/test split and one cross-validation seed.
Results are sensitive to library versions, more than expected. Re-running 05_current_subgroup.ipynb in a different environment (same code, same random_state) produced a materially different result for the subgroup ablation — see Section 6. requirements.txt doesn't exist yet (see below); pinning scikit-learn and xgboost versions and committing a lockfile would make these numbers reproducible instead of environment-dependent.
Limited fairness analysis. Only recall gaps were examined, at one threshold, on training-set predictions. A fuller audit would examine precision, flag rates, and calibration by group.
Repository structure
predicting_credit_card_payment/
├── data/
│   └── raw/
│       └── UCI_Credit_Card.csv
├── notebooks/
│   ├── 01_eda.ipynb                  # cleaning, renaming, exploratory analysis
│   ├── 02_baseline.ipynb             # baseline logistic regression / XGBoost models
│   ├── 03_feature_engineering.ipynb  # payment ratios, utilization, aggregate features
│   ├── 04_modeling.ipynb             # feature-set ablation study (full population)
│   ├── 05_current_subgroup.ipynb     # ablation study restricted to currently-current customers
│   ├── 06_threshold_and_cost.ipynb   # threshold/cost/capacity analysis, final test evaluation, fairness check
│   ├── credit_card_cleaned.csv       # output of 01_eda.ipynb
│   └── credit_card_featured.csv      # output of 03_feature_engineering.ipynb
├── UCI_Credit_Card.csv.zip           # zipped copy of the raw dataset
└── README.md
Reproducing
The raw dataset is already included in the repo at data/raw/UCI_Credit_Card.csv (also zipped at the repo root as UCI_Credit_Card.csv.zip) — no download needed.
Install dependencies (no requirements.txt is currently checked in — see Limitations): pandas, numpy, scikit-learn, xgboost, matplotlib.
Run the notebooks in order, 01 through 06. Each downstream notebook depends on a CSV produced by an earlier one (credit_card_cleaned.csv from 01, credit_card_featured.csv from 03), so run them top to bottom in sequence rather than out of order.