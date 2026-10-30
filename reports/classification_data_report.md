# Classification Data Report (Member A, Milestone 1)

All numbers can be regenerated with `python -m src.data.eda`. Tables are in `outputs/classification/`, and plots are in `outputs/classification/eda/`.

## 1. Data Understanding

`H2.csv` loads with the published 79,330 rows and 31 columns. Classification starts from the team's shared file, `src/data/All_models_data.csv`, which has 78,092 rows.

| Check | Finding | Action |
|---|---|---|
| Missing values | Children 4, Country 24 in the raw file | Filled in the shared preprocessing (0 and the mode) |
| `NULL` in Agent and Company | 9.6% of Agent and 95.7% of Company | Not missing: the source paper defines `NULL` as "not applicable". Recoded to `NoAgent` and `NoCompany` |
| Text padding | Meal, room types, DepositType, Agent and Company contain padded spaces | Stripped |
| Exact duplicates | 25,902 in raw, 25,771 in the shared file | Removed (see section 1.1) |
| High cardinality | Country 166, Agent 223, Company 200 categories | The pipeline must group rare categories |
| Unusual values | Zero-guest and zero-night bookings and ADR of 0 or below | Removed by the shared preprocessing |
| Unusual values | 213 unique bookings with no adult | Removed in classification cleaning |

### 1.1 Cleaning results

| Rule | Rows removed | Rows left |
|---|---:|---:|
| Start: `All_models_data.csv` | 0 | 78,092 |
| Exact duplicate rows | 25,771 | 52,321 |
| No adult on the booking | 213 | 52,108 |
| ADR > 500 | 2 | 52,106 |
| LeadTime > 500 | 54 | 52,052 |
| Total nights > 21 | 45 | 52,007 |
| Babies > 2 | 2 | 52,005 |

**Duplicates.** This is the most consequential cleaning decision. The dataset has no booking ID, and the paper does not mention duplicates, so identical rows could be real bookings from a group block. Removing them lowers the cancel rate from 42.2% to 30.3%. Non Refund bookings fall from 12,868 to 845 before the later rules, because 93% of them are duplicates. They are removed because identical rows on both sides of a random split would let the model be tested on rows it trained on. To test the alternative, set `drop_duplicates: false` in `config.yaml`.

**Outliers.** Rows are removed using fixed limits rather than the IQR rule. The IQR rule would flag 26% of bookings on Adults alone, and 12.7% on BookingChanges, even though these are ordinary values. Fixed limits also compute nothing from the data, so cleaning can run before the split without leaking test statistics. Heavy-tailed count columns such as PreviousCancellations are kept, because their large values are plausible and informative. Models sensitive to skew should use scaling or a log transform inside the pipeline.

The outlier limits were chosen after viewing the full dataset. Each one removes a small number of extreme rows, and LeadTime > 500 removes the most, at 54 rows.

## 2. EDA Findings

- **Class balance (`01`):** 30.3% of cleaned bookings are canceled, a moderate imbalance. Accuracy alone is not enough, so F1 and PR-AUC should be reported too.
- **LeadTime (`02`):** the cancel rate rises steadily, from 11% at 0-7 days to 47% beyond 365 days.
- **Month (`03`):** the variation is mild, from about 26% in September and November to 35% in April.
- **DepositType (`04`):** Non Refund cancels at 97%, against 29% for No Deposit. This runs against intuition and is reviewed in section 4.
- **MarketSegment (`05`):** Online TA (36%) and Groups (34%) cancel most, while Corporate (12%) cancels least. The 100% Undefined bar is only 2 bookings, so it is not meaningful.
- **Distributions (`06`):** LeadTime and stay lengths are right-skewed. PreviousCancellations and PreviousBookingsNotCanceled are extremely skewed, with PreviousCancellations at a skew of 24.4.
- **Outliers (`07`):** the plot shows each limit against the deduplicated data. The ADR value of 5,400 is far from the next highest value, 510.
- **Correlation (`08`):** no numeric feature has a strong linear relationship with IsCanceled, so non-linear models and categorical features may add value.

## 3. Prediction Point

The prediction is made **when the booking is created**. This is the earliest point at which a hotel could act, for example by adjusting overbooking or contacting the guest. A feature is valid only if its value is known at that moment.

The source paper sets each observation's data point to **the day before arrival**, and values were taken from the booking as it stood then. Being a booking field therefore does not prove that a value was available at creation.

## 4. Preliminary Leakage Audit

Two feature lists are in `config.yaml`. **Strict** (20 features) is intended for reported results. **Relaxed** (24 features) adds the borderline columns so that the team can measure how much they raise scores.

| Column(s) | Decision | Reason |
|---|---|---|
| ReservationStatus | Drop | Direct leak: Canceled and No-Show correspond exactly to IsCanceled = 1 |
| ReservationStatusDate | Drop | Set when the booking reaches its final status, which is after the prediction point |
| AssignedRoomType | Drop | Finalized after creation. Bookings given a different room cancel at 5.2%, against 33.4% otherwise |
| BookingChanges | Drop | Counted until check-in or cancellation, so it is always 0 at creation |
| DaysInWaitingList | Drop | Known only once the booking is confirmed, which is after it enters the PMS |
| RequiredCarParkingSpaces | Drop | Indirect leak suspected: 0 of 1,847 bookings with a parking request canceled, against 31.4% otherwise. A perfect split like this is unlikely to be real behaviour |
| DepositType | Relaxed only | The paper derives it from payments made before arrival or cancellation, so it can reflect events after creation |
| TotalOfSpecialRequests | Relaxed only | Requests can be added after creation, so bookings that stay active longer have more chances to add them (22.2% cancel with a request, 38.7% without) |
| Country | Relaxed only | Possibly confirmed or corrected at check-in. This is our assumption and the paper does not state it, but PRT has one of the highest cancel rates (39%) |
| ArrivalDateYear | Relaxed only | Not a leak, but a future booking would have a year the model never saw |
| ADR | Strict, with a note | The paper computes it from lodging transactions, so it may include later changes. It is kept because a price is quoted at booking |
| LeadTime, stay lengths, guest counts | Strict, with a note | Known at creation, but these are the day-before-arrival values. 16.2% of cleaned bookings have at least one recorded change, and some of those changes may affect these fields. This cannot be undone with the available data |
| Agent, Company, MarketSegment, DistributionChannel, CustomerType, Meal, ReservedRoomType | Strict | Booking attributes set at creation |
| IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled | Strict | History that exists before the booking |
| ArrivalDateMonth, ArrivalDateWeekNumber, ArrivalDateDayOfMonth, Children, Babies | Strict | Requested at booking |

### Process leakage

- Cleaning uses only row-level rules with fixed limits, so it can run before the split.
- The shared preprocessing fills Country with a mode computed on all the data. This is a tiny effect (24 rows), and Country is excluded from the strict list.
- Scaling, encoding, imputation and rare-category grouping must be fitted on the training data only, inside the pipeline.
- This EDA looked at all the data, including rows that will later form the test set. It should only be used to describe the data, and final test scores must not be used to revise these choices.

## Notes for Member B

- Run `clean_classification()` before splitting, then use `get_features("strict")`.
- Report strict results as the main results. The gap between strict and relaxed scores is useful evidence for the leakage section.
- Rows are removed for training only. The application cannot drop a user's booking, so it needs input validation for values beyond the outlier limits.
- `src/classification` and `src/features` are still 2-byte placeholder files in the repository. They must be deleted before those paths can become folders.
