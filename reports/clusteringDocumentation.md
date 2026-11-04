# **Analysis**
We loaded the CSV file into the variable cc and started inspecting and analyzing the data.

Initially, we set our goal to identify customer segments, understand how customers relate to each other, and group similar customers into clusters. Later, we can use these clusters to determine which customer segments are most important to the hotel, which ones generate more revenue and why, what makes those customers happier, and what they look for in our hotel.

At the beginning, we excluded some features that did not seem to contribute to our goal, such as IsCanceled, AssignedRoomType, BookingChanges, ArrivalDateYear, ArrivalDateWeekNumber, ReservationStatus, ReservationStatusDate, Agent, and ADR.

We might use ADR later to assess the clusters and understand which customer segments generate more revenue. However, we decided not to use it for training because our goal is to identify similarities between customers based on their booking information, rather than grouping them based on the revenue they generate.

These excluded features either represent an outcome of the booking or contain information that could make the model's job easier in a way that does not help us achieve our actual goal.

After analyzing the remaining data, as shown in the analysis notebook, we found some columns with very low variance, where almost all rows had the same value. These columns were not expected to contribute much to distinguishing between customers. Examples include IsRepeatedGuest, PreviousCancelations, PreviousBookingNotCanceled, DepositType, Company, DaysInWaitingList, and RequiredCarParkingSpaces.

A lot of the analysis was done using the same function while changing the columns being analyzed. Therefore, it might look like we only analyzed one column in the notebook, but we actually repeated the analysis for all the columns we had.

Now that the initial analysis is complete, we can move on to processing the data.



# Data Processing

## Feature Encoding

After selecting the features that we want to use for K-Means, we started converting the categorical features into numerical values.

We used One-Hot Encoding for the following features:
- Meal
- MarketSegment
- DistributionChannel
- ReservedRoomType
- CustomerType

Each category was converted into its own column containing 0s and 1s. This allows us to represent categorical values numerically without assigning arbitrary numerical rankings to them.

For the Country feature, we found that it contained 166 unique categories. Instead of creating a column for every country, we decided to keep the 15 most frequent countries as separate categories and group the remaining countries into an "Other" category.

We then applied One-Hot Encoding to the resulting Country categories.

For ArrivalDateMonth, we used sine and cosine transformations to represent the cyclical nature of the months. This resulted in two new features:
- Month_sin
- Month_cos

We also included ArrivalDateDayOfMonth as a numerical feature without applying any encoding, since we decided to keep its original values.

## Creating the K-Means Dataset

After finishing the encoding, we created a new DataFrame called `kmeans_data` containing only the selected features and their new numerical representations.

This allowed us to work on the K-Means data separately without making further changes to the original DataFrame.

The resulting dataset initially contained 79,330 rows and 54 columns.

## Handling Missing Values

We checked the selected features for missing values and found only 4 missing values, all located in the Children column.

Since these were only 4 rows out of the entire dataset, we decided to replace the missing values with 0.

## Checking Invalid Values

We inspected the numerical features to check for negative values.

All numerical features had zero negative values.

We also checked bookings with zero adults. We found 390 bookings with no adults, so we decided to remove these rows because they do not represent normal hotel customer bookings for our clustering purpose.

After removing them, the dataset contained 78,940 rows and 54 columns.

We also checked the one-hot encoded features to make sure that each booking had exactly one active category for each categorical feature.

## Outlier Inspection

We used boxplots and the IQR method to identify potential outliers in the numerical features.

Some features contained a number of potential outliers, especially LeadTime, Adults, Children, ArrivalDateDayOfMonth, and Babies.

However, after examining the distributions, we decided to keep these values because they can represent valid hotel bookings. For example, a booking with 3 adults or a long stay is not necessarily an invalid record.

Therefore, we kept the identified outliers instead of automatically removing them.

## Handling ArrivalDateDayOfMonth

We included ArrivalDateDayOfMonth as a regular numerical feature without applying any encoding.

We checked its missing values and verified that its values were within the valid range of 1 to 31.

We decided to keep the original values because we wanted to use the day of the month as a numerical feature.

## Removing Extreme Values

During the inspection of the Babies feature, we found two bookings containing unusually high values: one booking had 9 babies and another had 10 babies.

Since these values were extremely unusual compared to the rest of the dataset, we decided to remove these two bookings from the clustering dataset.

After removing them, the dataset contained 78,938 bookings.

## Feature Scaling

Since K-Means relies on distance calculations, features with larger numerical ranges could have a greater influence on the clustering results.

To avoid this, we applied standardization using StandardScaler from Scikit-learn.

We standardized the following 8 numerical features:
- LeadTime
- Adults
- Children
- Babies
- TotalOfSpecialRequests
- StaysInWeekendNights
- StaysInWeekNights
- ArrivalDateDayOfMonth

Standardization transforms the features so that their mean is approximately 0 and their standard deviation is approximately 1.

We did not standardize the one-hot encoded categorical features because they were already represented using binary values (0 and 1).

We also left Month_sin and Month_cos unchanged because they were already represented within the range of -1 to 1.

## Final Clustering Dataset

After completing the preprocessing steps, we performed final checks on the dataset.

The final results were:
- Number of bookings: 78,938
- Number of features: 55
- Missing values: 0
- Numerical features: 8 standardized features
- Cyclical features: 2 unchanged features
- One-hot encoded features: 45 binary features

We saved the resulting dataset as `kmeans_cleaned_scaled.csv`.

This file will be used as the input dataset for the K-Means clustering experiments.