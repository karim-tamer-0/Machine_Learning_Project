# **Analysis**
We loaded the CSV file into the variable cc and started inspecting and analyzing the data.

Initially, we set our goal to identify customer segments, understand how customers relate to each other, and group similar customers into clusters. Later, we can use these clusters to determine which customer segments are most important to the hotel, which ones generate more revenue and why, what makes those customers happier, and what they look for in our hotel.

At the beginning, we excluded some features that did not seem to contribute to our goal, such as IsCanceled, AssignedRoomType, BookingChanges, ArrivalDateYear, ArrivalDateWeekNumber, ReservationStatus, ReservationStatusDate, Agent, and ADR.

We might use ADR later to assess the clusters and understand which customer segments generate more revenue. However, we decided not to use it for training because our goal is to identify similarities between customers based on their booking information, rather than grouping them based on the revenue they generate.

These excluded features either represent an outcome of the booking or contain information that could make the model's job easier in a way that does not help us achieve our actual goal.

After analyzing the remaining data, as shown in the analysis notebook, we found some columns with very low variance, where almost all rows had the same value. These columns were not expected to contribute much to distinguishing between customers. Examples include IsRepeatedGuest, PreviousCancelations, PreviousBookingNotCanceled, DepositType, Company, DaysInWaitingList, and RequiredCarParkingSpaces.

A lot of the analysis was done using the same function while changing the columns being analyzed. Therefore, it might look like we only analyzed one column in the notebook, but we actually repeated the analysis for all the columns we had.

Now that the initial analysis is complete, we can move on to processing the data.