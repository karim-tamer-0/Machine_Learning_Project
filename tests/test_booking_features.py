import pandas as pd
 
from src.features.booking_features import BookingFeatures
 
 
def _toy():
    return pd.DataFrame({
        "Company": ["NoCompany", "40", "NoCompany", "223"],
        "StaysInWeekendNights": [1, 0, 2, 1],
        "StaysInWeekNights": [2, 3, 0, 4],
        "Adults": [2, 1, 2, 2],
        "Children": [0, 0, 1, 0],
        "Babies": [0, 0, 0, 1],
        "LeadTime": [10, 200, 5, 90],
    })
 
 
def test_has_company_flag_and_company_id_dropped():
    out = BookingFeatures().fit_transform(_toy())
    assert list(out["has_company"]) == [0, 1, 0, 1]
    assert "Company" not in out.columns
 
 
def test_totals_are_correct():
    out = BookingFeatures().fit_transform(_toy())
    assert list(out["total_nights"]) == [3, 3, 2, 5]
    assert list(out["total_guests"]) == [2, 1, 3, 3]
 
 
def test_input_is_not_modified():
    df = _toy()
    BookingFeatures().fit_transform(df)
    assert "Company" in df.columns and "has_company" not in df.columns
 
 
def test_stateless_fit_on_other_data_gives_same_output():
    other = _toy().iloc[:2]
    a = BookingFeatures().fit(other).transform(_toy())
    b = BookingFeatures().fit(_toy()).transform(_toy())
    pd.testing.assert_frame_equal(a, b)
 
 
def test_works_when_company_column_is_absent():
    out = BookingFeatures().fit_transform(_toy().drop(columns="Company"))
    assert "has_company" not in out.columns
    assert "total_nights" in out.columns