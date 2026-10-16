# Data manifest

Raw SEC bulk archives are not redistributed. To reproduce, download them under the SEC fair-access policy (declare a contact User-Agent):

- https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
- https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip

Place them in `data/raw/` and check them against `SHA256SUMS` (archives retrieved on the UTC date in `retrieved_at_utc.txt`). SEC refreshes these archives nightly, so a later download will differ; the hashes identify the exact snapshot used in the paper.
