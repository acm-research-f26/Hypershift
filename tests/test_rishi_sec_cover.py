import unittest
from rishi_sec_cover import parse_filing, select_filing

class SecCoverTests(unittest.TestCase):
    def test_cover_and_header_require_consistent_identity(self):
        header=b'ACCESSION NUMBER: 0001-24-000001\nCENTRAL INDEX KEY: 0000000001\nSTANDARD INDUSTRIAL CLASSIFICATION: COMPUTERS [3571]'
        doc=b'<html><ix:nonnumeric name="dei:TradingSymbol">AAPL</ix:nonnumeric></html>'
        self.assertEqual(parse_filing(header,doc,1,'0001-24-000001','AAPL')[0],'matched')
        self.assertEqual(parse_filing(header,doc,1,'0001-24-000001','MSFT')[0],'ticker_not_tagged')
        self.assertEqual(parse_filing(header,doc,2,'0001-24-000001','AAPL')[0],'header_identity_mismatch')

    def test_latest_original_preceding_filing_and_staleness(self):
        rows=[{'filingDate':'2024-01-01','form':'10-K','accessionNumber':'a','primaryDocument':'x'},
              {'filingDate':'2024-12-01','form':'10-Q/A','accessionNumber':'b','primaryDocument':'x'},
              {'filingDate':'2025-01-03','form':'10-Q','accessionNumber':'c','primaryDocument':'x'}]
        self.assertEqual(select_filing(rows,'2025-01-03')['accessionNumber'],'a')
        self.assertIsNone(select_filing(rows[:2],'2025-02-05'))
