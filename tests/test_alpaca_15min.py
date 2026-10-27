"""Offline behavior tests: pagination, secrets, bar timing, and immutable resume."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

import fetch_alpaca_15min as alpaca


@contextlib.contextmanager
def workspace_temporary_directory():
    base = (alpaca.ROOT / 'tmp').resolve()
    base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as directory:
        if not Path(directory).resolve().is_relative_to(base):
            raise RuntimeError('Temporary test directory must stay in workspace tmp.')
        yield directory


def response(status=200, bars=None, token=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = dict(bars=bars if bars is not None else {}, next_page_token=token)
    return result


def bar(timestamp='2025-01-02T14:30:00Z', close=101.):
    return dict(t=timestamp, o=100., h=102., l=99., c=close, v=1000, n=20, vw=100.5)


class AlpacaTests(unittest.TestCase):
    def setUp(self):
        self.start = alpaca.date_bound('2025-01-02')
        self.end = alpaca.date_bound('2025-01-03')

    def client(self, session):
        return alpaca.AlpacaClient({'APCA-API-KEY-ID': 'test-key',
                                   'APCA-API-SECRET-KEY': 'test-secret'},
                                  session=session, sleep=Mock(), clock=lambda: 1.)

    def test_env_literal_values_and_environment_precedence(self):
        with workspace_temporary_directory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('export APCA_API_KEY_ID="local-key"\n'
                            "APCA_API_SECRET_KEY='literal-$(anything)#value'\n"
                            'IGNORED_SETTING=anything\n', encoding='utf-8')
            os.environ['APCA_API_KEY_ID'] = 'process-key'
            result = alpaca.read_env(path)
            self.assertEqual(result['APCA_API_KEY_ID'], 'process-key')
            self.assertEqual(result['APCA_API_SECRET_KEY'], 'literal-$(anything)#value')
            self.assertNotIn('IGNORED_SETTING', result)

    def test_missing_credentials_send_no_request_or_output(self):
        with workspace_temporary_directory() as directory, patch.dict(os.environ, {}, clear=True):
            output = Path(directory) / 'output'
            with patch.object(requests.Session, 'get') as get, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    alpaca.main(['--env-file', str(Path(directory)/'missing'), '--symbols', 'AAPL',
                                 '--start', '2025-01-02', '--end', '2025-01-03', '--out', str(output)])
                get.assert_not_called()
            self.assertFalse(output.exists())

    def test_dry_run_works_without_credentials_or_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(requests.Session, 'get') as get:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(alpaca.main(['--env-file', 'nonexistent-env-test', '--symbols', 'AAPL',
                    '--start', '2025-01-02', '--end', '2025-01-03', '--dry-run']), 0)
            plan = json.loads(stdout.getvalue())['plan']
            self.assertEqual(plan['timeframe'], '15Min')
            self.assertEqual(plan['feed'], 'sip')
            self.assertNotIn('APCA_API_KEY_ID', stdout.getvalue())
            get.assert_not_called()

    def test_pagination_and_exclusive_new_york_end(self):
        session = Mock()
        session.get.side_effect = [response(bars={'AAPL': [bar()]}, token='page2'),
            response(bars={'AAPL': [bar('2025-01-02T14:45:00Z')]})]
        frame, pages = self.client(session).bars('AAPL', self.start, self.end, 'sip', 'raw')
        self.assertEqual((len(frame), pages), (2, 2))
        request = session.get.call_args
        self.assertEqual(request.args[0], alpaca.ENDPOINT)
        self.assertFalse(request.kwargs['allow_redirects'])
        self.assertEqual(request.kwargs['params']['page_token'], 'page2')
        self.assertEqual(request.kwargs['params']['asof'], '-')
        self.assertEqual(request.kwargs['params']['end'], '2025-01-03T04:59:59.999999+00:00')
        self.assertEqual(frame.bar_end_utc.iloc[0], '2025-01-02T14:45:00+00:00')

    def test_rate_limit_retries_and_no_feed_fallback(self):
        session = Mock()
        session.get.side_effect = [response(429, headers={'Retry-After': '2'}), response()]
        client = self.client(session)
        client.bars('AAPL', self.start, self.end, 'sip', 'raw')
        self.assertEqual(session.get.call_count, 2)
        client.sleep.assert_any_call(2.)
        self.assertTrue(all(call.kwargs['params']['feed']=='sip' for call in session.get.call_args_list))

    def test_auth_errors_do_not_echo_response_or_credentials(self):
        session = Mock()
        result = response(403)
        result.text = 'test-secret test-key'
        session.get.return_value = result
        with self.assertRaises(alpaca.AlpacaError) as error:
            self.client(session).bars('AAPL', self.start, self.end, 'sip', 'raw')
        self.assertNotIn('test-secret', str(error.exception))
        self.assertNotIn('test-key', str(error.exception))
        self.assertEqual(session.get.call_count, 1)

    def test_transport_error_is_redacted(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError('test-secret in transport details')
        client = self.client(session)
        client.retries = 1
        with self.assertRaises(alpaca.AlpacaError) as error:
            client.page({})
        self.assertNotIn('test-secret', str(error.exception))

    def test_repeated_token_rejects_incomplete_download(self):
        session = Mock()
        session.get.side_effect = [response(token='same'), response(token='same')]
        with self.assertRaisesRegex(alpaca.AlpacaError, 'pagination'):
            self.client(session).bars('AAPL', self.start, self.end, 'sip', 'raw')

    def test_invalid_bars_are_rejected(self):
        for values in [[bar(), bar()], [bar('2025-01-02T14:31:00Z')],
                       [bar(close=999.)], [bar('2025-01-03T05:00:00Z')],
                       [dict(bar(), v=float('nan'))]]:
            with self.subTest(values=values):
                with self.assertRaises(alpaca.AlpacaError):
                    alpaca.normalize(values, 'AAPL', self.start, self.end)

    def test_no_bars_remains_explicitly_empty(self):
        frame = alpaca.normalize([], 'OLD', self.start, self.end)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame), alpaca.COLUMNS)

    def test_resume_verifies_hash_before_reusing_chunk(self):
        with workspace_temporary_directory() as directory:
            out, client = Path(directory), Mock()
            client.bars.return_value = (alpaca.normalize([bar()], 'AAPL', self.start, self.end), 1)
            first = alpaca.save_chunk(client, out, 'AAPL', self.start, self.end, 'sip', 'raw')
            second = alpaca.save_chunk(client, out, 'AAPL', self.start, self.end, 'sip', 'raw')
            self.assertEqual(first, second)
            self.assertEqual(client.bars.call_count, 1)
            with (out/first['file']).open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(alpaca.AlpacaError, 'cache chunk changed'):
                alpaca.save_chunk(client, out, 'AAPL', self.start, self.end, 'sip', 'raw')

    def test_historical_symbol_spellings_are_not_replaced(self):
        with workspace_temporary_directory() as directory:
            path = Path(directory)/'members.csv'
            path.write_text('ticker,security_id\nFB,META\nMETA,META\nOLD,OLD\n', encoding='utf-8')
            symbols, source = alpaca.symbols_for(None, path)
            self.assertEqual(symbols, ['FB', 'META', 'OLD', 'QQQ', 'SPY'])
            self.assertEqual(source['sha256'], alpaca.digest(path))

    def test_year_chunks_do_not_overlap_and_preserve_dst(self):
        start, end = alpaca.date_bound('2024-06-01'), alpaca.date_bound('2026-01-01')
        parts = alpaca.chunks(start, end)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], start)
        self.assertEqual(parts[-1][-1], end)
        self.assertEqual(parts[0][-1], parts[1][0])
        self.assertEqual(start.tz_convert('UTC').hour, 4)
        self.assertEqual(end.tz_convert('UTC').hour, 5)

    def test_full_mocked_run_records_empty_coverage_and_resumes(self):
        with workspace_temporary_directory() as directory, patch.dict(os.environ, {}, clear=True):
            env = Path(directory)/'.env'
            env.write_text('APCA_API_KEY_ID=test-key\nAPCA_API_SECRET_KEY=test-secret\n')
            out = Path(directory)/'dataset'
            args = ['--env-file', str(env), '--symbols', 'AAPL', 'OLD', '--start', '2025-01-02',
                    '--end', '2025-01-03', '--out', str(out)]
            values = alpaca.normalize([bar()], 'AAPL', self.start, self.end)
            empty = alpaca.normalize([], 'OLD', self.start, self.end)
            with patch.object(alpaca.AlpacaClient, 'bars', side_effect=[(values, 1), (empty, 1)]) as get:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(alpaca.main(args), 0)
                    self.assertEqual(alpaca.main(args), 0)
                self.assertEqual(get.call_count, 2)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['empty_chunks'], 1)
            self.assertEqual(manifest['download_status'], 'completed_with_review_required')
            self.assertEqual(pd.read_csv(out/'coverage.csv').rows.tolist(), [1, 0])
            self.assertNotIn('test-secret', (out/'manifest.json').read_text())
            self.assertNotIn('test-key', (out/'REQUEST.json').read_text())


if __name__ == '__main__':
    unittest.main()
