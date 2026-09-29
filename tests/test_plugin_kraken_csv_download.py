# Copyright 2023 Neal Chambers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=protected-access

import tracemalloc
from datetime import datetime, timedelta, timezone
from os import listdir, makedirs, path, remove, unlink
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pytest_mock import MockerFixture
from rp2.rp2_decimal import RP2Decimal
from rp2.rp2_error import RP2RuntimeError

from dali.cache import CACHE_DIR
from dali.configuration import Keyword
from dali.historical_bar import HistoricalBar
from dali.in_transaction import InTransaction
from dali.plugin.pair_converter.csv.kraken import Kraken
from dali.transaction_manifest import TransactionManifest

_CACHE_DIRECTORY: str = "output/kraken_test"
_UNIFIED_TEST_FILE: str = "input/USD_OHLCVT_test.zip"

# Last 1 minute candle in the USDTUSD_1.csv of _UNIFIED_TEST_FILE, and a newer one delivered by an update file
_LAST_UNIFIED_ROW: str = "1604448120,1.9999,1.9999,1.9999,1.9999,1999.9999,99\n"
_UPDATE_TIMESTAMP: int = 1604448180
_UPDATE_ROW: str = f"{_UPDATE_TIMESTAMP},2.1111,2.2222,2.0000,2.1234,21.21,21\n"
# Candle for a pair that isn't in _UNIFIED_TEST_FILE
_NEW_PAIR_ROW: str = f"{_UPDATE_TIMESTAMP},13500.1,13500.2,13500.0,13500.1,1.5,3\n"

# Fake Transaction
FAKE_TRANSACTION: InTransaction = InTransaction(
    plugin="Plugin",
    unique_id=Keyword.UNKNOWN.value,
    raw_data="raw",
    timestamp=datetime.fromtimestamp(1504541500, timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
    asset="BTC",
    exchange="Kraken",
    holder="test",
    transaction_type=Keyword.BUY.value,
    spot_price=Keyword.UNKNOWN.value,
    crypto_in="1",
    crypto_fee=None,
    fiat_in_no_fee=None,
    fiat_in_with_fee=None,
    fiat_fee=None,
    notes="notes",
)


def _manifest() -> TransactionManifest:
    return TransactionManifest([FAKE_TRANSACTION], 1, "USD")


def _write_zip(zip_path: Path, csv_files: Dict[str, str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zip_file:
        for file_name, contents in csv_files.items():
            zip_file.writestr(file_name, contents)


def _read_zip(zip_path: Path) -> Dict[str, str]:
    with ZipFile(zip_path) as zip_file:
        return {file_name: zip_file.read(file_name).decode(encoding="utf-8") for file_name in zip_file.namelist()}


@pytest.fixture(name="unified_csv_file")
def unified_csv_file_fixture(tmp_path: Path, mocker: MockerFixture) -> Path:
    # Point every file the plugin touches into tmp_path, so tests never read or clobber a real download or cache.
    # Directories are not created, just like on a first run.
    cache_directory: Path = tmp_path / "kraken"
    csv_directory: Path = cache_directory / "csv"
    unified_csv_file: Path = csv_directory / "Kraken_OHLCVT.zip"

    mocker.patch("dali.cache.CACHE_DIR", str(tmp_path / "dali_cache"))
    mocker.patch.object(Kraken, "_Kraken__CACHE_DIRECTORY", f"{cache_directory}/")
    mocker.patch.object(Kraken, "_Kraken__CSV_DIRECTORY", f"{csv_directory}/")
    mocker.patch.object(Kraken, "_Kraken__UNIFIED_CSV_FILE", str(unified_csv_file))

    # Never block on input(). Declining the download is the default.
    mocker.patch.object(Kraken, "_prompt_download_confirmation", return_value=False)
    mocker.patch.object(Kraken, "_prompt_delete_confirmation", return_value=False)

    return unified_csv_file


class TestKrakenCsvDownload:
    def test_chunking(self, mocker: Any) -> None:
        kraken_csv = Kraken(transaction_manifest=TransactionManifest([FAKE_TRANSACTION], 1, "USD"))

        if not path.exists(_CACHE_DIRECTORY):
            makedirs(_CACHE_DIRECTORY)

        # Flush test cache directory
        for filename in listdir(_CACHE_DIRECTORY):
            file_path = path.join(_CACHE_DIRECTORY, filename)
            try:
                if path.isfile(file_path):
                    unlink(file_path)
            except RP2RuntimeError:
                pass

        cache_path = path.join(CACHE_DIR, kraken_csv.cache_key())
        if path.exists(cache_path):
            remove(cache_path)

        mocker.patch.object(kraken_csv, "_Kraken__CACHE_DIRECTORY", _CACHE_DIRECTORY)
        if not path.exists(_CACHE_DIRECTORY):
            makedirs(_CACHE_DIRECTORY)

        mocker.patch.object(kraken_csv, "_Kraken__UNIFIED_CSV_FILE", "input/USD_OHLCVT_test.zip")

        test_bar: Optional[HistoricalBar] = kraken_csv.find_historical_bar("USDT", "USD", datetime.fromtimestamp(1601856000))
        files: List[str] = listdir(_CACHE_DIRECTORY)

        # Test if proper price was retrieved and file was chunked
        assert test_bar
        assert test_bar.low == RP2Decimal("1.778")
        assert "USDTUSD_1594080000_5.csv.gz" in files
        assert "USDTUSD_1296000000_10080.csv.gz" in files

        test_bar = kraken_csv.find_historical_bar("USDT", "USD", datetime.fromtimestamp(1601683300))

        # Check if price for longer time span was retrieved even though the timestamp doesn't exist in the csv
        # Also that proper price was retrieved from chunked files in the cache folder
        assert test_bar
        assert test_bar.low == RP2Decimal("1.6668")

        test_bars = kraken_csv.find_historical_bars("USDT", "USD", datetime.fromtimestamp(1601856000, timezone.utc), True, "1w")

        assert test_bars
        test_bar = test_bars[0]

        # Test to make sure it only emulates full weeks
        assert len(test_bars) == 5
        assert test_bar.low == RP2Decimal("1.9999")
        assert test_bar.volume == RP2Decimal("5999.9997")


class TestKrakenCsvUpdateFile:
    def test_update_file_is_merged_into_unified_file(self, unified_csv_file: Path, tmp_path: Path) -> None:
        _write_zip(
            unified_csv_file,
            {
                "XBTUSD_1.csv": "1601855760,1.1,1.2,1.0,1.1,10,1\n",
                "ETHUSD_1.csv": "1601855760,2.1,2.2,2.0,2.1,20,2\n",
            },
        )
        update_file: Path = tmp_path / "update.zip"
        _write_zip(
            update_file,
            {
                "XBTUSD_1.csv": "1601855820,1.3,1.4,1.2,1.3,30,3\n",
                "SOLUSD_1.csv": "1601855820,3.1,3.2,3.0,3.1,40,4\n",
            },
        )

        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        # Update rows are appended to existing pairs, new pairs are added and untouched pairs are kept
        assert _read_zip(unified_csv_file) == {
            "XBTUSD_1.csv": "1601855760,1.1,1.2,1.0,1.1,10,1\n1601855820,1.3,1.4,1.2,1.3,30,3\n",
            "ETHUSD_1.csv": "1601855760,2.1,2.2,2.0,2.1,20,2\n",
            "SOLUSD_1.csv": "1601855820,3.1,3.2,3.0,3.1,40,4\n",
        }
        # The update file is removed so the merge isn't repeated on the next run
        assert not update_file.exists()

    def test_update_file_refreshes_already_chunked_pairs(self, unified_csv_file: Path, tmp_path: Path) -> None:
        unified_csv_file.parent.mkdir(parents=True)
        copyfile(_UNIFIED_TEST_FILE, unified_csv_file)

        # First run chunks and caches USDTUSD, whose 1 minute candles end at _LAST_UNIFIED_ROW
        assert Kraken(transaction_manifest=_manifest()).find_historical_bar("USDT", "USD", datetime.fromtimestamp(1601856000, timezone.utc))

        # Second run brings in a newer candle through an update file
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})
        kraken_csv = Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        # The new candle must be found, not a stale lower resolution one from the old chunks
        test_bar: Optional[HistoricalBar] = kraken_csv.find_historical_bar("USDT", "USD", datetime.fromtimestamp(_UPDATE_TIMESTAMP, timezone.utc))
        assert test_bar
        assert test_bar.duration == timedelta(minutes=1)
        assert test_bar.close == RP2Decimal("2.1234")

    def test_update_file_is_kept_when_unified_file_is_missing(self, unified_csv_file: Path, tmp_path: Path) -> None:
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})

        # The user declines to download the unified file (fixture default), so there is nothing to merge into
        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        # Keep the update file so it can be merged on a later run
        assert update_file.exists()
        assert not unified_csv_file.exists()

    def test_unified_file_is_downloaded_before_merging_update_file(self, unified_csv_file: Path, tmp_path: Path, mocker: MockerFixture) -> None:
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})
        mocker.patch.object(Kraken, "_prompt_download_confirmation", return_value=True)
        # Like the real download, this fails if the CSV directory hasn't been created yet
        download = mocker.patch.object(Kraken, "_Kraken__download_unified_csv", side_effect=lambda: copyfile(_UNIFIED_TEST_FILE, unified_csv_file))

        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        download.assert_called_once()
        assert _read_zip(unified_csv_file)["USDTUSD_1.csv"].endswith(_LAST_UNIFIED_ROW + _UPDATE_ROW)
        assert not update_file.exists()

    def test_failed_merge_leaves_unified_file_intact(self, unified_csv_file: Path, tmp_path: Path, mocker: MockerFixture) -> None:
        unified_csv_file.parent.mkdir(parents=True)
        copyfile(_UNIFIED_TEST_FILE, unified_csv_file)
        original_contents: Dict[str, str] = _read_zip(unified_csv_file)
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})

        # Simulate running out of disk space as soon as anything is written into a zip
        open_zip_entry = ZipFile.open

        def open_zip_entry_without_disk_space(zip_file: ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
            if (args[0] if args else kwargs.get("mode", "r")) == "w":
                raise OSError(28, "No space left on device")
            return open_zip_entry(zip_file, name, *args, **kwargs)

        mocker.patch.object(ZipFile, "open", open_zip_entry_without_disk_space)

        with pytest.raises(OSError, match="No space left on device"):
            Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        # The 4+ GB unified file must survive a failed merge, the update file must be kept for a retry
        # and no partially written temporary file may be left behind
        assert _read_zip(unified_csv_file) == original_contents
        assert update_file.exists()
        assert listdir(unified_csv_file.parent) == [unified_csv_file.name]

    def test_update_file_merge_streams_instead_of_loading_into_memory(self, unified_csv_file: Path, tmp_path: Path) -> None:
        # About 10 MB uncompressed, but compresses to almost nothing, so the test stays fast
        large_csv: str = _LAST_UNIFIED_ROW * 200_000
        _write_zip(unified_csv_file, {"USDTUSD_1.csv": large_csv})
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})

        tracemalloc.start()
        try:
            Kraken(transaction_manifest=_manifest(), update_file=str(update_file))
            _, peak_memory = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # The real unified file is 4+ GB, so the merge must never hold a whole CSV in memory
        assert peak_memory < len(large_csv) // 4

    def test_update_rows_start_on_a_new_line(self, unified_csv_file: Path, tmp_path: Path) -> None:
        # The last row of the unified CSV has no trailing newline
        _write_zip(unified_csv_file, {"USDTUSD_1.csv": _LAST_UNIFIED_ROW.rstrip("\n")})
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": _UPDATE_ROW})

        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        assert _read_zip(unified_csv_file)["USDTUSD_1.csv"] == _LAST_UNIFIED_ROW + _UPDATE_ROW

    def test_update_rows_already_in_unified_file_are_skipped(self, unified_csv_file: Path, tmp_path: Path) -> None:
        unified_csv_file.parent.mkdir(parents=True)
        copyfile(_UNIFIED_TEST_FILE, unified_csv_file)
        original_csv: str = _read_zip(unified_csv_file)["USDTUSD_1.csv"]
        # The update overlaps the last two rows of the unified file, e.g. because the unified file was refreshed after the update came out
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"USDTUSD_1.csv": "1604448060,1.9999,1.9999,1.9999,1.9999,1999.9999,99\n" + _LAST_UNIFIED_ROW + _UPDATE_ROW})

        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        assert _read_zip(unified_csv_file)["USDTUSD_1.csv"] == original_csv + _UPDATE_ROW

    def test_merging_the_same_update_file_twice_changes_nothing(self, unified_csv_file: Path, tmp_path: Path) -> None:
        unified_csv_file.parent.mkdir(parents=True)
        copyfile(_UNIFIED_TEST_FILE, unified_csv_file)
        update_file: Path = tmp_path / "update.zip"
        update_csv_files: Dict[str, str] = {"USDTUSD_1.csv": _UPDATE_ROW, "XBTUSD_1.csv": _NEW_PAIR_ROW}
        _write_zip(update_file, update_csv_files)
        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))
        merged_contents: Dict[str, str] = _read_zip(unified_csv_file)

        # E.g. the previous run was interrupted after the merge, but before the update file was deleted
        _write_zip(update_file, update_csv_files)
        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        assert _read_zip(unified_csv_file) == merged_contents

    def test_update_csv_files_in_a_folder_are_matched_by_file_name(self, unified_csv_file: Path, tmp_path: Path) -> None:
        _write_zip(unified_csv_file, {"USDTUSD_1.csv": _LAST_UNIFIED_ROW})
        update_file: Path = tmp_path / "update.zip"
        _write_zip(update_file, {"Kraken_OHLCVT_Q4_2020/USDTUSD_1.csv": _UPDATE_ROW, "Kraken_OHLCVT_Q4_2020/XBTUSD_1.csv": _NEW_PAIR_ROW})

        Kraken(transaction_manifest=_manifest(), update_file=str(update_file))

        # The plugin only looks for CSVs at the top level of the unified file
        assert _read_zip(unified_csv_file) == {"USDTUSD_1.csv": _LAST_UNIFIED_ROW + _UPDATE_ROW, "XBTUSD_1.csv": _NEW_PAIR_ROW}
