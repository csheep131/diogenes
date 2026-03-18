"""Test result storage with JSONL and SQLite backends."""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from diogenes.model import EpistemicMode


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test execution."""

    # Test identification
    test_id: str
    prompt: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Model response
    response: str = ""
    epistemic_mode: str = ""
    confidence: float = 0.0

    # Token-level data
    tokens: list[int] = field(default_factory=list)
    logprobs: Optional[list[float]] = None

    # Ground truth (if available)
    expected_response: Optional[str] = None
    expected_mode: Optional[str] = None
    is_correct: Optional[bool] = None

    # Metadata
    model_name: str = ""
    model_path: Optional[str] = None
    temperature: float = 0.7
    max_length: int = 512
    suite_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Computed metrics
    latency_ms: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        """Create TestResult from dictionary."""
        return cls(**data)


class TestStorage:
    """Storage backend for test results.

    Supports both JSONL (for simple logging) and SQLite (for queries).
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        backend: str = "jsonl",
    ):
        """Initialize storage.

        Args:
            storage_path: Path to storage file/directory
            backend: Storage backend ('jsonl' or 'sqlite')
        """
        self.backend = backend
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / "test_results"

        if backend == "jsonl":
            self._init_jsonl()
        elif backend == "sqlite":
            self._init_sqlite()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"TestStorage initialized: backend={backend}, path={self.storage_path}")

    def _init_jsonl(self) -> None:
        """Initialize JSONL storage."""
        # Ensure directory exists
        if not self.storage_path.suffix:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.storage_path = self.storage_path / "results.jsonl"
        else:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_sqlite(self) -> None:
        """Initialize SQLite storage."""
        if not self.storage_path.suffix:
            self.storage_path = self.storage_path / "results.db"
        else:
            self.storage_path = self.storage_path.with_suffix(".db")

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database and tables
        self._conn = sqlite3.connect(str(self.storage_path))
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                test_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT,
                epistemic_mode TEXT,
                confidence REAL,
                tokens TEXT,
                logprobs TEXT,
                expected_response TEXT,
                expected_mode TEXT,
                is_correct INTEGER,
                model_name TEXT,
                model_path TEXT,
                temperature REAL,
                max_length INTEGER,
                suite_name TEXT,
                tags TEXT,
                latency_ms REAL,
                token_count INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON test_results(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mode ON test_results(epistemic_mode)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_suite ON test_results(suite_name)
        """)

        self._conn.commit()

    def save(self, result: TestResult) -> None:
        """Save a test result.

        Args:
            result: TestResult to save
        """
        if self.backend == "jsonl":
            self._save_jsonl(result)
        elif self.backend == "sqlite":
            self._save_sqlite(result)

    def _save_jsonl(self, result: TestResult) -> None:
        """Save result to JSONL file."""
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def _save_sqlite(self, result: TestResult) -> None:
        """Save result to SQLite database."""
        cursor = self._conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO test_results (
                test_id, prompt, response, epistemic_mode, confidence,
                tokens, logprobs, expected_response, expected_mode, is_correct,
                model_name, model_path, temperature, max_length, suite_name,
                tags, latency_ms, token_count, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_id,
            result.prompt,
            result.response,
            result.epistemic_mode,
            result.confidence,
            json.dumps(result.tokens),
            json.dumps(result.logprobs) if result.logprobs else None,
            result.expected_response,
            result.expected_mode,
            int(result.is_correct) if result.is_correct is not None else None,
            result.model_name,
            result.model_path,
            result.temperature,
            result.max_length,
            result.suite_name,
            json.dumps(result.tags),
            result.latency_ms,
            result.token_count,
            result.timestamp,
        ))

        self._conn.commit()

    def load(
        self,
        suite_name: Optional[str] = None,
        mode: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TestResult]:
        """Load test results with optional filters.

        Args:
            suite_name: Filter by suite name
            mode: Filter by epistemic mode
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of results

        Returns:
            List of TestResult objects
        """
        if self.backend == "jsonl":
            return self._load_jsonl(suite_name, mode, start_date, end_date, limit)
        elif self.backend == "sqlite":
            return self._load_sqlite(suite_name, mode, start_date, end_date, limit)
        return []

    def _load_jsonl(
        self,
        suite_name: Optional[str] = None,
        mode: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TestResult]:
        """Load results from JSONL file."""
        results = []

        if not self.storage_path.exists():
            return results

        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    result = TestResult.from_dict(data)

                    # Apply filters
                    if suite_name and result.suite_name != suite_name:
                        continue
                    if mode and result.epistemic_mode != mode:
                        continue
                    if start_date and result.timestamp < start_date:
                        continue
                    if end_date and result.timestamp > end_date:
                        continue

                    results.append(result)

                    if limit and len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON line in {self.storage_path}")

        return results

    def _load_sqlite(
        self,
        suite_name: Optional[str] = None,
        mode: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TestResult]:
        """Load results from SQLite database."""
        cursor = self._conn.cursor()

        query = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if suite_name:
            query += " AND suite_name = ?"
            params.append(suite_name)
        if mode:
            query += " AND epistemic_mode = ?"
            params.append(mode)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            result = TestResult(
                test_id=row[0],
                prompt=row[1],
                response=row[2],
                epistemic_mode=row[3],
                confidence=row[4],
                tokens=json.loads(row[5]) if row[5] else [],
                logprobs=json.loads(row[6]) if row[6] else None,
                expected_response=row[7],
                expected_mode=row[8],
                is_correct=bool(row[9]) if row[9] is not None else None,
                model_name=row[10],
                model_path=row[11],
                temperature=row[12],
                max_length=row[13],
                suite_name=row[14],
                tags=json.loads(row[15]) if row[15] else [],
                latency_ms=row[16],
                token_count=row[17],
                timestamp=row[18],
            )
            results.append(result)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        results = self.load()

        mode_counts: dict[str, int] = {}
        total_confidence = 0.0
        total_latency = 0.0

        for result in results:
            mode_counts[result.epistemic_mode] = mode_counts.get(result.epistemic_mode, 0) + 1
            total_confidence += result.confidence
            total_latency += result.latency_ms

        return {
            "total_results": len(results),
            "mode_distribution": mode_counts,
            "avg_confidence": total_confidence / len(results) if results else 0.0,
            "avg_latency_ms": total_latency / len(results) if results else 0.0,
            "storage_path": str(self.storage_path),
            "backend": self.backend,
        }

    def close(self) -> None:
        """Close storage connection."""
        if self.backend == "sqlite" and hasattr(self, "_conn"):
            self._conn.close()
            logger.info("SQLite connection closed")


def save_results(results: list[TestResult], path: str, backend: str = "jsonl") -> None:
    """Convenience function to save multiple results.

    Args:
        results: List of TestResult objects
        path: Storage path
        backend: Storage backend
    """
    storage = TestStorage(path, backend)
    for result in results:
        storage.save(result)
    storage.close()


def load_results(path: str, backend: str = "jsonl", **kwargs) -> list[TestResult]:
    """Convenience function to load results.

    Args:
        path: Storage path
        backend: Storage backend
        **kwargs: Filter arguments

    Returns:
        List of TestResult objects
    """
    storage = TestStorage(path, backend)
    results = storage.load(**kwargs)
    storage.close()
    return results
