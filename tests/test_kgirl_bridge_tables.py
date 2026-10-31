import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kgirl_bridge import KGirlBridge


class TableNameTests(unittest.TestCase):
    def bridge(self):
        b = KGirlBridge.__new__(KGirlBridge)
        b.pg_conn = mock.MagicMock()
        return b

    def test_hostile_table_name_never_reaches_sql(self):
        for name in ("x; DROP TABLE users;--", "a b", "1abc", "", "a" * 64):
            b = self.bridge()
            self.assertIn("error", b.pgvector_persistent_store([[1.0]], [{}], table_name=name))
            self.assertIn("error", b.pgvector_similarity_search([1.0], table_name=name))
            b.pg_conn.cursor.assert_not_called()

    def test_valid_table_name_is_used(self):
        b = self.bridge()
        self.assertNotIn("Invalid", str(b.pgvector_persistent_store([[1.0]], [{}], table_name="kgirl_vectors")))
        b.pg_conn.cursor.assert_called()


if __name__ == "__main__":
    unittest.main()
