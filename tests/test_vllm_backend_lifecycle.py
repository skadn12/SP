import unittest
from unittest.mock import Mock

from scripts.current_method.restart_n5_spell.vllm_backends import (
    VLLMSolverBackend,
)


class VLLMBackendLifecycleTests(unittest.TestCase):
    def test_close_shuts_down_engine_core_and_is_idempotent(self):
        backend = object.__new__(VLLMSolverBackend)
        engine_core = Mock()
        llm_engine = Mock(engine_core=engine_core)
        backend.engine = Mock(llm_engine=llm_engine)

        backend.close()
        backend.close()

        engine_core.shutdown.assert_called_once_with()
        self.assertIsNone(backend.engine)


if __name__ == "__main__":
    unittest.main()
