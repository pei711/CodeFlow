import time
from io import StringIO

from codeflow.terminal_ui import ThinkingIndicator


def test_thinking_indicator_renders_and_clears():
    stream = StringIO()
    indicator = ThinkingIndicator(stream=stream, interval=0.01, enabled=True)

    indicator.start()
    time.sleep(0.03)
    indicator.stop()

    output = stream.getvalue()
    assert "CodeFlow 思考中" in output
    assert output.endswith("\r")


def test_thinking_indicator_stays_quiet_when_disabled():
    stream = StringIO()
    indicator = ThinkingIndicator(stream=stream, enabled=False)

    indicator.start()
    indicator.stop()

    assert stream.getvalue() == ""
