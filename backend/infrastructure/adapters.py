from domain.models import ActionChannel, ActionType, action_channel

# The simulated sandbox "completes" an action according to its explicit channel in
# domain.models.ACTION_SPECS - never according to words in the action's name (a
# `"REMINDER" in action` test used to decide this). Anything that is not a known
# payment or communication action FAILS loudly rather than silently succeeding.


class MockExecutionAdapter:
    def execute(self, execution_request: dict) -> dict:
        action = execution_request["action"]
        try:
            channel = action_channel(ActionType(action))
        except ValueError:
            channel = None

        if channel == ActionChannel.PAYMENT:
            return {
                "adapter_status": "COMPLETED_SIMULATED",
                "metadata": {"gateway": "sandbox", "message": "Payment retried successfully."},
            }
        if channel == ActionChannel.COMMUNICATION:
            return {
                "adapter_status": "COMPLETED_SIMULATED",
                "metadata": {"gateway": "mock_twilio", "message": "Message sent."},
            }
        return {
            "adapter_status": "FAILED",
            "metadata": {"error": f"Unsupported action simulation: {action}"},
        }
