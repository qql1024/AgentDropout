from AgentDropout.agents.agent_registry import AgentRegistry

@AgentRegistry.register("FinalRefer")
class FinalRefer:
    pass

print(list(AgentRegistry.keys()))  # Should now contain "FinalRefer"