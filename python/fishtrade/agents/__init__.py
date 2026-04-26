"""LangGraph node functions.

Every public callable here has the signature::

    def node(state: GraphState) -> dict

and returns a *state patch* (a dict of just the keys this node owns).
LangGraph's reducers merge those patches back into the global state, so
parallel nodes never collide on shared dict keys.
"""
