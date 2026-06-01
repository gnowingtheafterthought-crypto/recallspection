# Contributing to Recallspection Phase 5

All agents and contributors must adhere to the Hive-Mind Consensus standards:
1. **Weighted Inputs**: All distributed memory events must be validated via the `ConsensusEngine`.
2. **Confidence Thresholds**: A minimum 90% confidence score across 3+ nodes is required for Global Truth sync.
3. **Safety First**: All recalls must pass through the `AlignmentSentinel` before being broadcast to the network.
