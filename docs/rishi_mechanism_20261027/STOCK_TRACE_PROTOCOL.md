# Stock accounting trace — 2026-10-27

Before stock-level outcomes: instrument the unchanged canonical accounting function with Python line tracing. Capture actual share changes, cash flows, costs and daily positions for saved-export B, D, E, Hlabel and E_Hlabel, ten-seed ensembles, top 5 and top 10. These ten cells are fixed; no stock removal search or portfolio selection follows.

Reconcile daily fees, shares, cash and the sum of per-stock net return contributions against the canonical account. Match each daily return to the already reproduced saved-export account within 1e-10. Report annual arithmetic contributions (252 times daily mean), holding days and trading costs; individual stock CAGRs are not additive. Aggregate share classes by issuer when discussing concentration. A trace reconstructs the canonical execution assumptions; it is not an independent execution model or actual broker record. Acquisition share units are economic settlement units.

Freeze outputs in fresh execution_batch3_trace and results/RISHI_MECHANISM_TRACE_20261027 directories. No training, checkpoint tolerance change, protected scoring or capital decision. Original checkpoint regeneration remains unresolved.
